"""Training loop with periodic validation and metric callbacks."""

from collections.abc import Callable
from math import isfinite
from time import perf_counter

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .data_processing.batching import CodeDataLoader
from .utils.metric_calculation import token_loss_sum
from .utils.mixed_precision import get_grad_scaler
from .utils.ddp_distribution import (
    broadcast,
    get_process_count,
    is_main_process,
    reduce_all,
    sum_metrics_across_processes,
)


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    device: str | torch.device,
    pad_id: int,
    *,
    use_mixed_precision: bool = True,
    max_grad_norm: float | None = 1.0,
    examples_per_mini_epoch: int | None = None,
    scheduler=None,
    max_train_seconds: float | None = None,
    on_validation: Callable[[dict], None] | None = None,
    validation_fn: Callable[[], dict] | None = None,
    examples_seen: int = 0,
    batches_seen: int = 0,
    start_batch: int = 0,
    mini_epoch_in_epoch: int = 0,
    next_validation_examples: int | None = None,
) -> float:
    """Train a full epoch, validating and stepping LR at mini-epoch boundaries.

    Mini-epochs end after the batch crossing each N-example threshold, including
    the last partial mini-epoch. The callback handles metrics and checkpoints.
    model is the training model, optionally wrapped in DDP. validation_fn captures
    the unwrapped model and returns metrics; only the main process calls it.
    Mixed precision uses FP16 on CUDA; CPU and MPS keep full precision.
    max_grad_norm clips the global L2 gradient norm after AMP unscaling;
    None disables clipping. DDP synchronizes gradients during backward first.
    max_train_seconds limits elapsed time, including validation and callbacks.
    The CLI subtracts earlier elapsed time across epochs and resume.
    A time limit finishes the current batch and validates its partial mini-epoch.
    """
    interval = examples_per_mini_epoch
    next_validation = next_validation_examples if interval is not None else None
    if interval is not None and next_validation is None:
        next_validation = examples_seen + interval

    device = torch.device(device)
    amp_enabled = use_mixed_precision and device.type == "cuda"
    scaler = get_grad_scaler(optimizer) if amp_enabled else None
    process_count = get_process_count()

    model.train()
    metric_dtype = torch.float32 if device.type == "mps" else torch.float64
    total_loss = torch.zeros((), dtype=metric_dtype, device=device)
    total_tokens = 0
    interval_loss = torch.zeros_like(total_loss)
    interval_tokens = 0
    interval_examples = 0
    mini_epoch = mini_epoch_in_epoch
    started = mini_started = perf_counter()

    if isinstance(train_loader, CodeDataLoader):
        batches = train_loader.iter_from(start_batch)
        first_batch = start_batch + 1
    else:
        batches = train_loader
        first_batch = 1

    for batch_index, batch in enumerate(batches, start=first_batch):
        if batch_index <= start_batch:
            mini_started = perf_counter()
            continue

        local_tokens = (batch["label_ids"] != pad_id).sum().item()
        global_tokens = local_tokens
        global_examples = batch["label_ids"].size(0)
        encoder_input_ids = batch["encoder_input_ids"].to(device, non_blocking=True)
        decoder_input_ids = batch["decoder_input_ids"].to(device, non_blocking=True)
        label_ids = batch["label_ids"].to(device, non_blocking=True)
        encoder_key_padding_mask = batch["encoder_key_padding_mask"].to(
            device, non_blocking=True
        )
        decoder_key_padding_mask = batch["decoder_key_padding_mask"].to(
            device, non_blocking=True
        )

        optimizer.zero_grad()

        if process_count > 1:
            counts = torch.tensor([local_tokens, global_examples], device=device)
            global_tokens, global_examples = reduce_all(counts).tolist()

        with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                encoder_key_padding_mask=encoder_key_padding_mask,
                decoder_key_padding_mask=decoder_key_padding_mask,
            )

            loss_sum = token_loss_sum(logits, label_ids, pad_id)
            loss = loss_sum * process_count / global_tokens

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        num_tokens = local_tokens
        batch_loss = loss_sum.detach()
        total_loss.add_(batch_loss)
        total_tokens += num_tokens
        interval_loss.add_(batch_loss)
        interval_tokens += num_tokens
        examples_seen += global_examples
        interval_examples += global_examples
        batches_seen += 1

        time_limit_reached = False
        if max_train_seconds is not None:
            time_limit_reached = broadcast(
                perf_counter() - started >= max_train_seconds
                if is_main_process() else None
            )
        validation_boundary = next_validation is not None and (
            examples_seen >= next_validation
            or batch_index == len(train_loader)
        )
        if validation_boundary or time_limit_reached:
            del logits, loss, loss_sum

            train_loss_sum, train_tokens = sum_metrics_across_processes(
                interval_loss, interval_tokens, device
            )

            train_seconds = perf_counter() - mini_started
            val_started = perf_counter()
            validation_metrics = {}
            if is_main_process() and validation_fn is not None:
                validation_metrics = validation_fn()
            validation_metrics = broadcast(validation_metrics)
            val_seconds = perf_counter() - val_started
            model.train()

            mini_epoch += 1
            learning_rate = optimizer.param_groups[0]["lr"]
            if scheduler is not None and validation_boundary:
                scheduler.step()

            if next_validation is not None:
                next_validation += ((examples_seen - next_validation) // interval + 1) * interval
            if on_validation is not None:
                on_validation(
                    {
                        "examples_seen": examples_seen,
                        "batches_seen": batches_seen,
                        "batch_index": batch_index,
                        "mini_epoch_in_epoch": mini_epoch,
                        "mini_epoch_examples": interval_examples,
                        "train_loss": train_loss_sum / train_tokens,
                        "target_tokens": train_tokens,
                        **validation_metrics,
                        "train_seconds": train_seconds,
                        "val_seconds": val_seconds,
                        "mini_epoch_seconds": train_seconds + val_seconds,
                        "seconds_per_example": train_seconds / interval_examples,
                        "next_validation_examples": next_validation,
                        "lr": learning_rate,
                    }
                )

            interval_loss.zero_()
            interval_tokens = 0
            interval_examples = 0
            mini_started = perf_counter()
        if time_limit_reached:
            break

    total_loss, total_tokens = sum_metrics_across_processes(total_loss, total_tokens, device)
    return total_loss / total_tokens if total_tokens else 0.0
