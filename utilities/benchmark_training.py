"""Time the real training loop on Kaggle without validation or checkpoints."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import torch

from lib.experiment import Experiment
from lib.train import train_loop
from lib.utils.ddp_distribution import (
    broadcast,
    gather_from_all_processes,
    get_process_count,
    get_process_index,
    initialize_processes,
    is_main_process,
    shutdown_processes,
    wait_for_all_processes,
)
from lib.experiment.builders import build_objects
from lib.experiment.setup import load_config
from lib.utils.paths import PROJECT_DIR


class TimedBatches:
    """Read real batches until every process receives the same stop signal."""

    def __init__(self, loader, pad_id, device, seconds):
        self.loader = loader
        self.pad_id = pad_id
        self.device = device
        self.seconds = seconds
        self.examples = self.tokens = self.batches = 0
        self.loading_seconds = self.training_seconds = 0.0
        self.started = perf_counter()
        self.next_report = 30.0

    def __iter__(self):
        batch_iterator = None
        while True:
            elapsed = perf_counter() - self.started
            finished = elapsed >= self.seconds
            if is_main_process() and (elapsed >= self.next_report or finished):
                print(
                    f"Measurement: {elapsed:.1f}/{self.seconds:g} s | "
                    f"completed batches per process: {self.batches}",
                    flush=True,
                )
                self.next_report = elapsed + 30.0
            # Every process stops before fetching another batch.
            if broadcast(finished):
                return

            detailed = self.batches < 3
            if detailed:
                print(f"[{self.device}] Loading measurement batch {self.batches + 1}...", flush=True)
            loading_started = perf_counter()
            if batch_iterator is None:
                batch_iterator = iter(self.loader)
            try:
                batch = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(self.loader)
                batch = next(batch_iterator)
            loading_seconds = perf_counter() - loading_started
            self.loading_seconds += loading_seconds
            self.examples += batch["label_ids"].size(0)
            self.tokens += (batch["label_ids"] != self.pad_id).sum().item()
            self.batches += 1
            if detailed:
                print(
                    f"[{self.device}] Batch {self.batches}: loading={loading_seconds:.3f} s | "
                    f"encoder={tuple(batch['encoder_input_ids'].shape)}, "
                    f"decoder={tuple(batch['decoder_input_ids'].shape)} | training...",
                    flush=True,
                )
            training_started = perf_counter()
            yield batch
            torch.cuda.synchronize(self.device)
            training_seconds = perf_counter() - training_started
            self.training_seconds += training_seconds
            if detailed:
                print(
                    f"[{self.device}] Batch {self.batches}: training={training_seconds:.3f} s",
                    flush=True,
                )


def benchmark(state, seconds, use_mixed_precision, warmup_batches):
    def show_precision(model, inputs, logits):
        scaler = getattr(state.optimizer, "_amp_grad_scaler", None)
        print(
            f"process={get_process_index()} | device={state.device} | "
            f"DDP={state.train_model is not state.model} | "
            f"autocast={torch.is_autocast_enabled()} | logits={logits.dtype} | "
            f"GradScaler={scaler is not None and scaler.is_enabled()} | "
            f"loader={type(state.train_loader).__name__} | "
            f"sampler_process={state.train_loader.batch_sampler.process_index}/"
            f"{state.train_loader.batch_sampler.process_count}",
            flush=True,
        )
        hook.remove()

    hook = state.model.register_forward_hook(show_precision)
    training_args = dict(
        model=state.train_model,
        optimizer=state.optimizer,
        device=state.device,
        pad_id=state.pad_id,
        use_mixed_precision=use_mixed_precision,
        max_grad_norm=state.config["training"]["max_grad_norm"],
    )
    if is_main_process():
        print(
            f"GPUs/processes: {get_process_count()} | "
            f"max_tokens per GPU: {state.config['batching']['max_tokens']} | "
            f"CPU threads per process: {torch.get_num_threads()} | "
            f"warmup: {warmup_batches} batches, measurement: {seconds:g} s",
            flush=True,
        )

    def warmup():
        batch_iterator = iter(state.train_loader)
        for index in range(1, warmup_batches + 1):
            if is_main_process():
                print(f"Warmup {index}/{warmup_batches}: loading batch...", flush=True)
            started = perf_counter()
            batch = next(batch_iterator, None)
            if batch is None:
                return
            if is_main_process():
                print(f"Warmup {index}: loading={perf_counter() - started:.3f} s; training...", flush=True)
            started = perf_counter()
            yield batch
            torch.cuda.synchronize(state.device)
            if is_main_process():
                print(f"Warmup {index}: training={perf_counter() - started:.3f} s", flush=True)

    train_loop(train_loader=warmup(), **training_args)

    torch.cuda.synchronize(state.device)
    torch.cuda.reset_peak_memory_stats(state.device)
    wait_for_all_processes()
    if is_main_process():
        print("Warmup finished. Starting measurement.", flush=True)
    batches = TimedBatches(state.train_loader, state.pad_id, state.device, seconds)
    loss = train_loop(train_loader=batches, **training_args)
    torch.cuda.synchronize(state.device)
    elapsed = perf_counter() - batches.started
    per_process = gather_from_all_processes({
        "process_index": get_process_index(),
        "device": str(state.device),
        "seconds": elapsed,
        "examples": batches.examples,
        "target_tokens": batches.tokens,
        "batches": batches.batches,
        "batch_loading_seconds": round(batches.loading_seconds, 3),
        "training_step_seconds": round(batches.training_seconds, 3),
        "peak_memory_mib": round(torch.cuda.max_memory_allocated(state.device) / 2**20),
    })
    if is_main_process():
        elapsed = max(item["seconds"] for item in per_process)
        result = {
            "process_count": get_process_count(),
            "mixed_precision": use_mixed_precision,
            "max_tokens_per_gpu": state.config["batching"]["max_tokens"],
            "seconds": round(elapsed, 2),
            "examples_per_second": round(sum(item["examples"] for item in per_process) / elapsed, 2),
            "target_tokens_per_second": round(sum(item["target_tokens"] for item in per_process) / elapsed, 2),
            "train_loss": loss,
            "per_process": per_process,
        }
        print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiment_configs/baseline_plus_plus_ddp.yaml")
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--data-dir", help="Override the tokenized dataset directory")
    parser.add_argument("--tokenizer-path", help="Override the tokenizer JSON path")
    parser.add_argument("--no-mixed-precision", action="store_true")
    args = parser.parse_args()
    if min(args.seconds, args.max_tokens, args.train_size, args.warmup_batches) <= 0:
        parser.error("seconds, max-tokens, train-size and warmup-batches must be positive")
    if not torch.cuda.is_available() or "LOCAL_RANK" not in os.environ:
        parser.error("Requires CUDA and torchrun --standalone --nproc_per_node=1 or 2")

    # Keep CPU parallelism equal in the one-GPU and two-GPU comparisons.
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    initialize_processes("nccl", device)
    try:
        config = load_config(args.config)
        config["data"].update(train_size=args.train_size, val_size=1)
        config["batching"]["max_tokens"] = args.max_tokens
        if args.data_dir:
            config["data"]["tokenized_dir"] = args.data_dir
            config["data"]["tokenizer_path"] = str(Path(args.data_dir) / "tokenizer.json")
        if args.tokenizer_path:
            config["data"]["tokenizer_path"] = args.tokenizer_path
        state = Experiment(config)
        preparation_started = perf_counter()
        if is_main_process():
            print("Preparing model and data loaders...", flush=True)
        build_objects(state, PROJECT_DIR / config["data"]["tokenizer_path"], device)
        if is_main_process():
            print(f"Preparation: {perf_counter() - preparation_started:.1f} s", flush=True)
        benchmark(state, args.seconds, not args.no_mixed_precision, args.warmup_batches)
    finally:
        shutdown_processes()


if __name__ == "__main__":
    main()
