"""Saving last and best checkpoints with training and random generator states."""

from math import isfinite

import torch

from ..utils.metrics import save_metrics, serialize_metrics
from ..utils.files import write_dict
from ..utils.mixed_precision import grad_scaler_state_dict
from ..utils.ddp_distribution import (
    gather_from_all_processes,
    get_process_count,
    is_main_process,
    wait_for_all_processes,
)

def capture_rng_state(device: torch.device) -> dict:
    state = {"torch_rng_state": torch.get_rng_state(), "device": str(device)}
    if device.type == "cuda":
        state["device_rng_state"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["device_rng_state"] = torch.mps.get_rng_state()
    return state


def save_checkpoint(state) -> None:
    local_rng = capture_rng_state(state.device)
    rng_states = gather_from_all_processes(local_rng)
    rows = state.validation_history
    best_loss = state.best_val_loss
    selection = state.config.get("checkpoint", {})
    metric = selection.get("metric", "pass_at_1")
    mode = selection.get("mode", "max")
    best_value = state.best_metric_value
    is_best = False
    if rows:
        value = rows[-1].get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"Checkpoint metric {metric!r} must be present and finite.")
        is_best = best_value is None or (
            value < best_value if mode == "min" else value > best_value
        )
        if is_best:
            best_value = value
        loss = rows[-1]["val_loss"]
        best_loss = loss if best_loss is None else min(best_loss, loss)
    state.best_val_loss = best_loss
    state.best_metric_value = best_value
    if not is_main_process():
        wait_for_all_processes()
        return
    metrics = serialize_metrics(state, best_loss)

    saved = {
        "model_state_dict": state.model.state_dict(),
        "optimizer_state_dict": state.optimizer.state_dict(),
        "grad_scaler_state_dict": grad_scaler_state_dict(state.optimizer),
        "scheduler_state_dict": state.scheduler.state_dict() if state.scheduler is not None else None,
        "history": state.history,
        "validation_history": state.validation_history,
        "details": state.details,
        "best_val_loss": best_loss,
        "checkpoint_metric": metric,
        "checkpoint_mode": mode,
        "best_metric_value": best_value,
        "progress": state.progress,
        "max_tokens": state.config["batching"]["max_tokens"],
        "examples_per_mini_epoch": state.config["training"]["examples_per_mini_epoch"],
        "process_count": get_process_count(),
        "rng_states": rng_states,
        "bucket_width": state.config["batching"].get("bucket_width", 16),
        "sampler_seed": state.config["seed"],
        **local_rng,
    }

    write_dict(state.run_dir / "checkpoints" / "last.pt", saved)
    if is_best:
        write_dict(state.run_dir / "checkpoints" / "best.pt", saved)
    save_metrics(state, serialized=metrics)
    wait_for_all_processes()
