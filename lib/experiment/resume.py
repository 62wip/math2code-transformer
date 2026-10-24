"""Model weights, random generators and full experiment restoration."""

import json
from pathlib import Path

import torch

from ..utils.metrics import save_metrics
from ..utils.mixed_precision import load_grad_scaler_state_dict
from ..utils.ddp_distribution import (
    get_process_count,
    get_process_index,
    is_main_process,
    wait_for_all_processes,
)
from ..utils.paths import PROJECT_DIR
from .builders import build_objects, build_scheduler

def restore_model_weights(state, checkpoint_path: str | Path) -> None:
    """Load compatible model weights for a new experiment."""
    path = PROJECT_DIR / checkpoint_path
    tokenizer_path = path.parent.parent / "init_state" / "tokenizer.json"
    if json.loads(tokenizer_path.read_text(encoding="utf-8")) != json.loads(
        state.tokenizer.to_str()
    ):
        raise ValueError("The new experiment must use the checkpoint's tokenizer.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state.model.load_state_dict(saved["model_state_dict"])


def restore_rng_state(checkpoint: dict, device: torch.device) -> None:
    torch.set_rng_state(checkpoint["torch_rng_state"])
    saved_device = checkpoint["device"]
    if device.type == "cuda" and saved_device.startswith("cuda"):
        rng = checkpoint["device_rng_state"]
        if isinstance(rng, list):
            torch.cuda.set_rng_state_all(rng)
        else:
            torch.cuda.set_rng_state(rng, device)
    elif device.type == "mps" and saved_device == "mps":
        torch.mps.set_rng_state(checkpoint["device_rng_state"])


def restore_training_state(state, saved: dict) -> None:
    """Restore weights, the scheduler and optimizer settings."""

    state.model.load_state_dict(saved["model_state_dict"])

    scheduler_config = state.config.get("scheduler")
    scheduler_state = saved["scheduler_state_dict"]
    if (scheduler_config is None) != (scheduler_state is None):
        raise ValueError(
            "The run config and checkpoint disagree on whether a scheduler is enabled."
        )
    state.scheduler = build_scheduler(scheduler_config, state.optimizer)
    if state.scheduler is not None:
        state.scheduler.load_state_dict(scheduler_state)

    state.optimizer.load_state_dict(saved["optimizer_state_dict"])
    load_grad_scaler_state_dict(state.optimizer, saved.get("grad_scaler_state_dict"))


def resume_experiment(
    cls: type,
    run_dir: str | Path,
    *,
    device: str | torch.device = "auto",
    checkpoint: str = "last",
    max_tokens: int | None = None,
    examples_per_mini_epoch: int | None = None,
):
    from .setup import check_config, load_config, save_config

    run_dir = PROJECT_DIR / run_dir
    config = load_config(run_dir / "config.yaml")
    saved = torch.load(
        run_dir / "checkpoints" / f"{checkpoint}.pt",
        map_location="cpu",
        weights_only=True,
    )

    saved_process_count = saved.get("process_count", 1)
    if saved_process_count != get_process_count():
        raise ValueError(
            "Resume requires the same process_count. Start a new run with weights_from to change the number of GPUs."
        )

    if max_tokens is not None:
        config["batching"]["max_tokens"] = max_tokens
    if examples_per_mini_epoch is not None:
        config["training"]["examples_per_mini_epoch"] = examples_per_mini_epoch

    check_config(config)
    state = cls(config, run_dir)

    init_dir = run_dir / "init_state"
    sample_indices = json.loads(
        (init_dir / "sample_indices.json").read_text(encoding="utf-8")
    )

    build_objects(state, init_dir / "tokenizer.json", device, sample_indices)
    restore_training_state(state, saved)

    state.history = saved["history"]
    state.validation_history = saved["validation_history"]
    state.details = {**saved["details"], **state.details}
    state.best_val_loss = saved["best_val_loss"]

    selection = state.config["checkpoint"]
    if (
        saved.get("checkpoint_metric", "val_loss") == selection["metric"]
        and saved.get("checkpoint_mode", "min") == selection["mode"]
    ):
        state.best_metric_value = saved.get("best_metric_value", saved["best_val_loss"])
    state.progress = saved["progress"]

    if config["batching"]["max_tokens"] != saved["max_tokens"]:
        state.progress.update(
            start_batch=0, mini_epoch_in_epoch=0, next_validation_examples=None
        )
    elif (
        config["training"]["examples_per_mini_epoch"]
        != saved["examples_per_mini_epoch"]
    ):
        state.progress["next_validation_examples"] = None

    state.train_loader.set_epoch(state.completed_epochs)
    restore_rng_state(
        saved["rng_states"][get_process_index()] if "rng_states" in saved else saved,
        state.device,
    )

    if is_main_process():
        save_config(state)
        save_metrics(state)
    wait_for_all_processes()
    return state
