"""Experiment configuration and initialization of a new run."""

import inspect
import json
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

import torch
import yaml

from ..utils.files import atomic_write_text
from ..utils.ddp_distribution import (
    broadcast,
    get_process_count,
    is_main_process,
    wait_for_all_processes,
)
from ..utils.paths import PROJECT_DIR, EXPERIMENT_RUNS_DIR, create_run_dir
from .builders import MODELS, build_objects, build_scheduler
from .resume import restore_model_weights

def check_config(config: dict) -> None:
    """Check constructor argument names against the installed classes."""

    model = config["model"]
    model_class = MODELS[model["type"]]
    inspect.signature(model_class).bind_partial(**model["params"])

    optimizer = config["optimizer"]
    optimizer_class = getattr(torch.optim, optimizer["type"])
    inspect.signature(optimizer_class).bind([], **optimizer["params"])

    scheduler = config.get("scheduler")
    if scheduler is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler["type"])
        inspect.signature(scheduler_class).bind(None, **scheduler["params"])
        inspect.signature(scheduler_class.step).bind(None)

    config["data"].setdefault("heavy_val_size", 128)

    checkpoint = config.setdefault("checkpoint", {})
    checkpoint.setdefault("metric", "pass_at_1")
    checkpoint.setdefault("mode", "max")

    validation = config.setdefault("validation", {})
    defaults = {
        "max_new_tokens": 128,
        "beam_top_k": 8,
        "timeout": 2.0,
        "workers": 4,
        "rel_tol": 1e-6,
        "abs_tol": 1e-6,
        "repetition_ngram_size": 4,
    }
    for name, value in defaults.items():
        validation.setdefault(name, value)


def load_config(path: str | Path) -> dict:
    """Load experiment YAML from the project and check constructor arguments."""

    with (PROJECT_DIR / path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    check_config(config)
    return config


def save_config(state) -> None:
    atomic_write_text(
        state.run_dir / "config.yaml",
        yaml.safe_dump(state.config, allow_unicode=True, sort_keys=False),
    )


def setup_experiment(
    cls: type,
    config_path: str | Path,
    *,
    weights_from: str | Path | None = None,
    device: str | torch.device = "auto",
    details: dict | None = None,
):
    config = load_config(config_path)

    run_dir = EXPERIMENT_RUNS_DIR / config["name"]
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    state = cls(config, run_dir)
    build_objects(state, PROJECT_DIR / config["data"]["tokenizer_path"], device)

    if weights_from is not None:
        restore_model_weights(state, weights_from)

    scheduler_config = state.config.get("scheduler")
    if (
        scheduler_config is not None
        and scheduler_config["params"].get("T_max") == "auto"
    ):
        training = state.config["training"]
        total_steps = training["num_epochs"] * ceil(
            state.config["data"]["train_size"] / training["examples_per_mini_epoch"]
        )
        warmup_steps = (scheduler_config.get("warmup") or {}).get("steps", 0)
        scheduler_config["params"]["T_max"] = total_steps - warmup_steps
    state.scheduler = build_scheduler(scheduler_config, state.optimizer)

    state.details = broadcast(
        {
            **(details or {}),
            **state.details,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "process_count": get_process_count(),
        }
    )

    if is_main_process():
        create_run_dir(config["name"])
        for folder in ("init_state", "checkpoints", "metrics", "results"):
            (run_dir / folder).mkdir()

        state.tokenizer.save(str(run_dir / "init_state" / "tokenizer.json"))
        atomic_write_text(
            run_dir / "init_state" / "sample_indices.json",
            json.dumps(state.sample_indices) + "\n",
        )

        save_config(state)
    wait_for_all_processes()
    return state
