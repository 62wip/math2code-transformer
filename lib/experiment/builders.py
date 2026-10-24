"""Model, optimizer, scheduler and data loader assembly."""

from collections.abc import Iterable
from pathlib import Path
from random import Random
from typing import Any

import torch
from datasets import Dataset, load_from_disk
from tokenizers import Tokenizer

from ..models.baseline import BaselineTransformerModel
from ..models.baseline_plus_plus import BaselinePlusPlusModel
from ..data_processing.batching import CodeDataLoader
from ..data_processing.evaluation import prepare_evaluation_data
from ..utils.paths import PROJECT_DIR
from ..utils.ddp_distribution import (
    broadcast,
    get_process_index,
    get_process_count,
    is_main_process,
    wrap_model_for_ddp,
)

# Only one core architecture: baseline++
MODELS = {
    "baseline_transformer": BaselineTransformerModel,
    "baseline++": BaselinePlusPlusModel,
}


def build_model(model_config: dict, **runtime_params: Any) -> torch.nn.Module:
    """Fill in runtime model parameters and build the selected model."""

    params = dict(model_config["params"])
    for name, value in runtime_params.items():
        if name in params and params[name] != value:
            raise ValueError(
                f"Configured model.{name} does not match its runtime value."
            )
        params[name] = value

    model_class = MODELS[model_config["type"]]
    return model_class(**params)


def build_optimizer(
    optimizer_config: dict,
    parameters: Iterable[torch.Tensor] | Iterable[dict],
) -> torch.optim.Optimizer:
    """Build the selected PyTorch optimizer for the model parameters."""

    optimizer_class = getattr(torch.optim, optimizer_config["type"])
    return optimizer_class(parameters, **optimizer_config["params"])


def build_scheduler(
    scheduler_config: dict | None,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build the configured schedule, optionally preceded by linear warmup."""

    if scheduler_config is None:
        return None
    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_config["type"])
    scheduler = scheduler_class(optimizer, **scheduler_config["params"])
    warmup = scheduler_config.get("warmup")
    if warmup:
        linear = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup["start_factor"],
            total_iters=warmup["steps"],
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[linear, scheduler],
            milestones=[warmup["steps"]],
        )
    return scheduler


def _sample_train_val_indices(
    data: dict[str, Dataset], config: dict
) -> dict[str, list[int] | None]:
    data_config = config["data"]
    sample_indices = {}
    for split, seed in (
        ("train", config["seed"]),
        ("val", data_config["val_seed"]),
    ):
        requested_size = data_config[f"{split}_size"]
        size = (
            min(requested_size, len(data[split]))
            if requested_size is not None
            else len(data[split])
        )
        sample_indices[split] = (
            None
            if size == len(data[split])
            else Random(seed).sample(range(len(data[split])), size)
        )
    return sample_indices


def _build_heavy_loader(
    val_data: Dataset,
    config: dict,
    tokenizer: Tokenizer,
    pad_id: int,
    sample_indices: list[int] | None,
) -> tuple[CodeDataLoader | None, list[int]]:
    data_config = config["data"]
    heavy_loader = None
    heavy_indices = None
    if is_main_process():
        heavy_data, heavy_indices, _ = prepare_evaluation_data(
            val_data,
            tokenizer,
            sample_size=data_config["heavy_val_size"],
            sample_seed=data_config["val_seed"],
            sample_indices=sample_indices,
        )
        heavy_loader = CodeDataLoader(
            heavy_data,
            pad_id=pad_id,
            shuffle=False,
            max_new_tokens=config["validation"]["max_new_tokens"],
            **config["batching"],
        )
    return heavy_loader, broadcast(heavy_indices)


def build_loaders(
    config: dict,
    tokenizer: Tokenizer,
    sample_indices: dict[str, list[int] | None] | None = None,
) -> tuple[
    CodeDataLoader, CodeDataLoader, CodeDataLoader | None, dict[str, list[int] | None]
]:
    """Build train, small and heavy loaders, preserving saved indices."""

    data_config = config["data"]
    pad_id = tokenizer.token_to_id("<pad>")
    data = load_from_disk(str(PROJECT_DIR / data_config["tokenized_dir"]))
    if sample_indices is None:
        sample_indices = _sample_train_val_indices(data, config)

    heavy_loader, sample_indices["heavy_val"] = _build_heavy_loader(
        data["val"], config, tokenizer, pad_id, sample_indices.get("heavy_val")
    )

    loaders = {}
    max_seq_len = config["model"]["params"].get("max_seq_len")
    for split in ("train", "val"):
        indices = sample_indices[split]
        data_subset = data[split] if indices is None else data[split].select(indices)
        sequence_lengths = _check_lengths(data_subset, split, max_seq_len)

        data_config[f"{split}_size"] = len(data_subset)
        loaders[split] = CodeDataLoader(
            data_subset,
            pad_id=pad_id,
            shuffle=(split == "train"),
            seed=config["seed"],
            process_index=get_process_index() if split == "train" else 0,
            process_count=get_process_count() if split == "train" else 1,
            sequence_lengths=sequence_lengths,
            **config["batching"],
        )
    return loaders["train"], loaders["val"], heavy_loader, sample_indices


def _check_lengths(dataset: Dataset, split: str, max_seq_len: int | None) -> dict:
    if len(dataset) == 0:
        raise ValueError(f"{split} is empty.")
    lengths = dataset.with_format("numpy", columns=["input_length", "target_length"])[:]
    inputs = lengths["input_length"]
    targets = lengths["target_length"] - 1
    if inputs.min() <= 0 or targets.min() <= 0:
        raise ValueError(f"{split} contains an empty input or target.")
    if max_seq_len is not None and max(inputs.max(), targets.max()) > max_seq_len:
        raise ValueError(f"{split} exceeds max_seq_len={max_seq_len}.")
    return lengths


def get_device(device: str | torch.device) -> str | torch.device:
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    return device


def build_objects(
    state,
    tokenizer_path: str | Path,
    device: str | torch.device,
    sample_indices: dict[str, list[int] | None] | None = None,
) -> None:
    """Use the same assembly for a new run and a restored run."""

    state.device = torch.device(get_device(device))
    torch.manual_seed(state.config["seed"])

    state.tokenizer = Tokenizer.from_file(str(tokenizer_path))
    state.pad_id = state.tokenizer.token_to_id("<pad>")
    if state.pad_id is None:
        raise ValueError("Tokenizer has no <pad> token.")

    state.model = build_model(
        state.config["model"],
        vocab_size=state.tokenizer.get_vocab_size(),
        pad_id=state.pad_id,
    ).to(state.device)

    state.train_model = wrap_model_for_ddp(state.model, state.device)
    if get_process_count() > 1:
        torch.manual_seed(state.config["seed"] + get_process_index())

    state.optimizer = build_optimizer(
        state.config["optimizer"], state.model.parameters()
    )

    state.train_loader, state.val_loader, state.heavy_loader, state.sample_indices = (
        build_loaders(state.config, state.tokenizer, sample_indices)
    )
    state.heavy_options = dict(state.config["validation"])
    state.heavy_options.pop("max_new_tokens")
    state.details["validation"] = {
        **state.config["validation"],
        "max_tokens": state.config["batching"]["max_tokens"],
        "bucket_width": state.config["batching"].get("bucket_width", 16),
        "heavy_val_size": len(state.sample_indices["heavy_val"]),
        "val_seed": state.config["data"]["val_seed"],
    }
