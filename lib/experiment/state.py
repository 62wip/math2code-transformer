"""Experiment objects, initialization and checkpoint restoration."""

from copy import deepcopy
from pathlib import Path

from ..utils.metrics import save_metrics
from .checkpoint import save_checkpoint
from .resume import resume_experiment
from .setup import setup_experiment


class Experiment:
    """Training objects and their saved state; the training loop lives outside."""

    def __init__(self, config: dict, run_dir: Path | None = None) -> None:
        self.config = deepcopy(config)
        self.run_dir = run_dir

        # Both references share weights; only train_model may be wrapped in DDP.
        self.model = self.train_model = None
        self.optimizer = self.scheduler = None
        self.tokenizer = self.pad_id = self.device = None
        self.train_loader = self.val_loader = self.sample_indices = None
        self.heavy_loader = self.heavy_options = None

        self.details = {}
        self.history = []
        self.validation_history = []
        self.best_val_loss = None
        self.best_metric_value = None

        # These keys can be passed directly to train_loop as **state.progress.
        self.progress = {
            "examples_seen": 0,
            "batches_seen": 0,
            "start_batch": 0,
            "mini_epoch_in_epoch": 0,
            "next_validation_examples": None,
        }

    @classmethod
    def setup(
        cls: type["Experiment"],
        config_path: str | Path,
        *,
        weights_from: str | Path | None = None,
        device: str = "auto",
        details: dict | None = None,
    ) -> "Experiment":
        """Create a new run, optionally starting from another run's model weights."""
        return setup_experiment(
            cls, config_path, weights_from=weights_from, device=device, details=details
        )

    @classmethod
    def resume(
        cls: type["Experiment"],
        run_dir: str | Path,
        *,
        device: str = "auto",
        checkpoint: str = "last",
        max_tokens: int | None = None,
        examples_per_mini_epoch: int | None = None,
    ) -> "Experiment":
        """Restore a run; a changed token budget restarts the unfinished epoch."""
        return resume_experiment(
            cls,
            run_dir,
            device=device,
            checkpoint=checkpoint,
            max_tokens=max_tokens,
            examples_per_mini_epoch=examples_per_mini_epoch,
        )

    @property
    def completed_epochs(self) -> int:
        return len(self.history)

    def record_validation(self, metrics: dict) -> None:
        """Record validation and update the position for the next training call."""

        row = deepcopy(metrics)
        row.setdefault("epoch", self.completed_epochs + 1)
        row.setdefault("mini_epoch", len(self.validation_history) + 1)

        self.validation_history.append(row)
        epoch_finished = row["batch_index"] == len(self.train_loader)
        self.progress = {
            "examples_seen": row["examples_seen"],
            "batches_seen": row["batches_seen"],
            "start_batch": 0 if epoch_finished else row["batch_index"],
            "mini_epoch_in_epoch": 0 if epoch_finished else row["mini_epoch_in_epoch"],
            "next_validation_examples": None if epoch_finished else row["next_validation_examples"],
        }

    def save_checkpoint(self) -> None:
        """Save last, update best on improvement, and write readable metrics."""
        save_checkpoint(self)

    def save_results(self, *, details: dict | None = None) -> None:
        """Optional metadata update; checkpoint saving already writes a summary."""
        self.details.update(details or {})
        save_metrics(self)
