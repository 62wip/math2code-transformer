"""Training metrics, epoch summaries, JSON persistence and notebook reporting."""

import json

from .files import atomic_write_text
from .ddp_distribution import is_main_process

VALIDATION_KEYS = (
    "val_loss",
    "val_token_accuracy",
    "val_top5_accuracy",
    "val_target_tokens",
    "val_correct_tokens",
    "val_top5_correct_tokens",
    "pass_at_1",
    "syntax_valid_rate",
    "numeric_return_rate",
    "no_eos_rate",
    "bleu",
    "bleu_signature",
    "repetition_rate",
    "timeout_rate",
    "correct_programs_count",
    "total",
    "syntax_valid_count",
    "numeric_returns_count",
    "no_eos_count",
    "timeouts_count",
)


def serialize_metrics(state, best_val_loss) -> dict:
    """Validate and serialize all metrics before any checkpoint is replaced."""

    summary = {
        **state.details,
        "completed_epochs": state.completed_epochs,
        "completed_mini_epochs": len(state.validation_history),
        "examples_seen": state.progress["examples_seen"],
        "best_val_loss": best_val_loss,
        "checkpoint_metric": state.config.get("checkpoint", {}).get(
            "metric", "pass_at_1"
        ),
        "checkpoint_mode": state.config.get("checkpoint", {}).get("mode", "max"),
        "best_metric_value": state.best_metric_value,
        "last_validation": state.validation_history[-1]
        if state.validation_history
        else None,
    }

    contents = {
        "metrics/epochs_history.json": state.history,
        "metrics/examples_history.json": state.validation_history,
        "results/summary.json": summary,
    }

    return {
        name: json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        for name, value in contents.items()
    }


def save_metrics(state, *, serialized: dict | None = None) -> None:
    if not is_main_process():
        return
    if serialized is None:
        serialized = serialize_metrics(state, state.best_val_loss)

    for name, contents in serialized.items():
        path = state.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, contents)


def summarize_epoch(validation_history: list[dict], epoch: int) -> dict:
    """Summarize an epoch's work, including any replay after a batch-size change."""

    rows = [row for row in validation_history if row["epoch"] == epoch]
    last = rows[-1]
    tokens = sum(row["target_tokens"] for row in rows)
    examples = sum(row["mini_epoch_examples"] for row in rows)
    train_seconds = sum(row["train_seconds"] for row in rows)

    return {
        "epoch": epoch,
        "examples_seen": last["examples_seen"],
        "batches_seen": last["batches_seen"],
        "train_loss": (
            sum(row["train_loss"] * row["target_tokens"] for row in rows) / tokens
        ),
        **{key: last[key] for key in VALIDATION_KEYS if key in last},
        "lr": last["lr"],
        "train_seconds": train_seconds,
        "val_seconds": sum(row["val_seconds"] for row in rows),
        "small_validation_seconds": sum(
            row.get("small_validation_seconds", 0.0) for row in rows
        ),
        "heavy_validation_seconds": sum(
            row.get("heavy_validation_seconds", 0.0) for row in rows
        ),
        "seconds_per_example": train_seconds / examples,
    }


def format_metrics(metrics: dict) -> str:
    mini = metrics["mini_epoch_in_epoch"]
    message = (
        f"Epoch {metrics['epoch']}, mini-epoch {mini} | "
        f"train loss = {metrics['train_loss']:.4f}, val loss = {metrics['val_loss']:.4f}\n"
        f"Examples: {metrics['examples_seen']:,} | "
        f"time = {metrics['mini_epoch_seconds']:.1f} s "
        f"(train {metrics['train_seconds']:.1f}, val {metrics['val_seconds']:.1f}) | "
        f"train/example = {metrics['seconds_per_example']:.6f} s"
    )
    if "val_token_accuracy" in metrics:
        message += (
            f"\nSmall validation: token accuracy = {metrics['val_token_accuracy']:.2%}, "
            f"top-5 = {metrics['val_top5_accuracy']:.2%}"
        )
    if "pass_at_1" in metrics:
        message += (
            f"\nHeavy validation: pass@1 = {metrics['pass_at_1']:.2%} "
            f"({metrics['correct_programs_count']} / {metrics['total']}) | "
            f"time = {metrics['heavy_validation_seconds']:.1f} s"
        )
        if "syntax_valid_rate" in metrics:
            message += (
                f"\nSyntax valid = {metrics['syntax_valid_rate']:.2%}, "
                f"numeric return = {metrics['numeric_return_rate']:.2%}, "
                f"no EOS = {metrics['no_eos_rate']:.2%}\n"
                f"BLEU = {metrics['bleu']:.2f}, "
                f"repetition = {metrics['repetition_rate']:.2%}, "
                f"timeout = {metrics['timeout_rate']:.2%}"
            )
    return message


def plot_losses(state):
    """Return a figure; displaying and saving it is the caller's choice."""
    import matplotlib.pyplot as plt

    by_examples = {row["examples_seen"]: row for row in state.history}
    by_examples.update({row["examples_seen"]: row for row in state.validation_history})
    examples = sorted(by_examples)
    rows = [by_examples[count] for count in examples]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(examples, [row["train_loss"] for row in rows], marker="o", label="Train")
    ax.plot(examples, [row["val_loss"] for row in rows], marker="o", label="Validation")
    ax.set(
        xlabel="Training examples", ylabel="Loss", title="Training and validation loss"
    )
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig
