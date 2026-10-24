import os
from pathlib import Path


PROJECT_DIR = Path(
    os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[2])
).expanduser().resolve()
EXPERIMENT_RUNS_DIR = PROJECT_DIR / "experiment_runs"


def create_run_dir(name: str) -> Path:
    """Create a new experiment directory and fail if it already exists."""

    run_dir = EXPERIMENT_RUNS_DIR / name
    run_dir.mkdir(parents=True)
    return run_dir
