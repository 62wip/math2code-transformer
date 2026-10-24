from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def atomic_write(path: str | Path) -> Iterator[Path]:
    """Provide a temporary path and replace the target only on successful exit."""

    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        yield temporary_path
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Replace a UTF-8 text file only after its temporary copy is fully written."""

    with atomic_write(path) as temporary_path:
        temporary_path.write_text(text, encoding="utf-8")


def write_dict(path: str | Path, data: dict) -> None:
    """Atomically save a dictionary in PyTorch's binary format."""

    with atomic_write(path) as temporary_path:
        torch.save(data, temporary_path)
