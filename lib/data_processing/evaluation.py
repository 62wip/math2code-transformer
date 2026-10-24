"""Prepare question subsets and reference programs for evaluation."""

import math
from collections.abc import Sequence
from random import Random

from datasets import Dataset
from tokenizers import Tokenizer


def prepare_evaluation_data(
    val_data: Dataset,
    tokenizer: Tokenizer,
    *,
    sample_size: int | None = 128,
    sample_seed: int = 62,
    sample_indices: Sequence[int] | None = None,
) -> tuple[Dataset, list[int], dict[str, int]]:
    """Return unique questions, saved indices and counts of excluded rows."""

    question_rows = {}
    for index, row in enumerate(val_data.select_columns(["input_ids", "answer"])):
        answer = row["answer"]
        if answer is None or isinstance(answer, bool) or not math.isfinite(answer):
            continue
        question_rows.setdefault(tuple(row["input_ids"]), []).append(index)

    if sample_indices is None:
        indices = [rows[0] for rows in question_rows.values()]
        if sample_size is not None:
            indices = Random(sample_seed).sample(
                indices, min(sample_size, len(indices))
            )
    else:
        indices = list(sample_indices)
    if not indices:
        raise ValueError(
            "Needs at least one question with a numeric answer."
        )

    selected = val_data.select(indices)
    references = []
    for input_ids in selected["input_ids"]:
        target_ids = val_data.select(question_rows[tuple(input_ids)])["target_ids"]
        codes = tokenizer.decode_batch(list(target_ids), skip_special_tokens=True)
        references.append(list(dict.fromkeys(codes)))

    valid_rows = sum(len(rows) for rows in question_rows.values())
    skipped = {
        "skipped_invalid_answers": len(val_data) - valid_rows,
        "skipped_duplicates": valid_rows - len(question_rows),
    }
    return selected.add_column("references", references), indices, skipped
