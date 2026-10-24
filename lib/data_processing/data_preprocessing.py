"""Dataset preparation, question splits and validation reference answers."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO
import json
import math
from pathlib import Path
import random
import tempfile
import tokenize

from datasets import Dataset, Features, Value, load_dataset
from tqdm.auto import tqdm

from ..utils.execution_check import execute_solutions
from ..utils.paths import PROJECT_DIR


@dataclass(frozen=True)
class PreprocessingConfig:
    # Data source and output directory
    dataset_name: str = "SlavaYus/tinysms_qa_12m"
    revision: str | None = None
    source_split: str = "train"
    output_dir: str = "data/prepared"
    # Length filters for original examples
    max_question_chars: int = 1000
    max_code_chars: int = 1200
    # Split by unique questions
    val_questions: int = 7500
    seed: int = 62
    # Preprocessing batch size and parallelism
    batch_size: int = 10_000
    num_proc: int = 4
    # Validation reference execution in local Python processes
    execution_timeout: float = 2.0
    execution_workers: int = 8
    # Tolerances for numeric answer comparison
    answer_rel_tol: float = 1e-6
    answer_abs_tol: float = 1e-8


def remove_comments(code: str) -> str:
    lines = code.split("\n")
    for token in tokenize.generate_tokens(StringIO(code).readline):
        if token.type == tokenize.COMMENT:
            row, column = token.start
            before = lines[row - 1][:column].rstrip()
            after = lines[row - 1][token.end[1] :]
            lines[row - 1] = before + after if before else None
    return "\n".join(line for line in lines if line is not None)


def clean_example(example: dict, *, config: PreprocessingConfig) -> dict:
    question, code = example["question"], example["code"]
    if (
        question is None
        or code is None
        or not question.strip()
        or not code.strip()
        or len(question) > config.max_question_chars
        or len(code) > config.max_code_chars
    ):
        return {"code_without_comments": None}
    try:
        cleaned = remove_comments(code)
    except (tokenize.TokenError, SyntaxError):
        return {"code_without_comments": None}
    return {"code_without_comments": cleaned if cleaned.strip() else None}


def deduplicate(dataset: Dataset) -> Dataset:
    existed_examples = set()

    def is_unique(example):
        example = (example["question"], example["code_without_comments"])
        if example in existed_examples:
            return False
        existed_examples.add(example)
        return True

    return dataset.filter(
        is_unique,
        load_from_cache_file=False,
        desc="Removing duplicate pairs",
    )


def split_by_question(
    dataset: Dataset,
    *,
    val_questions: int,
    seed: int,
    work_dir: Path,
    batch_size: int,
) -> tuple[dict[str, Dataset], list[str]]:
    questions = sorted(dataset.unique("question"))
    assert len(questions) > val_questions, (
        f"Need more than {val_questions} unique cleaned questions; "
        f"found {len(questions)}. No prepared dataset was saved."
    )
    selected_val = set(random.Random(seed).sample(questions, val_questions))

    splits = {}
    for name in ("train", "val"):
        splits[name] = dataset.filter(
            lambda questions: [
                (question in selected_val) == (name == "val") for question in questions
            ],
            input_columns=["question"],
            batched=True,
            batch_size=batch_size,
            cache_file_name=str(work_dir / f"{name}_indices.arrow"),
            load_from_cache_file=False,
            desc=f"Selecting {name} by question",
        )
    return splits, sorted(selected_val)


def evaluate_validation(
    dataset: Dataset,
    *,
    config: PreprocessingConfig,
) -> Dataset:
    """Add a shared answer per question; None if numeric returns conflict or are absent."""
    results = execute_solutions(
        dataset["code"],
        timeout=config.execution_timeout,
        workers=config.execution_workers,
    )

    by_question = defaultdict(list)
    for question, value in zip(
        dataset["question"],
        tqdm(results, total=len(dataset), desc="Executing validation references"),
    ):
        values = by_question[question]
        if value is not None:
            values.append(value)

    answers = {}
    for question, values in by_question.items():
        if values and math.isclose(
            min(values),
            max(values),
            rel_tol=config.answer_rel_tol,
            abs_tol=config.answer_abs_tol,
        ):
            answers[question] = values[0]
        else:
            answers[question] = None

    return dataset.add_column(
        "answer",
        [answers[question] for question in dataset["question"]],
        feature=Value("float64"),
    )


def save_splits(
    splits: dict[str, Dataset],
    *,
    output_dir: Path,
    validation_questions: list[str],
    metadata: dict,
    batch_size: int,
) -> None:
    # Save dataset splits
    for name, split in splits.items():
        split.to_parquet(output_dir / f"{name}.parquet", batch_size=batch_size)

    # Save validation questions
    (output_dir / "val_questions.json").write_text(
        json.dumps(validation_questions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Save metadata
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_data(
    config: PreprocessingConfig | None = None,
    *,
    dataset: Dataset | None = None,
) -> dict:
    """Clean, deduplicate, split, execute val references and save; return processing statistics."""

    config = config or PreprocessingConfig()

    output = (PROJECT_DIR / Path(config.output_dir).expanduser()).resolve()
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new output_dir."
        )

    if dataset is None:
        dataset = load_dataset(
            config.dataset_name,
            split=config.source_split,
            revision=config.revision,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        work = staging / "work"
        result = staging / "result"
        work.mkdir()
        result.mkdir()

        features = Features(
            {
                name: Value("string")
                for name in ("question", "code", "code_without_comments")
            }
        )

        # Processing pipeline
        cleaned_dataset = dataset.map(
            clean_example,
            fn_kwargs={"config": config},
            features=features,
            num_proc=config.num_proc if config.num_proc > 1 else None,
            cache_file_name=str(work / "cleaned.arrow"),
            load_from_cache_file=False,
            writer_batch_size=config.batch_size,
            desc="Filtering lengths and removing comments",
        )
        filtered_dataset = cleaned_dataset.filter(
            lambda code: code is not None,
            input_columns=["code_without_comments"],
            cache_file_name=str(work / "filtered.arrow"),
            load_from_cache_file=False,
            desc="Removing invalid examples",
        )
        unique_dataset = deduplicate(filtered_dataset)

        splits, questions = split_by_question(
            unique_dataset,
            val_questions=config.val_questions,
            seed=config.seed,
            work_dir=work,
            batch_size=config.batch_size,
        )

        splits["val"] = evaluate_validation(
            splits["val"],
            config=config,
        )
        val_with_answers = splits["val"].filter(
            lambda example: example["answer"] is not None,
        )

        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "input_rows": len(dataset),
            "filtered_rows": len(dataset) - len(filtered_dataset),
            "duplicate_rows": len(filtered_dataset) - len(unique_dataset),
            "retained_rows": len(unique_dataset),
            "train_rows": len(splits["train"]),
            "val_rows": len(splits["val"]),
            "val_questions_with_answer": len(val_with_answers.unique("question")),
        }
        assert (
            len(dataset)
            == metadata["retained_rows"]
            + metadata["filtered_rows"]
            + metadata["duplicate_rows"]
        )
        assert len(unique_dataset) == sum(len(split) for split in splits.values())

        save_splits(
            splits,
            output_dir=result,
            validation_questions=questions,
            metadata=metadata,
            batch_size=config.batch_size,
        )

        # Publish a complete directory only after all stages have succeeded.
        # work stays in the temporary directory and is removed on context exit.
        if output.exists():
            raise FileExistsError(f"Output appeared during processing: {output}")
        result.rename(output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="SlavaYus/tinysms_qa_12m")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--output-dir", default="data/prepared")
    parser.add_argument("--max-question-chars", type=int, default=1000)
    parser.add_argument("--max-code-chars", type=int, default=1200)
    parser.add_argument("--val-questions", type=int, default=7500)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--execution-timeout", type=float, default=2.0)
    parser.add_argument("--execution-workers", type=int, default=8)
    parser.add_argument("--answer-rel-tol", type=float, default=1e-6)
    parser.add_argument("--answer-abs-tol", type=float, default=1e-8)
    metadata = prepare_data(PreprocessingConfig(**vars(parser.parse_args())))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
