"""Length bucketing, token-budget batching and padded model inputs."""

from collections import defaultdict
from collections.abc import Iterator
from functools import partial
from itertools import islice, repeat
from random import Random

import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler


def collate_batch(
    examples: list[dict], *, pad_id: int, input_only: bool = False
) -> dict:
    """Pad questions and optionally build shifted training targets."""

    questions = [
        torch.tensor(example["input_ids"], dtype=torch.long) for example in examples
    ]
    encoder_input_ids = pad_sequence(
        questions,
        batch_first=True,
        padding_value=pad_id,
    )
    batch = {
        "encoder_input_ids": encoder_input_ids,
        "encoder_key_padding_mask": encoder_input_ids == pad_id,
    }
    if not input_only:
        codes = [
            torch.tensor(example["target_ids"], dtype=torch.long) for example in examples
        ]
        decoder_input_ids = pad_sequence(
            [code[:-1] for code in codes],
            batch_first=True,
            padding_value=pad_id,
        )
        label_ids = pad_sequence(
            [code[1:] for code in codes],
            batch_first=True,
            padding_value=pad_id,
        )
        batch.update(
            decoder_input_ids=decoder_input_ids,
            label_ids=label_ids,
            decoder_key_padding_mask=decoder_input_ids == pad_id,
        )

    if "index" in examples[0]:
        batch["index"] = [example["index"] for example in examples]
    if "answer" in examples[0]:
        batch["answer"] = [example["answer"] for example in examples]

    return batch


class BucketBatchSampler(Sampler[list[int]]):
    """Group question/code lengths in 2D buckets under a padded-token budget."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        max_tokens: int,
        bucket_width: int = 16,
        shuffle: bool = True,
        seed: int = 62,
        process_index: int = 0,
        process_count: int = 1,
        sequence_lengths: dict | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        self.process_index = process_index
        self.process_count = process_count
        self.max_tokens = max_tokens
        self.bucket_width = bucket_width
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.start_batch = 0

        lengths = sequence_lengths
        if lengths is None:
            columns = ["input_length"]
            if max_new_tokens is None:
                columns.append("target_length")
            lengths = dataset.with_format(
                "numpy",
                columns=columns,
            )[:]
        question_lengths = lengths["input_length"]
        code_lengths = (
            lengths["target_length"] - 1
            if max_new_tokens is None
            else repeat(max_new_tokens, len(question_lengths))
        )

        self.buckets = defaultdict(list)
        for index, (question_length, code_length) in enumerate(
            zip(
                question_lengths,
                code_lengths,
            )
        ):
            assert question_length + code_length <= max_tokens, (
                f"Example {index} exceeds max_tokens={max_tokens}"
            )

            bucket_key = (
                int(question_length) // bucket_width,
                int(code_length) // bucket_width,
            )
            self.buckets[bucket_key].append(index)

        self.batches = self._build_batches()

    def _build_batches(self) -> list[list[int]]:
        rng = Random(self.seed + self.epoch)
        batches = []
        for (question_bucket, code_bucket), indices in self.buckets.items():
            indices = indices.copy()
            if self.shuffle:
                rng.shuffle(indices)

            max_question_length = (question_bucket + 1) * self.bucket_width - 1
            max_code_length = (code_bucket + 1) * self.bucket_width - 1
            batch_size = max(
                1, self.max_tokens // (max_question_length + max_code_length)
            )

            for start in range(0, len(indices), batch_size):
                batches.append(indices[start : start + batch_size])

        if self.shuffle:
            rng.shuffle(batches)

        while len(batches) % self.process_count:
            for batch_index in reversed(range(len(batches))):
                if len(batches[batch_index]) > 1:
                    split_batch_index = batch_index
                    break
            else:
                raise ValueError(
                    "Cannot balance DDP batches; increase max_tokens or train_size."
                )

            batch = batches[split_batch_index]
            middle = len(batch) // 2
            batches[split_batch_index] = batch[:middle]
            batches.append(batch[middle:])
        return batches[self.process_index :: self.process_count]

    def set_epoch(self, epoch: int) -> None:
        """Rebuild batches for the given epoch using seed + epoch."""
        if epoch != self.epoch:
            self.epoch = epoch
            self.batches = self._build_batches()

    def __iter__(self) -> Iterator[list[int]]:
        return islice(self.batches, self.start_batch, None)

    def __len__(self) -> int:
        return len(self.batches)


class CodeDataLoader(DataLoader):
    """Connect code collation and bucketing for training or inference."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        pad_id: int,
        max_tokens: int,
        bucket_width: int = 16,
        shuffle: bool = True,
        seed: int = 62,
        process_index: int = 0,
        process_count: int = 1,
        sequence_lengths: dict | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        if max_new_tokens is not None:
            if "index" in dataset.column_names:
                dataset = dataset.remove_columns("index")
            dataset = dataset.add_column("index", list(range(len(dataset))))
        batch_sampler = BucketBatchSampler(
            dataset,
            max_tokens=max_tokens,
            bucket_width=bucket_width,
            shuffle=shuffle,
            seed=seed,
            process_index=process_index,
            process_count=process_count,
            sequence_lengths=sequence_lengths,
            max_new_tokens=max_new_tokens,
        )
        super().__init__(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=partial(
                collate_batch, pad_id=pad_id, input_only=max_new_tokens is not None
            ),
            generator=torch.Generator().manual_seed(seed),
            **kwargs,
        )
        self._bucket_sampler = batch_sampler
        self._next_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._bucket_sampler.set_epoch(epoch)
        self._next_epoch = epoch

    def __iter__(self) -> Iterator[dict]:
        return self.iter_from(0)

    def iter_from(self, start_batch: int) -> Iterator[dict]:
        if self._bucket_sampler.shuffle:
            self._bucket_sampler.set_epoch(self._next_epoch)
            self._next_epoch += 1
        self._bucket_sampler.start_batch = start_batch
        return super().__iter__()
