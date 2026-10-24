"""Shared byte-level BPE training and tokenization of questions and code."""

from datasets import Dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


def train_tokenizer(
    dataset: Dataset,
    *,
    code_column: str = "code_without_comments",
    vocab_size: int = 16_384,
    min_frequency: int = 2,
    batch_size: int = 1000,
) -> Tokenizer:
    """Train a shared byte-level BPE with automatic BOS/EOS tokens when encoding."""

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )

    def batches():
        texts = dataset.select_columns(["question", code_column])
        for batch in texts.iter(batch_size=batch_size):
            yield batch["question"]
            yield batch[code_column]

    tokenizer.train_from_iterator(batches(), trainer=trainer, length=2 * len(dataset))
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    return tokenizer


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: Tokenizer,
    *,
    code_column: str = "code_without_comments",
    batch_size: int = 1000,
) -> Dataset:
    """Add token IDs and lengths, including special tokens; retain source columns."""

    def tokenize_batch(batch):
        input_ids = [
            encoding.ids for encoding in tokenizer.encode_batch(batch["question"])
        ]
        target_ids = [
            encoding.ids for encoding in tokenizer.encode_batch(batch[code_column])
        ]
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "input_length": [len(ids) for ids in input_ids],
            "target_length": [len(ids) for ids in target_ids],
        }

    return dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=batch_size,
        desc="Tokenizing questions and code",
    )
