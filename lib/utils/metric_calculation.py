"""Metric calculations from tensors, texts and execution reports."""

import math
from collections.abc import Sequence
from statistics import fmean

from torch.nn import functional as F
from sacrebleu.metrics import BLEU


def token_loss_sum(logits, labels, pad_id):
    """Sum token cross-entropy over non-PAD targets."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
        reduction="sum",
    )


def correct_token_count(logits, labels, pad_id, *, top_k=1):
    """Count top-k hits without PAD. EOS is an ordinary target token."""
    if top_k == 1:
        correct = logits.argmax(dim=-1) == labels
    else:
        predictions = logits.topk(min(top_k, logits.size(-1)), dim=-1).indices
        correct = (predictions == labels.unsqueeze(-1)).any(dim=-1)
    return (correct & (labels != pad_id)).sum()


def numeric_return_count(answers: Sequence[float | None]) -> int:
    """Count finite numeric answers, excluding booleans and missing values."""
    return sum(
        answer is not None and not isinstance(answer, bool) and math.isfinite(answer)
        for answer in answers
    )


def correct_answer_count(answers, expected_answers, *, rel_tol=1e-6, abs_tol=1e-6):
    """Count finite numeric answers matching references within tolerance."""
    return sum(
        answer is not None
        and not isinstance(answer, bool)
        and math.isfinite(answer)
        and math.isclose(answer, expected, rel_tol=rel_tol, abs_tol=abs_tol)
        for answer, expected in zip(answers, expected_answers, strict=True)
    )


def syntax_valid_count(codes):
    """Compile without execution; a runtime error does not make syntax invalid."""
    valid = 0
    for code in codes:
        try:
            compile(code, "<generated>", "exec")
        except (SyntaxError, ValueError):
            continue
        valid += 1
    return valid


def no_eos_count(eos_reached) -> int:
    """Count generations that ended without reaching EOS."""
    return len(eos_reached) - sum(eos_reached)


def timeout_count(reports: Sequence[dict]) -> int:
    """Count programs whose execution exceeded the time limit."""
    return sum(report["status"] == "timeout" for report in reports)


def bleu_metrics(codes: Sequence[str], references: Sequence[Sequence[str]]) -> dict:
    """Return corpus BLEU (0–100) and its reproducibility signature."""
    reference_streams = [
        [refs[index] if index < len(refs) else None for refs in references]
        for index in range(max(len(refs) for refs in references))
    ]
    metric = BLEU(tokenize="13a", smooth_method="exp", effective_order=False)
    score = metric.corpus_score(list(codes), reference_streams)
    return {"bleu": score.score, "bleu_signature": str(metric.get_signature())}


def repetition_rate(token_sequences: Sequence[Sequence[int]], *, ngram_size=4) -> float:
    """Mean fraction of repeated n-grams per program, excluding special tokens."""
    rates = []
    for tokens in token_sequences:
        ngrams = [
            tuple(tokens[start : start + ngram_size])
            for start in range(len(tokens) - ngram_size + 1)
        ]
        rates.append(1 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0)
    return fmean(rates)
