"""Run small and heavy validation and return metric dictionaries."""

from time import perf_counter

import torch

from .data_processing.batching import CodeDataLoader
from .data_processing.evaluation import prepare_evaluation_data
from .inference import _generate_from_loader
from .utils import metric_calculation as metrics
from .utils.execution_check import execute_solutions


@torch.no_grad()
def small_validation_metrics(
    model, val_loader, device, pad_id, *, use_mixed_precision=True
):
    """Count losses and correct tokens across all batches, then divide once."""

    device = torch.device(device)
    amp_enabled = use_mixed_precision and device.type == "cuda"
    dtype = torch.float32 if device.type == "mps" else torch.float64
    total_loss = torch.zeros((), dtype=dtype, device=device)
    correct_tokens = torch.zeros((), dtype=torch.long, device=device)
    top5_correct_tokens = torch.zeros_like(correct_tokens)
    total_tokens = torch.zeros_like(correct_tokens)
    was_training = model.training
    model.eval()
    try:
        for batch in val_loader:
            labels = batch["label_ids"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(
                    encoder_input_ids=batch["encoder_input_ids"].to(
                        device, non_blocking=True
                    ),
                    decoder_input_ids=batch["decoder_input_ids"].to(
                        device, non_blocking=True
                    ),
                    encoder_key_padding_mask=batch["encoder_key_padding_mask"].to(
                        device, non_blocking=True
                    ),
                    decoder_key_padding_mask=batch["decoder_key_padding_mask"].to(
                        device, non_blocking=True
                    ),
                )
                total_loss += metrics.token_loss_sum(logits, labels, pad_id)
            correct_tokens += metrics.correct_token_count(logits, labels, pad_id)
            top5_correct_tokens += metrics.correct_token_count(
                logits, labels, pad_id, top_k=5
            )
            total_tokens += (labels != pad_id).sum()
    finally:
        model.train(was_training)

    tokens = total_tokens.item()
    correct = correct_tokens.item()
    top5_correct = top5_correct_tokens.item()

    return {
        "val_loss": total_loss.item() / tokens,
        "val_token_accuracy": correct / tokens,
        "val_top5_accuracy": top5_correct / tokens,
        "val_target_tokens": tokens,
        "val_correct_tokens": correct,
        "val_top5_correct_tokens": top5_correct,
    }


def heavy_validation_metrics(
    model,
    tokenizer,
    heavy_loader,
    *,
    beam_top_k=8,
    timeout=2.0,
    workers=4,
    rel_tol=1e-6,
    abs_tol=1e-6,
    repetition_ngram_size=4,
    use_mixed_precision=True,
):
    """Generate in batches; evaluate each selected program once."""

    generated = _generate_from_loader(
        model,
        tokenizer,
        heavy_loader,
        beam_top_k=beam_top_k,
        use_mixed_precision=use_mixed_precision,
    )

    codes = generated["codes"]
    reports = list(
        execute_solutions(codes, timeout=timeout, workers=workers, return_reports=True)
    )
    answers = [report["return_value"] for report in reports]
    etalon_aswers = heavy_loader.dataset["answer"]
    total = len(codes)

    correct_count = metrics.correct_answer_count(
        answers, etalon_aswers, rel_tol=rel_tol, abs_tol=abs_tol
    )
    syntax_valid_count = metrics.syntax_valid_count(codes)
    numeric_returns_count = metrics.numeric_return_count(answers)
    missing_eos_count = metrics.no_eos_count(generated["eos_reached"])
    timeouts_count = metrics.timeout_count(reports)

    reptition_rate = metrics.repetition_rate(
        generated["token_ids"],
        ngram_size=repetition_ngram_size,
    )
    bleu_score_and_signute = metrics.bleu_metrics(
        codes, heavy_loader.dataset["references"]
    )

    return {
        "pass_at_1": correct_count / total,
        "syntax_valid_rate": syntax_valid_count / total,
        "numeric_return_rate": numeric_returns_count / total,
        "no_eos_rate": missing_eos_count / total,
        "timeout_rate": timeouts_count / total,
        "repetition_rate": reptition_rate,
        **bleu_score_and_signute,
        "correct_programs_count": correct_count,
        "total": total,
        "syntax_valid_count": syntax_valid_count,
        "numeric_returns_count": numeric_returns_count,
        "no_eos_count": missing_eos_count,
        "timeouts_count": timeouts_count,
    }


def validation(
    model,
    val_loader,
    device,
    pad_id,
    *,
    tokenizer=None,
    heavy_loader=None,
    heavy_options=None,
    use_mixed_precision=True,
):
    """Run small validation and, when a heavy loader is supplied, heavy validation."""

    started = perf_counter()
    result = small_validation_metrics(
        model,
        val_loader,
        device,
        pad_id,
        use_mixed_precision=use_mixed_precision,
    )
    result["small_validation_seconds"] = perf_counter() - started
    if heavy_loader is not None:
        started = perf_counter()
        result.update(
            heavy_validation_metrics(
                model,
                tokenizer,
                heavy_loader,
                use_mixed_precision=use_mixed_precision,
                **(heavy_options or {}),
            )
        )
        result["heavy_validation_seconds"] = perf_counter() - started
    return result


def evaluate_pass_at_1(
    model,
    tokenizer,
    val_data,
    *,
    max_tokens=2048,
    max_new_tokens=256,
    bucket_width=16,
    timeout=2.0,
    workers=4,
    rel_tol=1e-6,
    abs_tol=1e-6,
    use_mixed_precision=True,
    sample_size=None,
    sample_seed=62,
):
    """Compute only pass@1 metric"""

    data, _, skipped = prepare_evaluation_data(
        val_data,
        tokenizer,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )
    loader = CodeDataLoader(
        data,
        pad_id=tokenizer.token_to_id("<pad>"),
        max_tokens=max_tokens,
        max_new_tokens=max_new_tokens,
        bucket_width=bucket_width,
        shuffle=False,
    )
    codes = _generate_from_loader(
        model,
        tokenizer,
        loader,
        use_mixed_precision=use_mixed_precision,
    )["codes"]
    answers = list(execute_solutions(codes, timeout=timeout, workers=workers))
    etalon_aswers = data["answer"]
    correct = metrics.correct_answer_count(
        answers,
        etalon_aswers,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    return {
        "pass_at_1": correct / len(codes),
        "correct_programs_count": correct,
        "total": len(codes),
        **skipped,
    }
