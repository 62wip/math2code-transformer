"""Load saved models for inference and generate code."""

import math
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
import yaml
from datasets import Dataset
from tokenizers import Tokenizer

from .data_processing.batching import CodeDataLoader
from .experiment.builders import build_model, get_device
from .utils.execution_check import SolutionExecutor
from .utils.paths import PROJECT_DIR


VotingMode = Literal["counter", "weighted_counter"] | None


def load_for_inference(
    run_dir: str | Path,
    *,
    checkpoint: str = "best",
    device: str | torch.device = "auto",
) -> tuple[torch.nn.Module, Tokenizer]:
    """Load an evaluation model and its tokenizer from an experiment directory."""

    run_dir = PROJECT_DIR / run_dir
    device = torch.device(get_device(device))

    with (run_dir / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    tokenizer = Tokenizer.from_file(str(run_dir / "init_state" / "tokenizer.json"))
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("Tokenizer has no <pad> token.")

    model = build_model(
        config["model"],
        vocab_size=tokenizer.get_vocab_size(),
        pad_id=pad_id,
    )
    saved = torch.load(
        run_dir / "checkpoints" / f"{checkpoint}.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(saved["model_state_dict"])
    del saved

    model.to(device)
    model.eval()
    return model, tokenizer


def _select_cache_rows(caches, rows, *, select_cross=True):
    return [
        {
            name: (
                (key, value)
                if name == "cross" and not select_cross
                else (key.index_select(0, rows), value.index_select(0, rows))
            )
            for name, (key, value) in layer.items()
        }
        for layer in caches
    ]


def _select_beam_by_answer(answers, *, rel_tol, abs_tol, beam_scores=None):
    """Group numeric answers and count votes or sum their probability mass."""

    answer_groups = []
    for beam_index, beam_answer in enumerate(answers):
        if beam_answer is None:
            continue

        added_to_group = False
        for group_answer, group_beam_indices in answer_groups:
            if math.isclose(
                beam_answer, group_answer, rel_tol=rel_tol, abs_tol=abs_tol
            ):
                group_beam_indices.append(beam_index)
                added_to_group = True
                break

        if not added_to_group:
            answer_groups.append((beam_answer, [beam_index]))

    if not answer_groups:
        return 0

    if beam_scores is None:
        group_weight = lambda group: len(group[1])
    else:
        scores = torch.tensor(beam_scores, dtype=torch.float64)

        def group_weight(group):
            return torch.logsumexp(scores[group[1]], dim=0).item()

    _, winning_beam_indices = max(answer_groups, key=group_weight)
    return winning_beam_indices[0]


def _beam_search_finished(finished, beam_scores, *, return_all_beams):
    all_finished = finished.all()

    if return_all_beams or beam_scores.size(1) == 1:
        return all_finished

    best_finished = finished.reshape(beam_scores.shape)[:, 0]
    unique_best = beam_scores[:, 0] > beam_scores[:, 1]

    return all_finished | (best_finished & unique_best).all()


def _beam_search_batch(
    model,
    batch,
    *,
    pad_id,
    bos_id,
    eos_id,
    max_new_tokens,
    beam_top_k,
    return_all_beams,
):
    """Return the best beam or all beams and their cumulative log-probabilities."""

    device = next(model.parameters()).device
    encoder_ids = batch["encoder_input_ids"].to(device)
    encoder_mask = batch["encoder_key_padding_mask"].to(device)
    encoder_outputs = model.encode(encoder_ids, encoder_mask)

    batch_size = encoder_ids.size(0)

    # Start: one <bos> per question
    bos = torch.full(
        (batch_size, 1),
        bos_id,
        dtype=torch.long,
        device=device,
    )

    logits, kv_caches = model.decode_step(
        bos,
        encoder_outputs,
        encoder_key_padding_mask=encoder_mask,
        kv_caches=None,
    )

    vocab_size = logits.size(-1)
    if beam_top_k > vocab_size:
        raise ValueError("beam_top_k exceeds the vocabulary size.")

    beam_scores, first_tokens = (
        logits.float().log_softmax(dim=-1).topk(beam_top_k, dim=-1)
    )

    initial_rows = torch.arange(batch_size, device=device).repeat_interleave(beam_top_k)

    # [B*K, 2]: <bos> + first_tokens
    generated = torch.cat(
        [bos.index_select(0, initial_rows), first_tokens.reshape(-1, 1)],
        dim=1,
    )

    kv_caches = _select_cache_rows(kv_caches, initial_rows)
    encoder_outputs = encoder_outputs.index_select(0, initial_rows)
    encoder_mask = encoder_mask.index_select(0, initial_rows)

    finished = first_tokens.reshape(-1) == eos_id
    offsets = torch.arange(batch_size, device=device)[:, None] * beam_top_k

    for _ in range(1, max_new_tokens):
        if _beam_search_finished(
            finished,
            beam_scores,
            return_all_beams=return_all_beams,
        ):
            break

        logits, kv_caches = model.decode_step(
            generated[:, -1:],
            encoder_outputs,
            encoder_key_padding_mask=encoder_mask,
            kv_caches=kv_caches,
        )

        log_probs = logits.float().log_softmax(dim=-1)
        log_probs[finished] = -torch.inf
        log_probs[finished, pad_id] = 0.0

        candidate_scores = beam_scores[:, :, None] + log_probs.reshape(
            batch_size, beam_top_k, vocab_size
        )

        beam_scores, candidate_ids = candidate_scores.flatten(1).topk(beam_top_k, dim=1)

        parents = (offsets + candidate_ids // vocab_size).reshape(-1)
        next_tokens = (candidate_ids % vocab_size).reshape(-1)

        generated = torch.cat(
            [
                generated.index_select(0, parents),
                next_tokens[:, None],
            ],
            dim=1,
        )

        finished = finished.index_select(0, parents) | (next_tokens == eos_id)

        kv_caches = _select_cache_rows(kv_caches, parents, select_cross=False)

    if return_all_beams:
        return generated, beam_scores.reshape(-1)
    return generated.index_select(0, offsets[:, 0]), beam_scores[:, 0]


def _generate_batch(
    model, tokenizer, batch, *, max_new_tokens, beam_top_k, return_all_beams
):
    """Generate and decode candidates for one batch."""
    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if pad_id is None or bos_id is None or eos_id is None:
        raise ValueError("Tokenizer must contain <pad>, <bos> and <eos> tokens.")
    special_token_ids = {bos_id, pad_id, eos_id}

    selected, beam_scores = _beam_search_batch(
        model,
        batch,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        max_new_tokens=max_new_tokens,
        beam_top_k=beam_top_k,
        return_all_beams=return_all_beams,
    )

    sequences = []
    eos_reached = []
    for generated_token_ids in selected[:, 1:].tolist():
        cleaned_token_ids = []
        reached_eos = False
        for token_id in generated_token_ids:
            if token_id == eos_id:
                reached_eos = True
                break
            if token_id not in special_token_ids:
                cleaned_token_ids.append(token_id)

        sequences.append(cleaned_token_ids)
        eos_reached.append(reached_eos)

    codes = tokenizer.decode_batch(
        sequences,
        skip_special_tokens=True,
    )

    result = {"codes": codes, "token_ids": sequences, "eos_reached": eos_reached}
    if return_all_beams:
        result["beam_scores"] = beam_scores.tolist()
    return result


def _execution_voting(
    batch_result, answers, *, beam_top_k, voting_mode, rel_tol, abs_tol
):
    """Select one candidate per question by its executed numeric answer."""
    selected_answers = []
    for question_start in range(0, len(answers), beam_top_k):
        question_end = question_start + beam_top_k
        question_answers = answers[question_start:question_end]

        winner_in_question = _select_beam_by_answer(
            question_answers,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            beam_scores=(
                batch_result["beam_scores"][question_start:question_end]
                if voting_mode == "weighted_counter"
                else None
            ),
        )
        winner_index = question_start + winner_in_question

        selected_answers.append(winner_index)

    return {
        field_name: [candidate_values[index] for index in selected_answers]
        for field_name, candidate_values in batch_result.items()
        if field_name != "beam_scores"
    }


def _store_batch_in_original_order(all_results, question_indices, batch_result):
    for field_name, batch_field_values in batch_result.items():
        for original_question_index, field_value in zip(
            question_indices, batch_field_values, strict=True
        ):
            all_results[field_name][original_question_index] = field_value


@torch.inference_mode()
def _generate_from_loader(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    loader: CodeDataLoader,
    *,
    use_mixed_precision: bool = True,
    beam_top_k: int = 8,
    voting_mode: VotingMode = None,
    timeout: float = 2.0,
    workers: int = 4,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> dict:
    """Generate from a prepared inference loader; always return full details."""

    question_count = len(loader.dataset)
    result = {
        "codes": [""] * question_count,
        "token_ids": [[] for _ in range(question_count)],
        "eos_reached": [False] * question_count,
    }
    if question_count == 0 or loader.max_new_tokens == 0:
        return result

    def store_execution(indices, batch_result, futures):
        selected = _execution_voting(
            batch_result,
            [future.result()["return_value"] for future in futures],
            beam_top_k=beam_top_k,
            voting_mode=voting_mode,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        _store_batch_in_original_order(result, indices, selected)

    device = next(model.parameters()).device
    amp_enabled = use_mixed_precision and device.type == "cuda"
    was_training = model.training
    model.eval()
    execution_pool = None
    pending_executions = deque()
    try:
        if voting_mode is not None:
            execution_pool = SolutionExecutor(timeout=timeout, workers=workers)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            for batch in loader:
                batch_result = _generate_batch(
                    model,
                    tokenizer,
                    batch,
                    max_new_tokens=loader.max_new_tokens,
                    beam_top_k=beam_top_k,
                    return_all_beams=voting_mode is not None,
                )
                if execution_pool is not None:
                    pending_executions.append(
                        (
                            batch["index"],
                            batch_result,
                            execution_pool.submit(batch_result["codes"]),
                        )
                    )
                    while pending_executions and all(
                        future.done() for future in pending_executions[0][2]
                    ):
                        store_execution(*pending_executions.popleft())
                else:
                    _store_batch_in_original_order(result, batch["index"], batch_result)

        for pending_batch in pending_executions:
            store_execution(*pending_batch)
    finally:
        model.train(was_training)
        if execution_pool is not None:
            execution_pool.close()
    return result


def generate_code(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    questions: Sequence[str],
    *,
    use_mixed_precision: bool = True,
    max_tokens: int = 2048,
    max_new_tokens: int = 64,
    bucket_width: int = 16,
    beam_top_k: int = 8,
    voting_mode: VotingMode = None,
    timeout: float = 2.0,
    workers: int = 4,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> list[str]:
    """Generate programs for question strings, preserving their input order.

    voting_mode: None selects the highest-scoring beam; 'counter' selects the
    most frequent executed answer; 'weighted_counter' sums exp(beam_score) for
    each answer. Failed executions do not vote; ties keep the best-scoring beam.
    """

    if len(questions) == 0 or max_new_tokens == 0:
        return [""] * len(questions)

    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("Tokenizer has no <pad> token.")

    input_ids = [encoding.ids for encoding in tokenizer.encode_batch(list(questions))]
    dataset = Dataset.from_dict(
        {"input_ids": input_ids, "input_length": [len(ids) for ids in input_ids]}
    )
    loader = CodeDataLoader(
        dataset,
        pad_id=pad_id,
        max_tokens=max_tokens,
        max_new_tokens=max_new_tokens,
        bucket_width=bucket_width,
        shuffle=False,
    )

    return _generate_from_loader(
        model,
        tokenizer,
        loader,
        use_mixed_precision=use_mixed_precision,
        beam_top_k=beam_top_k,
        voting_mode=voting_mode,
        timeout=timeout,
        workers=workers,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )["codes"]
