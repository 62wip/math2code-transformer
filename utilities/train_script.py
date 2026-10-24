"""Train on CPU, a single GPU, or multiple GPUs with torchrun.

CPU: python -m utilities.train_script --device cpu --config ...
GPU: python -m utilities.train_script --device cuda --config ...
DDP: torchrun --standalone --nproc_per_node=2 -m utilities.train_script --config ...

Add --wandb-project nlp-hw2 to log each validation to W&B (main process only).
Authenticate with WANDB_API_KEY; --resume reuses init_state/wandb.json.
Add --max-train-seconds 3600 for a total one-hour budget, including validation
and saving. Resume subtracts saved elapsed time; omit the flag for no time limit.
"""

import argparse
import json
import os
from math import isfinite
from time import perf_counter

import torch

from lib.evaluation import validation
from lib.experiment import Experiment
from lib.train import train_loop
from lib.utils.ddp_distribution import (
    get_process_count,
    initialize_processes,
    is_main_process,
    shutdown_processes,
)
from lib.utils.metrics import format_metrics, summarize_epoch
from lib.utils.files import atomic_write_text


def main():
    if int(os.environ.get("RANK", "0")) != 0:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)

    parser = argparse.ArgumentParser(
        description="Train on CPU or CUDA, optionally with DDP via torchrun."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="YAML config for a new experiment")
    source.add_argument("--resume", help="Experiment directory to resume")
    parser.add_argument(
        "--weights-from", help="Checkpoint weights for a new experiment"
    )
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--max-train-seconds",
        type=float,
        help="Total elapsed-time budget in seconds, including validation/saving and saved time on resume",
    )
    parser.add_argument("--wandb-project", help="Enable W&B logging to this project")
    parser.add_argument(
        "--wandb-entity",
        help="W&B user or team (default: WANDB_ENTITY or account default)",
    )
    args = parser.parse_args()
    if args.resume and args.weights_from:
        parser.error("--weights-from is only supported with --config")
    if args.max_train_seconds is not None and (
        not isfinite(args.max_train_seconds) or args.max_train_seconds <= 0
    ):
        parser.error("--max-train-seconds must be a positive finite number")
    launched_with_torchrun = "LOCAL_RANK" in os.environ

    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; use --device cpu to train on CPU")
        cuda_device_index = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(cuda_device_index)
        device = f"cuda:{cuda_device_index}"
        backend = "nccl"
    else:
        device, backend = "cpu", "gloo"
    if launched_with_torchrun:
        initialize_processes(backend, device)
    wandb_run = None
    exit_code = 1
    try:
        state = (
            Experiment.resume(args.resume, device=device)
            if args.resume
            else Experiment.setup(
                args.config, weights_from=args.weights_from, device=device
            )
        )
        if is_main_process():
            print(
                f"Run: {state.run_dir}; device: {device}; processes: {get_process_count()}; "
                f"max_tokens per process: {state.config['batching']['max_tokens']}",
                flush=True,
            )
        if is_main_process() and args.wandb_project:
            import wandb

            wandb_path = state.run_dir / "init_state" / "wandb.json"
            saved_wandb = (
                json.loads(wandb_path.read_text(encoding="utf-8"))
                if args.resume and wandb_path.exists()
                else {}
            )
            entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
            if saved_wandb:
                entity = saved_wandb["entity"]
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=entity,
                name=state.run_dir.name,
                id=saved_wandb.get("id"),
                resume="allow" if saved_wandb else None,
                config=state.config,
                dir=str(state.run_dir),
            )
            atomic_write_text(
                wandb_path,
                json.dumps(
                    {
                        "id": wandb_run.id,
                        "project": wandb_run.project,
                        "entity": wandb_run.entity,
                    }
                )
                + "\n",
            )
            wandb_run.define_metric("examples_seen")
            wandb_run.define_metric("*", step_metric="examples_seen")
        training = state.config["training"]
        time_budget = args.max_train_seconds
        elapsed_before = state.details.get(
            "elapsed_seconds",
            sum(
                row["train_seconds"] + row.get("val_seconds", 0.0)
                for row in state.validation_history
            ),
        )
        started = perf_counter()

        def run_validation():
            return validation(
                state.model,
                state.val_loader,
                state.device,
                state.pad_id,
                tokenizer=state.tokenizer,
                heavy_loader=state.heavy_loader,
                heavy_options=state.heavy_options,
                use_mixed_precision=not args.no_mixed_precision,
            )

        for epoch in range(state.completed_epochs, training["num_epochs"]):
            remaining_seconds = None
            if time_budget is not None:
                remaining_seconds = (
                    time_budget - elapsed_before - (perf_counter() - started)
                )
                if remaining_seconds <= 0:
                    break
            state.train_loader.set_epoch(epoch)

            def on_validation(metrics):
                state.record_validation({"epoch": epoch + 1, **metrics})
                if metrics["batch_index"] == len(state.train_loader):
                    state.history.append(
                        summarize_epoch(state.validation_history, epoch + 1)
                    )
                if time_budget is not None:
                    state.details["elapsed_seconds"] = (
                        elapsed_before + perf_counter() - started
                    )
                state.save_checkpoint()
                if is_main_process():
                    print(format_metrics(state.validation_history[-1]), flush=True)
                    if wandb_run is not None:
                        wandb_run.log(state.validation_history[-1])

            train_loop(
                state.model,
                state.train_loader,
                state.optimizer,
                state.device,
                state.pad_id,
                use_mixed_precision=not args.no_mixed_precision,
                max_grad_norm=training["max_grad_norm"],
                val_loader=state.val_loader,
                examples_per_mini_epoch=training["examples_per_mini_epoch"],
                scheduler=state.scheduler,
                max_train_seconds=remaining_seconds,
                on_validation=on_validation,
                validation_fn=run_validation,
                **state.progress,
            )
        if time_budget is not None and is_main_process():
            used = elapsed_before + perf_counter() - started
            print(
                f"Elapsed time: {used:.1f}/{time_budget:g} s (including validation and saving).",
                flush=True,
            )
        exit_code = 0
    finally:
        try:
            if wandb_run is not None:
                wandb_run.finish(exit_code=exit_code)
        finally:
            if launched_with_torchrun:
                shutdown_processes()


if __name__ == "__main__":
    main()
