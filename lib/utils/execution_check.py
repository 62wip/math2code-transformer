"""Call solve() in isolated local processes with a timeout."""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Iterable, Iterator


_WORKER_PATH = Path(__file__).with_name("_execution_worker.py")


def _failure(error_type: str, message: str, *, status: str = "error") -> dict:
    return {
        "status": status,
        "error_type": error_type,
        "error_message": message[:300],
        "return_type": None,
        "return_value": None,
    }


def _execute_one(code: str, *, timeout: float) -> dict:
    """Fallback for platforms without fork: start a fresh interpreter."""
    try:
        completed = subprocess.run(
            [sys.executable, str(_WORKER_PATH)],
            input=code,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        result = _failure(
            "TimeoutError", "Execution time limit exceeded", status="timeout"
        )
    else:
        try:
            result = json.loads(completed.stdout)
            if (
                completed.returncode != 0
                or not isinstance(result, dict)
                or result.get("status") not in {"ok", "error"}
                or not {"error_type", "error_message", "return_type", "return_value"}
                <= result.keys()
            ):
                raise ValueError("Missing worker report")
        except ValueError:
            output = completed.stderr or completed.stdout
            result = _failure(
                "ProcessExit",
                f"Exit code {completed.returncode}; missing worker report: {output[:300]}",
            )
    return result


class _ExecutionWorker:
    """Reuse a helper interpreter; fork one isolated child per candidate."""

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self.process = None

    def execute(self, code: str) -> dict:
        if not hasattr(os, "fork"):
            return _execute_one(code, timeout=self.timeout)

        deadline = time.monotonic() + self.timeout
        if self.process is None:
            self.process = subprocess.Popen(
                [sys.executable, str(_WORKER_PATH), "--server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        process = self.process
        try:
            process.stdin.write(json.dumps([code, deadline]).encode("utf-8") + b"\n")
            process.stdin.flush()
            response = bytearray()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while b"\n" not in response:
                    # Let the helper kill and reap a timed-out child first.
                    remaining = deadline + 0.1 - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        self.close()
                        return _failure(
                            "TimeoutError",
                            "Execution time limit exceeded",
                            status="timeout",
                        )
                    chunk = os.read(process.stdout.fileno(), 4096)
                    if not chunk:
                        self.close()
                        return _failure(
                            "ProcessExit", "Execution helper exited without a report"
                        )
                    response.extend(chunk)
            result = json.loads(response)
            if not isinstance(result, dict) or result.get("status") not in {
                "ok", "error", "timeout"
            }:
                raise ValueError("Missing worker report")
        except (OSError, ValueError) as error:
            self.close()
            return _failure("ProcessExit", str(error))
        return result

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        # The helper and its current candidate share a private process group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        for stream in (process.stdin, process.stdout):
            try:
                stream.close()
            except OSError:
                pass


class SolutionExecutor:
    """Submit candidates without blocking; reuse exact-code results per run.

    On POSIX, only workers helper interpreters are started. Each candidate runs
    in a fresh fork of a helper, with the parent's wall-clock timeout. Helpers
    never import torch or CUDA. On other platforms, use fresh interpreters.
    Duplicate code shares a Future, so callers can still count every beam vote.
    Result reuse assumes deterministic solve(), as in the arithmetic dataset.
    Every Future returns an execution report; return_value contains the answer.
    """

    def __init__(self, *, timeout: float = 2.0, workers: int = 4) -> None:
        if not _WORKER_PATH.is_file():
            raise FileNotFoundError(f"Execution worker not found: {_WORKER_PATH}")
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.workers = [_ExecutionWorker(timeout) for _ in range(workers)]
        self.available = Queue()
        for worker in self.workers:
            self.available.put(worker)
        self.futures = {}

    def _execute(self, code: str) -> dict:
        worker = self.available.get()
        started = time.perf_counter()
        try:
            result = worker.execute(code)
            result["elapsed_s"] = round(time.perf_counter() - started, 4)
            return result
        finally:
            self.available.put(worker)

    def submit(self, codes: Iterable[str]) -> list[Future]:
        futures = []
        for code in codes:
            if code not in self.futures:
                self.futures[code] = self.pool.submit(self._execute, code)
            futures.append(self.futures[code])
        return futures

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        for worker in self.workers:
            worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def execute_solutions(
    codes: Iterable[str],
    *,
    timeout: float = 2.0,
    workers: int = 4,
    return_reports: bool = False,
) -> Iterator[float | None | dict]:
    """Yield numeric solve() answers, or execution reports, in input order.

    POSIX helpers fork isolated children without restarting Python per solution.
    Repeated source is executed once, assuming deterministic arithmetic code.
    Errors, timeouts, nonnumeric returns and bool yield None; stdout is discarded.
    With return_reports=True, yield status, error details, original return_type,
    numeric return_value and elapsed_s instead. Status ok means no exception,
    not a correct numeric answer. The timeout covers execution and helper startup,
    but excludes time in the queue. Duplicate inputs retain separate output rows.
    Interpreter launch failures propagate. There are no filesystem or memory limits.
    """
    with SolutionExecutor(timeout=timeout, workers=workers) as executor:
        for future in executor.submit(codes):
            report = future.result()
            yield report if return_reports else report["return_value"]
