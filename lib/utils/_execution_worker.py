"""Execute source in a child process and emit a JSON execution report."""

import contextlib
import json
import math
import os
import sys
from decimal import Decimal
from numbers import Real


def failure(error_type: str, message: str, *, status: str = "error") -> dict:
    return {
        "status": status,
        "error_type": error_type,
        "error_message": message[:300],
        "return_type": None,
        "return_value": None,
    }


def execute(source: str) -> dict:
    result = {
        "status": "ok",
        "error_type": None,
        "error_message": None,
        "return_type": None,
        "return_value": None,
    }
    answer = None
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                namespace = {"__name__": "__dataset__"}
                exec(compile(source, "<dataset>", "exec"), namespace)
                if not callable(namespace.get("solve")):
                    raise TypeError("The source must define a callable solve()")
                value = namespace["solve"]()
                result["return_type"] = type(value).__name__
                if isinstance(value, (Real, Decimal)) and not isinstance(value, bool):
                    try:
                        number = float(value)
                        if math.isfinite(number):
                            answer = number
                    except (OverflowError, ValueError, TypeError):
                        pass
    except BaseException as error:
        answer = None
        result.update(
            status="error",
            error_type=type(error).__name__,
            error_message=str(error)[:300],
        )
    result["return_value"] = answer
    return result


def serve() -> None:
    """Fork from this small, single-threaded interpreter, never from torch."""
    import selectors
    import signal
    import time

    for request in sys.stdin:
        source, deadline = json.loads(request)
        read_fd, write_fd = os.pipe()
        child = os.fork()
        if child == 0:
            os.close(read_fd)
            exit_process = os._exit
            try:
                # A candidate cannot consume requests or write to the protocol.
                with open(os.devnull, "r+") as sink:
                    for fd in (0, 1, 2):
                        os.dup2(sink.fileno(), fd)
                    sys.stdin = sink
                    with os.fdopen(write_fd, "w", encoding="utf-8") as output:
                        output.write(json.dumps(execute(source), allow_nan=False))
            finally:
                exit_process(0)

        os.close(write_fd)
        response = bytearray()
        timed_out = False
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(read_fd, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        timed_out = True
                        break
                    chunk = os.read(read_fd, 4096)
                    if not chunk:
                        break
                    response.extend(chunk)
        finally:
            os.close(read_fd)
            if timed_out:
                try:
                    os.kill(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            _, status = os.waitpid(child, 0)
        if timed_out:
            result = failure(
                "TimeoutError", "Execution time limit exceeded", status="timeout"
            )
        else:
            try:
                if status != 0:
                    raise ValueError(f"Child exit status {status}")
                result = json.loads(response)
            except ValueError as error:
                result = failure("ProcessExit", str(error))
        print(json.dumps(result, allow_nan=False), flush=True)


def main() -> None:
    if "--server" in sys.argv[1:]:
        serve()
    else:
        result = execute(sys.stdin.read())
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
