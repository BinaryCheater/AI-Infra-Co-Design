"""Bound external tools and their descendants without shell command execution."""

import os
import signal
import subprocess
import threading
import time


def run_bounded(command, *, cwd, stdout, stderr, timeout, env=None, output_bytes=67108864):
    if not 0 < timeout <= 172800 or type(output_bytes) is not int or output_bytes < 128:
        raise ValueError("invalid process time/output budget")
    begin = time.monotonic()
    reason = None
    previous = {}

    def interrupted(_signal, _frame):
        raise InterruptedError("external tool cancelled")

    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True
        )
        if threading.current_thread() is threading.main_thread():
            for name in ("SIGTERM", "SIGHUP"):
                sig = getattr(signal, name, None)
                if sig is not None:
                    previous[sig] = signal.signal(sig, interrupted)
        try:
            while process.poll() is None:
                if time.monotonic() - begin >= timeout:
                    reason = "timeout"
                    break
                if stdout.stat().st_size + stderr.stat().st_size > output_bytes:
                    reason = "output_budget"
                    break
                time.sleep(0.05)
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                if process.poll() is None:
                    try:
                        process.wait(timeout=0.3)
                    except subprocess.TimeoutExpired:
                        pass
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    # A short-lived producer can exceed the budget before the polling edge.
    if stdout.stat().st_size + stderr.stat().st_size > output_bytes:
        reason = "output_budget"
    return {
        "returncode": process.returncode,
        "timed_out": reason == "timeout",
        "output_exhausted": reason == "output_budget",
        "wall_seconds": time.monotonic() - begin,
    }
