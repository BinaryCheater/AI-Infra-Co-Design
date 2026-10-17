"""Conservative completion evidence from the Ventus RTL driver's raw logs.

The driver returns zero even after a simulation time limit. Its generic
"Simulation finished" line also does not imply CTA completion. Require the exact
named kernel completion event and retain diagnostics from both output streams.
This helper does not establish numerical correctness or a timed trace endpoint.
"""

import re

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ERROR = re.compile(
    r"\[[^\]\n]*\b(?:error|critical|fatal)\b[^\]\n]*\]"
    r"|%Error\b|\bassertion\b.*\bfailed\b|\bsegmentation fault\b"
    r"|\bSimulation exited ABNORMALLY\b|\b(?:FATAL|Aborted)\b",
    re.IGNORECASE,
)
_LIMIT = re.compile(
    r"\btime(?:[ _-]*(?:limit|max))?[ _-]*(?:exceed(?:ed)?|reached)\b"
    r"|\b(?:maximum|max)[ _-]*(?:simulation[ _-]*)?time[ _-]*(?:reached|exceeded)\b"
    r"|\btimed[ _-]*out\b|\btimeout\b",
    re.IGNORECASE,
)


def classify_rtl_logs(stdout, stderr, *, kernel_name, returncode, timed_out=False, kernel_id=0):
    """Return log-supported completion separately from functional/trace checks.

    ``kernel_name`` is literal (escaped before matching). A completion marker for
    a different kernel, block completion, or generic process finish is rejected.
    Duplicate copies of one marker in the two streams are retained as evidence
    but do not manufacture multiple kernel completions.
    """
    if not isinstance(kernel_name, str) or not kernel_name or "\n" in kernel_name:
        raise ValueError("kernel_name must be a nonempty single-line string")
    if type(kernel_id) is not int or kernel_id < 0:
        raise ValueError("kernel_id must be a nonnegative integer")
    completed = re.compile(
        rf"^\[RTL\s+info\]@(?P<time>\d+)\s+kernel{kernel_id}\s+"
        rf"{re.escape(kernel_name)}\s+finished\s*$"
    )
    markers, errors, limits = [], [], []
    for stream, contents in (("stdout", stdout), ("stderr", stderr)):
        for line_number, raw_line in enumerate(contents.splitlines(), 1):
            line = _ANSI.sub("", raw_line).strip()
            record = {"stream": stream, "line": line_number, "text": line}
            match = completed.fullmatch(line)
            if match:
                markers.append({**record, "simulation_time": int(match["time"])})
            if _ERROR.search(line):
                errors.append(record)
            if _LIMIT.search(line):
                limits.append(record)
    reasons = []
    if type(returncode) is not int or returncode != 0:
        reasons.append("driver did not return integer exit status zero")
    if timed_out is not False:
        reasons.append("external timeout occurred or timeout status is unknown")
    if not markers:
        reasons.append("exact named kernel completion marker absent")
    if len({m["simulation_time"] for m in markers}) > 1:
        reasons.append("multiple distinct kernel completion events in one run")
    if errors:
        reasons.append("error/critical/fatal diagnostics present")
    if limits:
        reasons.append("time-limit/timeout diagnostics present")
    return {
        "kernel_completed": not reasons,
        "kernel_completion_markers": markers,
        "error_diagnostics": errors,
        "limit_diagnostics": limits,
        "completion_reasons": reasons,
        "scope": "Named CTA/kernel completion from logs; numerical and trace checks remain separate.",
    }
