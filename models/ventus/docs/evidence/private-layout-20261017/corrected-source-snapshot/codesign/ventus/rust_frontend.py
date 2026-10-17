"""Native execution of the explicit source-frontend-v3 instruction contract.

Only instruction metadata is encoded in Python. Fetch, RF arbitration, execution,
LSU control and cache service run in Rust; the Python predictor is not invoked.
"""

import ctypes
import hashlib
import re
from dataclasses import dataclass, fields, replace

from .config import Hardware
from .frontend import FRONTEND_VERSION, validate_straight_line
from .rust import COUNTERS, KINDS, NONE, library, runtime_hash, source_hash

ABI = 0x5654533A
NATIVE_VERSION = "ventus-frontend-rust-v1"
DEFAULT_MAX_CYCLES = 10_000_000
EVENTS = ("collect", "operands", "issue", "writeback", "endprg")
MECHANISM_COUNTERS = (
    "frontend_packets",
    "frontend_replays",
    "rf_duplicate_grants",
    "lsu_mshr_idle_cycles",
    "lsu_mshr_add_cycles",
    "lsu_mshr_out_cycles",
    "lsu_mshr_returns",
    "lsu_mshr_deferred_allocations",
)


@dataclass(frozen=True)
class PreparedFrontend:
    """Reusable instruction encoding; no timed events or hardware answers cached."""

    workload: object
    payload: bytes
    sha256: str
    validated_hardware: Hardware

    @classmethod
    def from_workload(cls, workload, hardware):
        # Canonical immutable inputs make reuse of structural validation safe.
        canonical = []
        for op in workload.operations:
            changes = {}
            for name in ("sources", "addresses", "dependencies", "active_lanes", "source_slots"):
                value = getattr(op, name)
                if value is not None and not isinstance(value, tuple):
                    changes[name] = tuple(value)
            canonical.append(replace(op, **changes) if changes else op)
        workload = replace(workload, operations=tuple(canonical))
        workload.validate(hardware)
        validate_straight_line(workload)
        names = {op.name: i for i, op in enumerate(workload.operations)}
        data = [
            ABI,
            len(workload.operations),
            workload.warps_per_block,
            workload.vgpr_per_warp,
            workload.sgpr_per_warp,
            workload.lds_per_block,
            0,
        ]

        def register(value):
            return int(value[1:]) + (256 if value[0] == "x" else 0)

        for op in workload.operations:
            kind = 21 if op.kind == "endprg" else KINDS.index(op.kind)
            pc = int(re.search(r"pc([0-9a-fA-F]{8})$", op.name)[1], 16)
            slots = (
                op.source_slots if op.source_slots is not None else tuple(range(len(op.sources)))
            )
            if len(slots) > 4:
                raise ValueError("native frontend supports at most four physical operand slots")
            data.extend(
                [
                    kind,
                    op.block,
                    op.warp,
                    register(op.destination) if op.destination else NONE,
                    len(op.sources),
                    len(op.addresses),
                    len(op.dependencies),
                    pc,
                ]
            )
            data.extend(register(reg) for reg in op.sources)
            data.extend(slots)
            data.extend(op.addresses)
            data.extend(names[dep] for dep in op.dependencies)
        if any(not 0 <= value < 2**64 for value in data):
            raise ValueError("frontend input exceeds native u64 ABI")
        payload = bytes((ctypes.c_uint64 * len(data))(*data))
        return cls(workload, payload, hashlib.sha256(payload).hexdigest(), hardware)


class _FrontendSession:
    """Fresh native-owned timing state, including the hardware service template."""

    def __init__(self, hardware):
        values = [getattr(hardware, field.name) for field in fields(Hardware)]
        if any(value >= 2**64 for value in values):
            raise ValueError("hardware exceeds native u64 ABI")
        self.lib = library()
        self.lib.vt_frontend_create.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        self.lib.vt_frontend_create.restype = ctypes.c_void_p
        self.handle = self.lib.vt_frontend_create((ctypes.c_uint64 * 32)(*values))
        if not self.handle:
            raise ValueError("native frontend initialization rejected hardware")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.lib.vt_destroy(self.handle)
        self.handle = None


def predict_span(workload, hardware, *, endpoint_names, trace=False, max_cycles=DEFAULT_MAX_CYCLES):
    """Run fresh native state, returning declared writeback endpoint spans.

    The cycle budget bounds execution and raises on exhaustion. No silent fallback
    to Python or legacy Rust timing is permitted. Explicit tracing exports native
    per-instruction event timestamps for diagnosis and parity checks.
    """
    if type(max_cycles) is not int or not 0 < max_cycles < 2**64:
        raise ValueError("max_cycles must be a positive u64")
    prepared = (
        workload
        if isinstance(workload, PreparedFrontend)
        else PreparedFrontend.from_workload(workload, hardware)
    )
    hardware.validate()
    if (hardware.threads, hardware.lds_bytes) != (
        prepared.validated_hardware.threads,
        prepared.validated_hardware.lds_bytes,
    ):
        prepared.workload.validate(hardware)
    elif prepared.workload.resident_blocks(hardware) < 1:
        raise ValueError("one block does not fit hardware")
    operations = prepared.workload.operations
    names = {op.name: i for i, op in enumerate(operations)}
    endpoints = frozenset(endpoint_names)
    if not endpoints or not endpoints <= names.keys():
        raise ValueError("endpoint_names must name operations in the supplied workload")
    for name in endpoints:
        op = operations[names[name]]
        if not op.destination or op.kind == "endprg":
            raise ValueError("requested endpoints must produce writebacks")
    data = (ctypes.c_uint64 * (len(prepared.payload) // 8)).from_buffer_copy(prepared.payload)
    data[6] = max_cycles
    with _FrontendSession(hardware) as session:
        lib = session.lib
        lib.vt_frontend_trace.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        lib.vt_frontend_trace.restype = ctypes.POINTER(ctypes.c_uint64)
        result = (ctypes.c_uint64 * (len(COUNTERS) + 5))()
        status = lib.vt_run(session.handle, data, len(data), result)
        if status:
            size = ctypes.c_size_t()
            message = lib.vt_error(session.handle, ctypes.byref(size))
            detail = ctypes.string_at(message, size.value).decode(errors="replace")
            raise ValueError("native frontend rejected execution: " + detail)
        size = ctypes.c_size_t()
        raw = lib.vt_frontend_trace(session.handle, ctypes.byref(size))
        if not raw or size.value != 10 + len(operations) * len(EVENTS):
            raise RuntimeError("native frontend trace ABI or operation count differs")
        values = list(raw[: size.value])
        if values[0] == NONE:
            raise ValueError("native frontend did not collect any instruction")
        endpoint_times = {name: values[10 + names[name] * 5 + 3] for name in endpoints}
        if any(value == NONE for value in endpoint_times.values()):
            raise ValueError("native frontend did not complete requested endpoints")
        counters = {key: value for key, value in zip(COUNTERS, result[5:], strict=True) if value}
        counters.update(
            {
                key: value
                for key, value in zip(MECHANISM_COUNTERS, values[2:10], strict=True)
                if value
            }
        )
        endprg = sum(values[10 + i * 5 + 4] != NONE for i in range(len(operations)))
        if endprg:
            counters["endprg"] = endprg
        answer = {
            "model_version": FRONTEND_VERSION,
            "backend": "rust",
            "native_version": NATIVE_VERSION,
            "native_source_sha256": source_hash(),
            "native_runtime_sha256": runtime_hash(),
            "prepared_input_sha256": prepared.sha256,
            "prediction_cycles": max(endpoint_times.values()) - values[0],
            "first_collect": values[0],
            "endpoint_times": endpoint_times,
            "counters": counters,
            "terminal_cycles": result[0],
            "host_ticks": result[3],
            "max_cycles": max_cycles,
            "full_dag_retained": False,
        }
        if trace:
            answer["instruction_events"] = {
                op.name: {
                    event: values[10 + i * 5 + j]
                    for j, event in enumerate(EVENTS)
                    if values[10 + i * 5 + j] != NONE
                }
                for i, op in enumerate(operations)
            }
        return answer
