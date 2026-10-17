"""Native frontend execution must preserve internal service, not just final spans."""

from dataclasses import replace

import pytest

from codesign.ventus.config import Hardware
from codesign.ventus.frontend_parity import compare, python_reference
from codesign.ventus.ir import Operation, Workload
from codesign.ventus.physical_operands import with_physical_operands
from codesign.ventus.rtl_accuracy import complete_gemm
from codesign.ventus.rust_frontend import PreparedFrontend, predict_span
from codesign.ventus.termination import append_endprg


def assert_parity(workload, hardware, endpoints):
    expected = python_reference(workload, hardware, endpoint_names=endpoints)
    actual = predict_span(workload, hardware, endpoint_names=endpoints, trace=True, causal=False)
    assert actual["backend"] == "rust"
    assert not compare(expected, actual)
    return actual


@pytest.mark.parametrize("banks", [4, 8])
@pytest.mark.parametrize("strategy", ["resident", "reload", "stream", "shared"])
@pytest.mark.parametrize("conflict", [False, True])
def test_complete_gemm_fetch_rf_lsu_and_terminal_parity(banks, strategy, conflict):
    program = complete_gemm(12, 6, strategy, conflict)
    workload = with_physical_operands(append_endprg(program.workload, program.words), program.words)
    actual = assert_parity(
        workload,
        Hardware().with_changes(rf_banks=banks),
        [f"w{warp}.pc{program.final_tensor_pc:08x}" for warp in range(6)],
    )
    assert actual["counters"]["frontend_replays"] > 0
    assert actual["counters"]["rf_duplicate_grants"] > 0
    assert sum("endprg" in row for row in actual["instruction_events"].values()) == 6


def pressure_workload():
    """Different line patterns, explicit RAW edges, x0 and noncompact slots."""
    ops = []
    for index in range(16):
        for warp in range(4):
            name = f"w{warp}.pc{0x80000000 + 4 * index:08x}"
            if index == 15:
                op = Operation(name, "endprg", warp=warp)
            elif index % 3 == 0:
                # Same-line merges and cache-index conflicts among warps.
                base = 0x90000000 + (index // 6) * 32768
                op = Operation(
                    name,
                    "load",
                    warp=warp,
                    sources=("v2",),
                    source_slots=(0,),
                    destination="v1",
                    addresses=tuple(base + 4 * lane for lane in range(32)),
                )
            elif index % 3 == 1:
                op = Operation(
                    name,
                    "vector",
                    warp=warp,
                    sources=("x0", "v1", "v1"),
                    source_slots=(0, 1, 3),
                    destination="v2",
                )
            else:
                op = Operation(
                    name, "scalar", warp=warp, sources=("x0",), source_slots=(0,), destination="x1"
                )
            ops.append(op)
    return Workload(tuple(ops), warps_per_block=4)


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"rf_banks": 8},
        {"lsu_entries": 2, "lsu_per_warp": 1},
        {"memory_latency": 100, "memory_bytes_per_cycle": 64},
    ],
)
def test_sparse_slots_cache_merge_and_lsu_pressure(changes):
    assert_parity(
        pressure_workload(),
        Hardware().with_changes(**changes),
        [f"w{warp}.pc80000038" for warp in range(4)],
    )


def test_prepared_calls_start_fresh_and_budget_is_enforced():
    hardware = Hardware()
    workload = pressure_workload()
    endpoints = [f"w{warp}.pc80000038" for warp in range(4)]
    prepared = PreparedFrontend.from_workload(workload, hardware)
    first = predict_span(prepared, hardware, endpoint_names=endpoints, trace=True, causal=False)
    second = predict_span(prepared, hardware, endpoint_names=endpoints, trace=True, causal=False)
    assert first == second
    with pytest.raises(ValueError):
        predict_span(prepared, hardware, endpoint_names=endpoints, max_cycles=1)


def test_frontend_input_errors_never_fall_back():
    hardware = Hardware()
    good = Workload((Operation("pc80000000", "vector", destination="v1"),))
    for bad in (
        replace(good, operations=(Operation("template", "vector", destination="v1"),)),
        replace(good, operations=good.operations + (Operation("pc80000008", "vector"),)),
        replace(good, operations=(Operation("pc80000000", "barrier"),)),
    ):
        with pytest.raises(ValueError):
            predict_span(bad, hardware, endpoint_names=[bad.operations[0].name])
    with pytest.raises(ValueError, match="endpoint"):
        predict_span(good, hardware, endpoint_names=["missing"])


def test_native_call_does_not_call_python_predictor(monkeypatch):
    from codesign.ventus import frontend, timing

    def forbidden(*args, **kwargs):
        raise AssertionError("Python timing fallback is forbidden")

    monkeypatch.setattr(frontend.FrontendEngine, "run", forbidden)
    monkeypatch.setattr(frontend, "predict_span", forbidden)
    monkeypatch.setattr(timing, "build_graph", forbidden)
    workload = Workload((Operation("pc80000000", "vector", destination="v1"),))
    assert predict_span(workload, Hardware(), endpoint_names=["pc80000000"])["backend"] == "rust"


@pytest.mark.parametrize(
    "changes",
    [
        {"sms": 1},
        {"sms": 2},
        {"sms": 2, "blocks_per_sm": 1},
        {"sms": 3, "warps_per_sm": 4, "blocks_per_sm": 1},
    ],
)
def test_multiple_blocks_reuse_physical_warps_with_resident_cache(changes):
    base = pressure_workload()
    workload = replace(
        base,
        operations=tuple(
            replace(op, name=f"b{block}.{op.name}", block=block)
            for block in range(7)
            for op in base.operations
        ),
    )
    endpoints = [f"b{block}.w{warp}.pc80000038" for block in range(7) for warp in range(4)]
    result = assert_parity(workload, Hardware().with_changes(**changes), endpoints)
    assert sum("endprg" in events for events in result["instruction_events"].values()) == 28


@pytest.mark.parametrize(
    "changes",
    [
        {"rf_read_ports": 2},
        {"rf_write_ports": 2, "writeback_ports": 2},
        {"collectors": 2},
        {"collectors": 16},
        {"l1_mshrs": 1},
        {"l1_subentries": 1},
        {"l1_subentries": 4},
        {"l1_mshrs": 2, "l1_subentries": 4, "memory_latency": 40},
        {"l1_sets": 4, "l1_ways": 1, "l2_sets": 4, "l2_ways": 1},
        {"memory_channels": 2, "memory_bytes_per_cycle": 16, "memory_latency": 20},
    ],
)
def test_finite_rf_collector_cache_and_memory_parameter_changes(changes):
    assert_parity(
        pressure_workload(),
        Hardware().with_changes(**changes),
        [f"w{warp}.pc80000038" for warp in range(4)],
    )


@pytest.mark.parametrize("writeback_ports", [1, 2])
def test_mixed_fpu_tensor_and_memory_writeback_competition(writeback_ports):
    kinds = ("load", "fma", "fadd", "fmax", "itof", "ftoi", "tensor", "vector")
    ops = []
    for index in range(17):
        for warp in range(4):
            name = f"w{warp}.pc{0x80000000 + 4 * index:08x}"
            if index == 16:
                ops.append(Operation(name, "endprg", warp=warp))
                continue
            kind = kinds[(index + warp) % len(kinds)]
            sources = ("v1", "v5", "v9") if kind in {"fma", "tensor"} else ("v1",)
            ops.append(
                Operation(
                    name,
                    kind,
                    warp=warp,
                    sources=sources,
                    source_slots=tuple(range(len(sources))),
                    destination="v1",
                    addresses=tuple(0x90000000 + lane * 4 for lane in range(32))
                    if kind == "load"
                    else (),
                )
            )
    workload = Workload(tuple(ops), warps_per_block=4)
    assert_parity(
        workload,
        Hardware().with_changes(writeback_ports=writeback_ports, rf_write_ports=writeback_ports),
        [f"w{warp}.pc8000003c" for warp in range(4)],
    )


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("changes", [{}, {"lds_banks": 16, "lds_ports": 2, "line_bytes": 256}])
def test_multiline_partial_lane_store_return_and_explicit_dependencies(shared, changes):
    operations = []
    base = 0x70000000 if shared else 0x90000000
    active = (0, 3, 5, 12, 20, 31)
    addresses = tuple(base + lane * 256 for lane in active)
    for index in range(9):
        for warp in range(2):
            name = f"w{warp}.pc{0x80000000 + 4 * index:08x}"
            dependency = (f"w{warp}.pc{0x80000000 + 4 * (index - 1):08x}",) if index else ()
            if index == 8:
                operation = Operation(name, "endprg", warp=warp)
            elif index % 2 == 0:
                operation = Operation(
                    name,
                    "load",
                    warp=warp,
                    destination="v1",
                    addresses=addresses,
                    active_lanes=active,
                    dependencies=dependency,
                )
            else:
                operation = Operation(
                    name,
                    "store",
                    warp=warp,
                    sources=("v1",),
                    source_slots=(2,),
                    addresses=addresses,
                    active_lanes=active,
                    dependencies=dependency,
                )
            operations.append(operation)
    assert_parity(
        Workload(tuple(operations), warps_per_block=2),
        Hardware().with_changes(**changes),
        [f"w{warp}.pc80000018" for warp in range(2)],
    )
