from dataclasses import replace

import pytest

from codesign.ventus.config import Hardware
from codesign.ventus.frontend import validate_straight_line
from codesign.ventus.graph import execute
from codesign.ventus.ir import Operation, Workload
from codesign.ventus.program import packed_gemm
from codesign.ventus.rtl_accuracy import complete_gemm, generate, predict, sha
from codesign.ventus.timing import build_graph


def _times(workload, frontend=True):
    graph = build_graph(workload, Hardware(), frontend=frontend)
    result = execute(graph)
    return dict(zip(graph.names, result["times"], strict=True)), result


def test_frontend_contention_and_epilogue_overlap():
    full = complete_gemm(8, 8)
    prefix = replace(
        full.workload,
        operations=tuple(
            op
            for op in full.workload.operations
            if int(op.name.rsplit("pc", 1)[1], 16) <= full.final_tensor_pc
        ),
    )
    full_times, result = _times(full.workload)
    prefix_times, _ = _times(prefix)
    end_names = [f"w{warp}.pc{full.final_tensor_pc:08x}.writeback" for warp in range(8)]
    assert max(full_times[n] for n in end_names) > max(prefix_times[n] for n in end_names)
    assert full_times["w7.pc80000000.collect"] > full_times["w1.pc80000000.collect"]
    assert result["counters"]["frontend_replays"] > 0
    assert result["counters"]["tensor"] == 64


def test_complete_multiwarp_program_keeps_output_private():
    program = complete_gemm(12, 6, "stream", True)
    program.workload.validate(Hardware())
    validate_straight_line(program.workload)
    for warp in range(6):
        ops = [op for op in program.workload.operations if op.warp == warp]
        assert len(ops) == len(program.words) - 1
        assert ops[-1].addresses == tuple(0x90002000 + warp * 128 + lane * 4 for lane in range(32))
    assert program.expected == packed_gemm(12, strategy="stream").expected * 6


@pytest.mark.parametrize("name", ["template", "pc80000002", "pc80000004"])
def test_frontend_rejects_nonconcrete_or_unaligned_stream(name):
    with pytest.raises(ValueError):
        _times(Workload((Operation(name, "vector", destination="v1"),)))


def test_frontend_rejects_gaps_and_collective_barrier():
    with pytest.raises(ValueError, match="omitted"):
        _times(Workload((Operation("pc80000000", "vector"), Operation("pc80000008", "vector"))))
    with pytest.raises(ValueError, match="straight-line"):
        _times(Workload((Operation("pc80000000", "barrier"),)))


def test_legacy_default_and_explicit_false_are_identical():
    workload = complete_gemm(3, 3).workload
    assert execute(build_graph(workload, Hardware())) == execute(
        build_graph(workload, Hardware(), frontend=False)
    )
    with pytest.raises(ValueError, match="boolean"):
        build_graph(workload, Hardware(), frontend="yes")


def test_probe_generation_and_prediction_refuse_overwrite(tmp_path):
    from pathlib import Path

    fixture = Path(
        "archive/analysis/ventus_flow_20261006/search_model/kernel-probes-v3/stream-4-rf0-lds1/input"
    )
    probes = tmp_path / "probes"
    generated = generate(probes, fixture, "calibration")
    assert generated["cases_sha256"] == sha(probes / "cases.json")
    output = tmp_path / "predictions.json"
    assert predict(probes / "cases.json", output, True, 4)["predictions"] == 6
    with pytest.raises(FileExistsError):
        generate(probes, fixture, "calibration")
    with pytest.raises(FileExistsError):
        predict(probes / "cases.json", output, True, 4)


@pytest.mark.parametrize("warps,strategy", [(1, "stream"), (3, "reload"), (8, "resident")])
def test_summary_span_matches_detailed_graph(warps, strategy):
    from codesign.ventus.frontend import predict_span

    program = complete_gemm(12, warps, strategy)
    names = [f"w{warp}.pc{program.final_tensor_pc:08x}" for warp in range(warps)]
    summary = predict_span(program.workload, Hardware(), endpoint_names=names)
    times, detailed = _times(program.workload)
    first = min(t for name, t in times.items() if name.endswith(".collect"))
    assert summary["prediction_cycles"] == max(times[n + ".writeback"] for n in names) - first
    assert summary["counters"] == detailed["counters"]
    with pytest.raises(ValueError, match="endpoint_names"):
        predict_span(program.workload, Hardware(), endpoint_names=["missing"])


def test_physical_operand_slots_and_zero_register_read():
    from codesign.ventus.physical_operands import with_physical_operands
    from codesign.ventus.termination import append_endprg

    program = complete_gemm(3, 2)
    workload = with_physical_operands(append_endprg(program.workload, program.words), program.words)
    workload.validate(Hardware())
    for op in workload.operations:
        if op.kind == "store":
            assert op.source_slots == (0, 2)
        if op.sources == ("x0",):
            assert op.source_slots == (0,)
        if op.sources == ("x2", "v14"):
            assert op.source_slots == (0, 1)
    assert sum(op.sources == ("x0",) for op in workload.operations) == 2
    assert sum(op.kind == "endprg" for op in workload.operations) == 2
    times, result = _times(workload)
    assert len([n for n in times if n.endswith(".endprg")]) == 2
    assert result["counters"]["rf_duplicate_grants"] > 0
    with pytest.raises(ValueError, match="frontend"):
        build_graph(workload, Hardware())


def test_machine_operand_mismatch_is_rejected():
    from codesign.ventus.physical_operands import with_physical_operands

    program = complete_gemm(3, 1)
    operations = list(program.workload.operations)
    operations[0] = replace(operations[0], sources=("v9", "v9"))
    with pytest.raises(ValueError, match="disagree"):
        with_physical_operands(
            replace(program.workload, operations=tuple(operations)), program.words
        )


def test_cli_rf8_prediction_records_requested_hardware(tmp_path, monkeypatch, capsys):
    import json
    from dataclasses import asdict

    from codesign.ventus.rtl_accuracy import main

    program = complete_gemm(5, 2, "resident", True)
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                dict(
                    name="rf8-check",
                    workload=asdict(program.workload),
                    words=program.words,
                    warps=2,
                    final_tensor_pc=program.final_tensor_pc,
                )
            ]
        )
    )
    output = tmp_path / "predictions.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rtl-accuracy",
            "predict",
            "--cases",
            str(cases),
            "--out",
            str(output),
            "--frontend",
            "--banks",
            "8",
        ],
    )
    main()
    capsys.readouterr()
    payload = json.loads(output.read_text())
    assert len(payload["predictions"]) == 1
    assert payload["predictions"][0]["banks"] == 8
    rf4 = tmp_path / "rf4.json"
    predict(cases, rf4, True, 4)
    assert (
        payload["predictions"][0]["prediction_cycles"]
        != json.loads(rf4.read_text())["predictions"][0]["prediction_cycles"]
    )


def test_cli_prediction_requires_explicit_bank_count(monkeypatch):
    from codesign.ventus.rtl_accuracy import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "rtl-accuracy",
            "predict",
            "--cases",
            "cases.json",
            "--out",
            "predictions.json",
            "--frontend",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
