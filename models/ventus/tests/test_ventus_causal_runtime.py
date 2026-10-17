"""Real unit binding, finite credit conservation and shared native data path."""

import json
from pathlib import Path

from codesign.ventus.config import Hardware
from codesign.ventus.elf import ELFProgram
from codesign.ventus.runtime_contract import contract
from codesign.ventus.rust import PreparedProgram, RustSession, compact_records
from codesign.ventus.rust_frontend import predict_span
from tests.test_ventus_frontend_native import pressure_workload

ROOT = Path(__file__).resolve().parents[1]


def test_actual_compiled_multiply_divide_subword_and_partial_warps():
    case = json.loads((ROOT / "examples/compiled-subword-workflow.json").read_text())["cases"][0]
    elf = ELFProgram.read(ROOT / "tests/fixtures/compiled-subword-r1/program.elf")
    initial = {b["address"] + i * 4: v for b in case["inputs"] for i, v in enumerate(b["words"])}
    launch = elf.launch(
        "subword_u32",
        case["arguments"],
        global_size=case["global_size"],
        local_size=case["local_size"],
        functional_words=initial,
        private_profile="r1-workgroup",
    )
    with RustSession(Hardware()) as expanded, RustSession(Hardware()) as streamed:
        packet = expanded.generate(elf.words, launch)
        rows = list(compact_records(packet))
        assert {r["width"] for r in rows if r["addresses"]} == {1, 2, 4}
        assert {r["unit"] for r in rows} >= {"integer_mul", "integer_div"}
        assert any(r["active_mask"] != 0xFFFFFFFF for r in rows)
        a, b = expanded.dispatch(packet), streamed.dispatch(PreparedProgram.from_elf(elf, launch))
        assert a == b
        m = a["memory_mechanisms"]
        assert m["integer_mul_instructions"] > 0 and m["integer_div_instructions"] > 0
        assert m["groups_accepted"] == m["groups_returned"] > m["instructions_retired"]
        assert m["instructions_allocated"] == m["instructions_retired"]
        assert m["external_accepted"] == m["external_returned"]
        assert m["external_peak_slots"] <= 5
    # No RTL correctness assertion here: the real packed output-store diagnostic
    # is a retained counterexample. Its integer result buffer matched separately.


def test_frontend_and_elf_use_the_same_finite_memory_controller():
    work = pressure_workload()
    result = predict_span(
        work, Hardware(memory_latency=80), endpoint_names=["w0.pc80000038"], trace=True
    )
    assert result["model_version"] == "ventus-source-frontend-v4-causal"
    m = result["memory_mechanisms"]
    assert m["groups_accepted"] == m["groups_returned"] > 0
    assert m["instructions_allocated"] == m["instructions_retired"]
    assert m["external_accepted"] == m["external_returned"] > 0
    assert m["external_peak_slots"] <= 5
    assert result["terminal_cycles"] >= result["prediction_cycles"]


def test_external_parameter_overrides_cannot_claim_fixed_rtl_boundary():
    assert contract(Hardware())["external_target"]["kind"] == "pinned Mem_SimWrapper"
    assert (
        "parametric extension" in contract(Hardware(memory_latency=80))["external_target"]["kind"]
    )
