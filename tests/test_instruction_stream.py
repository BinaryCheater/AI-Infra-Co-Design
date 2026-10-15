"""The instruction stream (notes/instruction_stream.md): the pipe table refuses unknown opcodes, the SASS reader expands the GEMM
phase to the compiled instruction count, and the softmax rows run through the stream move toward the card."""
import json
from pathlib import Path

import pytest

from agentic.sim.config import load
from agentic.sim.isa import PipeCosts, UnknownOpcode, lookup
from agentic.sim.sass import Function, function_text, gemm_phase_stream, parse_instruction, read_dump, summarize
from agentic.sim.warp import WarpModel

ROOT = Path(__file__).resolve().parents[1]
SASS = ROOT / "calibration" / "spark-6951" / "sass"
pytestmark = pytest.mark.skipif(not SASS.exists(), reason="no SASS dumps")
CFG = load("spark-6951-gb10"); C, S = CFG["contract"], CFG["structural"]; COSTS = PipeCosts(C)


def test_unknown_opcode_is_an_error():
    with pytest.raises(UnknownOpcode):
        lookup("FROBNICATE.32")
    assert lookup("HMMA.16816.F32.BF16")[0] == "tensor" and lookup("MUFU.EX2")[0] == "xu" and lookup("SHFL.BFLY")[0] == "lsu"


def test_registers_are_widened():
    i = parse_instruction(0, None, "LDS.128", "R4, [R18]")
    assert i.dst == ("R4", "R5", "R6", "R7") and i.src == ("R18",)
    h = parse_instruction(0, None, "HMMA.16816.F32.BF16", "R8, R40, R44, R8")
    assert len(h.dst) == 4 and len(h.src) == 10


def test_gemm_phase_matches_the_compiled_ffma_count():
    v = json.loads((ROOT / "calibration" / "spark-6951" / "validation.json").read_text())
    row = next(r for r in v["rows"] if (r["design"]["tm"], r["design"]["tn"], r["design"]["tk"]) == (128, 128, 32))
    streams, info = gemm_phase_stream(row["sass_key"], 128, 128, 32, COSTS)
    assert summarize(streams[0])["by_opcode"]["FFMA"] == 8 * 8 * 32 and info["k_trips"] == 8


def test_softmax_stream_moves_toward_the_card():
    F = Function(function_text(read_dump("c3_softmax"), "softmax_rows"))
    lo, hi = F.inner_loop(("MUFU",)); body = F.walk(lo, hi, {}, COSTS, _skip_loop=(lo, hi))
    rows = sum(1 for i in body if i.opcode.startswith("LDG")); assert rows == 2
    wm = WarpModel(C, dict(S, input="stream"))
    cyc, _ = wm.run_stream([body * 8] * 4, 4, 1, COSTS, mode="kernel")
    measured = json.loads((ROOT / "calibration" / "spark-6951" / "raw" / "softmax_cycles_per_row[bps=1].json").read_text())["value"]
    assert abs(cyc / (8 * rows) / measured - 1) < 0.15
