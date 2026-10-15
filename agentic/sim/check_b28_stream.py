"""B28 through the stream input (the instruction-stream round, follow-up 1): the mixed kernel (loads + one 4 KB cp.async tile per
block per phase) and the copy-only kernel, read from their SASS and run in kernel mode at 1, 2, 3, 4, 6 blocks per SM. The copy
completion is the existing memory model's (B8's 4 KB group latency and the chip copy queue). Question: does the copy-chunk
address code explain the measured ~530 cycles per 4 KB tile per SM?

  python -m agentic.sim.check_b28_stream [--jobs N]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import load
from .isa import PipeCosts
from .sass import Function, function_text, read_dump
from .tensor import TensorModel
from .warp import WarpModel

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "calibration" / "spark-6951" / "raw"
PHASES = 16


def stream_for(mix: int, costs):
    F = Function(function_text(read_dump("b28_lsu_port_contention"), f"patternILi{mix}EE"))
    phase = max((r for r in F.loops if F.count(r, ("BAR",))), key=lambda r: r[1] - r[0])
    k = F.inner_loop(("LDS",), exclude=("BAR",)) if mix != 1 else None
    if k:                                    # B28's addresses: 128-bit loads broadcast one address (1 wavefront), 32-bit lane-consecutive (1)
        ia = ib = 0
        for i in F.body(*k):
            if i.opcode.startswith("LDS"):
                if ".128" in i.opcode: i.pattern, i.tidx = "template:A", ia; ia += 1
                else: i.pattern, i.tidx = "template:B", ib; ib += 1
    body = F.walk(phase[0], phase[1], {k: 8} if k else {}, costs, _skip_loop=phase)
    copy_code = [i for i in body if not i.opcode.startswith(("LDS", "FADD", "FFMA", "FMUL"))]
    # phases per pass of the loop from its content (one 16 B cp.async per thread per phase): the compiler unrolls the copy-only loop 4x
    per_pass = max(1, sum(1 for i in body if i.opcode.startswith("LDGSTS")))
    return body, per_pass, len(copy_code) / per_pass, len(body) / per_pass


def one(args):
    mix, b = args
    cfg = load("spark-6951-gb10"); C, S = cfg["contract"], cfg["structural"]; costs = PipeCosts(C)
    body, per_pass, ncopy, n = stream_for(mix, costs)
    passes = max(1, PHASES // per_pass); phases = passes * per_pass
    tm = TensorModel(C, S); wm = WarpModel(C, dict(S, input="stream"))
    cyc, h = wm.run_stream([body * passes] * 8, 8, b, costs, mode="kernel", a_wavefronts=(1.0,), b_wavefronts=(1.0,), copy_rule=tm.stage_cycles, stage_bytes=4096)
    # the same kernel with the copy-completion time set to zero: what the instructions alone cost
    cyc0, _ = wm.run_stream([body * passes] * 8, 8, b, costs, mode="kernel", a_wavefronts=(1.0,), b_wavefronts=(1.0,), copy_rule=lambda _: (0.0, 0.0), stage_bytes=4096)
    name = {1: "copy", 2: "both"}[mix]
    r = json.loads((RAW / f"lsu_contention[mix={name}_b={b}].json").read_text())
    meas_tile = r["value"] * 8 / b
    return {"mix": name, "blocks": b, "measured_cycles_per_tile_per_sm": meas_tile, "stream_cycles_per_tile_per_sm": cyc / phases / b,
            "stream_instructions_only": cyc0 / phases / b, "instructions_per_phase": n, "copy_and_loop_instructions_per_phase": ncopy,
            "copy_wait_share": h["copy_wait"] / max(1, sum(v for k, v in h.items() if isinstance(v, (int, float)) and k not in ("issued", "done", "copy_dispatch_cycles")))}


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=10); a = ap.parse_args()
    cases = [(mix, b) for mix in (1, 2) for b in (1, 2, 3, 4, 6)]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(one, cases))
    print("| run | blocks | measured cycles per 4 KB tile per SM | stream | error | instructions alone | copy-wait share |\n|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['mix']} | {r['blocks']} | {r['measured_cycles_per_tile_per_sm']:.0f} | {r['stream_cycles_per_tile_per_sm']:.0f} | {r['stream_cycles_per_tile_per_sm'] / r['measured_cycles_per_tile_per_sm'] - 1:+.0%} | {r['stream_instructions_only']:.0f} | {r['copy_wait_share']:.0%} |")
    print("instructions per phase per warp:", {r["mix"]: (r["instructions_per_phase"], r["copy_and_loop_instructions_per_phase"]) for r in rows})
    (ROOT / "calibration" / "spark-6951" / "b28_stream.json").write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
