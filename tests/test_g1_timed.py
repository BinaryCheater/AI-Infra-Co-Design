"""G1 acceptance (notes/mechanistic_simulator_design.md §7): the timed core reproduces B2 within 2 % and
B7 within 5 %, a compute-only kernel's time equals flops / rate + barriers within 2 %, and the
consistency condition holds on the 24-point family: no proved obligation is violated on the trace."""
import pytest

from agentic.sim.check import check_trace
from agentic.sim.config import load
from agentic.sim.kernels import barrier_kernel, compute_only_kernel, fma_kernel
from agentic.sim.run import SimConfig, simulate
from agentic.templates.smem_tiling import TilingDesign, lifetime_precondition
from agentic.templates.smem_tiling.events import warp_programs

CFG = load("spark-6951-gb10")
SC = SimConfig(CFG["contract"], CFG["structural"], CFG["fixed"], "spark-6951-gb10")
C = CFG["contract"]
RAW = __import__("pathlib").Path(__file__).resolve().parents[1] / "calibration" / "spark-6951" / "raw"


def _raw(q):
    import json
    return json.loads((RAW / (q.replace("[", "_").replace("]", "_").replace("=", "_") if False else q + ".json")).read_text())


def test_g1_b2_fma_within_2_percent():
    sms = C["sms"]; iters = 2000          # the rate is what is compared; fewer iterations keep the test fast
    r = simulate(fma_kernel(sms, 8, 256, iters), SC, 256, 0)
    flops = 2.0 * 8 * iters * 256 * sms * 8
    simulated_rate = flops / r["time_s"] / sms
    assert abs(simulated_rate / C["fp32_flops_per_s_per_sm"] - 1) < 0.02


def test_g1_b7_barrier_within_5_percent():
    for threads in (128, 256, 1024):
        iters = 500
        r = simulate(barrier_kernel(C["sms"], threads, iters), SC, threads, 0)
        per_barrier = (r["time_s"] - C["launch_s"]) / iters          # M9 launch cost is separate from M6
        assert abs(per_barrier / C["barrier_s"][str(threads)] - 1) < 0.05, threads


def test_g1_compute_only_equals_flops_over_rate_plus_barriers():
    phases, flops_pw = 32, 2.0 * 128 * 128 * 32 / 8
    for trailing in (False, True):
        r = simulate(compute_only_kernel(C["sms"], 256, phases, flops_pw, trailing), SC, 256, 0)
        expect = (phases * flops_pw * 8) / C["fp32_flops_per_s_per_sm"] + phases * (2 if trailing else 1) * C["barrier_s"]["256"] + C["launch_s"]
        assert abs(r["time_s"] / expect - 1) < 0.02, (trailing, r["time_s"], expect)
    assert r["critical_counts"].get("compute", 0) + r["critical_counts"].get("barrier", 0) == phases


def test_g1_consistency_on_the_family():
    """Proved points leave no violation on the timed trace; refuted points are not required to show one
    (the arbitrated interleaving is one of many), but the checker must at least run on them."""
    for s in (1, 2, 3):
        for p in (0, 1, 2, 3):
            for tb in (False, True):
                d = TilingDesign(16, 16, 8, s, p, tb)
                prog = warp_programs(d, 2, 5, tile_bytes=4 * (16 * 8 + 8 * 16), flops_per_phase=2.0 * 16 * 16 * 8 / 2)
                r = simulate([prog], SC, 64, s * 4 * (16 * 8 + 8 * 16), record_trace=True)
                v = check_trace(r["trace"], [prog])
                if lifetime_precondition(s, p, tb):
                    assert not v["lifetime"] and not v["readiness"], (s, p, tb, v)
                assert r["time_s"] > 0
