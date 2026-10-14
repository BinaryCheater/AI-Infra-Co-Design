"""G0 (notes/mechanistic_simulator_design.md §7): the shared event model replaces build_program in
smem_verify.py and the verdicts do not change. The legacy builder is kept here, verbatim, as the
equivalence oracle over the 24-point family."""
import inspect

from agentic.sim.events import BarrierArrive, Compute, IssueCopy, Wait
from agentic.templates.smem_tiling import EventModel, TilingDesign, lifetime_precondition, schedule_obligations
from agentic.templates.smem_tiling import smem_verify
from agentic.templates.smem_tiling.events import verifier_projection, warp_programs

FAMILY = [(s, p, tb) for s in (1, 2, 3) for p in (0, 1, 2, 3) for tb in (False, True)]


def legacy_build_program(design, em):          # the former smem_verify.build_program, verbatim
    S, P = design.stages, design.prefetch
    progs = []
    for w in range(em.warps):
        prog = []
        for u in range(min(P, em.phases)):
            prog.append(("LOAD", w, u))
        for p in range(em.phases):
            if P == 0:
                prog.append(("LOAD", w, p))
            elif p + P < em.phases:
                prog.append(("LOAD", w, p + P))
            if design.leading_barrier:
                prog.append(("BAR", "A", p))
            prog.append(("COMP", w, p))
            if design.trailing_barrier:
                prog.append(("BAR", "B", p))
        progs.append(prog)
    return progs


def test_g0_build_program_is_gone():
    assert not hasattr(smem_verify, "build_program") and "build_program" not in inspect.getsource(smem_verify)


def test_g0_projection_equals_legacy_on_the_family():
    em = EventModel()
    for s, p, tb in FAMILY:
        d = TilingDesign(16, 16, 8, s, p, tb)
        assert verifier_projection(warp_programs(d, em.warps, em.phases)) == legacy_build_program(d, em), (s, p, tb)
    d = TilingDesign(16, 16, 8, 2, 1, False, leading_barrier=False)
    assert verifier_projection(warp_programs(d, 2, 5)) == legacy_build_program(d, em)


def test_g0_events_carry_what_both_readers_need():
    prog = warp_programs(TilingDesign(128, 128, 32, 3, 1, False), warps=8, phases=32, tile_bytes=32768, flops_per_phase=1e6)
    assert len(prog.warps) == 8 and prog.slots == 3 and prog.phases == 32
    kinds = [type(e).__name__ for e in prog.warps[0]]
    assert kinds[:5] == ["IssueCopy", "IssueCopy", "Wait", "BarrierArrive", "Compute"]
    copies = [e for e in prog.warps[0] if isinstance(e, IssueCopy)]
    assert [c.tile for c in copies] == list(range(32)) and all(c.dst_slot == c.tile % 3 for c in copies)
    assert all(isinstance(e, (IssueCopy, Wait, BarrierArrive, Compute)) for e in prog.warps[0])
    assert not any(isinstance(e, BarrierArrive) and e.kind == "trailing" for e in prog.warps[0])


def test_g0_family_exactness_unchanged():
    """The S1 claim: the checker's safe set equals the declared precondition on all 24 points."""
    for s, p, tb in FAMILY:
        outs = schedule_obligations(TilingDesign(16, 16, 8, s, p, tb))
        safe = all(o.status in ("proved", "agrees") for o in outs)
        assert safe == lifetime_precondition(s, p, tb), (s, p, tb, [o.status for o in outs])
