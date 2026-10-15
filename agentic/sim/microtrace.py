"""The microarchitecture view's trace (notes/microarchitecture_view_design.md §4, §10): one design run through the timed
reader with recorders attached, written as files the chip view reads.

  python -m agentic.sim.microtrace --workload ragged_1000x1000x1000 --tile 128x128x32 --stages 3 --prefetch 1 --beta 0

Nothing here changes what the simulator computes. Three recorders listen:
  - the block-level trace the Simulator already keeps (block start/end, copy issue/done, waits, barriers, compute), filtered
    to one block per SM (the one that ends last on that SM) and folded into a per-SM summary for the chip strip;
  - the memory hierarchy's requests (L2 hit or miss, DRAM bytes, queue backlog), chip-wide per 1000 cycles and per request
    for the detailed blocks;
  - the warp model (sim/warp.py): each distinct compute phase is re-run once with WarpModel.trace_phase, which hands over every
    issue, every shared load (dispatch, LSU start and end, result ready) and every warp's state per cycle. The simulator itself
    simulates each distinct phase once and reuses it, so the trace stores each phase once ("templates") and the blocks point at it.

Files (runs/microtrace/<id>/): summary.json (design, result, chip strip, L2/DRAM series, which SM to show first),
templates.json (the warp-model phases), sm<NN>.json (the detailed block on SM NN: its events and its phases).
Times are SM cycles at the sustained clock.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ..templates.smem_tiling import TilingDesign
from .config import load
from .events import Compute
from .memory import Hierarchy
from .run import Fifo, SimConfig, Simulator
from .tiling import design_programs
from .warp import FFMA, LDS32, LDS128, ALU, WarpModel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "microtrace"
WINDOW = 1000                                         # cycles per chip-strip and L2/DRAM sample (§10)
CLS = {LDS128: "lds128", LDS32: "lds32", FFMA: "fma", ALU: "alu"}
STATES = ["issuing", "ready", "scoreboard", "mio", "barrier", "copy", "done", "idle"]
# the regime of a compute phase: the warp model's histogram by the profiler's families (validate.WARP_TO_NCU), largest wins
WARP_TO_NCU = {"issued": "not_selected", "not_selected": "not_selected", "scoreboard": "short_scoreboard", "lsu": "mio_throttle"}
COMPUTE_REGIME = {"not_selected": "scheduler-bound", "short_scoreboard": "load-bound", "mio_throttle": "load-bound"}


def compute_regime(hist: dict) -> tuple[str, dict]:
    fam = defaultdict(float)
    for k, v in hist.items():
        if k in WARP_TO_NCU:
            fam[WARP_TO_NCU[k]] += v
    top = max(fam, key=fam.get) if fam else "not_selected"
    return COMPUTE_REGIME[top], dict(fam)


# ------------------------------------------------------------------------ recorders
class PhaseRecorder:
    """Receives the warp model's issues, loads and per-cycle warp states for one phase (WarpModel.trace_phase)."""
    def __init__(self, W: int):
        self.issues = [[], [], [], []]                # columnar: t, warp, scheduler, class
        self.loads = [[] for _ in range(9)]           # t, warp, scheduler, class, k-step, wavefronts, lsu start, lsu end, ready
        self.states = [[], [], []]                    # run-length: t, warp, state index (only when it changes)
        self.cur = [None] * W
        self.seen = set()

    def issue(self, t, w, s, cls):
        for col, v in zip(self.issues, (t, w, s, cls)):
            col.append(v)
        self._set(t, w, "issuing")

    def load(self, t, w, s, cls, dep, wf, start, end, ready):
        for col, v in zip(self.loads, (t, w, s, cls, dep, round(wf, 3), round(start, 2), round(end, 2), round(ready, 2))):
            col.append(v)

    def state(self, t, w, name):
        self._set(t, w, name)

    def _set(self, t, w, name):
        if (t, w) in self.seen and name != "issuing":
            return                                     # a warp is classified once per cycle; issuing wins
        self.seen.add((t, w))
        i = STATES.index(name)
        if self.cur[w] != i:
            if self.states[0] and self.states[0][-1] == t and self.states[1][-1] == w:
                self.states[2][-1] = i                 # same cycle: replace
            else:
                self.states[0].append(t); self.states[1].append(w); self.states[2].append(i)
            self.cur[w] = i


class RecFifo(Fifo):
    """A Fifo that remembers each request's arrival and end, for queue depth over time."""
    def __init__(self, log: list):
        super().__init__(); self.log = log

    def serve(self, now, seconds):
        start, end = super().serve(now, seconds)
        self.log.append((now, start, end))
        return start, end


class RecHierarchy(Hierarchy):
    """The memory hierarchy with its L2 lookups and DRAM traffic recorded (time from the engine at the request)."""
    def __init__(self, C, S, F, hz):
        super().__init__(C, S, F)
        self.hz = hz
        self.l2_log, self.dram_log = [], []
        self.l2_q = RecFifo(self.l2_log); self.dram_q = RecFifo(self.dram_log)
        self.lookups = []                              # (t, sm, key, hit, bytes)
        self.ctx = None                                # (t, sm, block, warp) set by the simulator before a request
        self.detail = {}                               # (block) -> list of request records for the detailed blocks
        self.watch = set()

    def lookup(self, key, bytes_):
        hit = super().lookup(key, bytes_)
        if self.ctx is not None:
            t, sm, blk, w = self.ctx
            self.lookups.append((t, hit, bytes_))
            if blk in self.watch:
                self.detail.setdefault(blk, []).append({"t": t, "warp": w, "key": list(map(str, key)), "hit": hit, "bytes": bytes_})
        return hit


class TraceSink(list):
    """Stands in for Simulator.trace: keeps every event of the watched blocks, and folds every event into per-warp intervals."""
    def __init__(self, sim, watch: set):
        super().__init__()
        self.sim, self.watch = sim, watch
        self.last = {}                                 # (block, warp) -> time of the warp's previous event
        self.kept = defaultdict(list)                  # block -> events (seconds)
        self.intervals = []                            # (sm, t0, t1, kind, block, phase) for every warp of every block

    def append(self, ev):
        t, sm, blk, w, kind, pay = ev
        if blk in self.watch:
            self.kept[blk].append(ev)
        if w is None:
            if kind == "block_start":
                b = self.sim.blocks[blk]
                for ww in range(len(b.program.warps)):
                    self.last[(blk, ww)] = t
            return
        if kind == "copy_done":
            return
        t0 = self.last.get((blk, w), t)
        self.intervals.append((sm, t0, t, kind, blk, pay.get("phase") if pay else None))
        self.last[(blk, w)] = t


class IssueProxy:
    """The warp model as the Simulator sees it, remembering the arguments of each compute phase of the watched blocks."""
    def __init__(self, model: WarpModel):
        self.m = model
        self.ctx = None
        self.watch = set()
        self.calls = {}                                # (block, phase) -> (args, t_start), watched blocks
        self.key_of = {}                               # (block, phase) -> template id, every block
        self.keys = {}                                 # args -> template id

    def phase(self, *args):
        k = self.keys.setdefault(args, f"p{len(self.keys)}")
        if self.ctx is not None:
            blk, p, t = self.ctx
            self.key_of.setdefault((blk, p), k)
            if blk in self.watch:
                self.calls.setdefault((blk, p), (args, t))
        return self.m.phase(*args)

    def __getattr__(self, name):
        return getattr(self.m, name)


class RecSimulator(Simulator):
    def __init__(self, cfg, memory, issue, watch: set):
        super().__init__(cfg, memory, None, None, issue)
        self.watch = watch
        self.trace = TraceSink(self, watch)

    def _step(self, b, w):
        seq = b.program.warps[w]
        e = seq[b.warps_pc[w]] if b.warps_pc[w] < len(seq) else None
        self.memory.ctx = (self.engine.t, b.sm, b.idx, w)
        if isinstance(e, Compute):
            self.issue.ctx = (b.idx, e.phase, self.engine.t)
        else:
            self.issue.ctx = None
        super()._step(b, w)
        self.memory.ctx = None; self.issue.ctx = None


# ------------------------------------------------------------------------ the run
def design_id(wl: str, d: TilingDesign) -> str:
    return f"{wl}__{d.tm}x{d.tn}x{d.tk}_S{d.stages}_P{d.prefetch}_b{int(d.trailing_barrier)}"


def _sass(d: TilingDesign, w, chip):
    """Per-warp instructions per phase from the cached SASS counts, when this kernel's are cached (no host is contacted)."""
    try:
        from .validate import instr_per_phase_for
        row = {"design": {"tm": d.tm, "tn": d.tn, "tk": d.tk, "stages": d.stages, "prefetch": d.prefetch, "trailing_barrier": d.trailing_barrier}}
        return instr_per_phase_for(row, w, chip, None)
    except Exception:
        return None


def build(workload: str, d: TilingDesign, config: str = "spark-6951-gb10", out: Path | None = None, label: str | None = None) -> Path:
    from .validate import WORKLOADS
    from ..chips import load_chip, scope_chip
    w = next(x for x in WORKLOADS if x.name == workload)
    cfg = load(config); C, S, F = cfg["contract"], cfg["structural"], cfg["fixed"]
    sc = SimConfig(C, S, F, config); hz = sc.clock_hz
    cyc = lambda s: s * hz
    ipp = _sass(d, w, scope_chip(load_chip(config)))
    progs, meta = design_programs(d, w, instr_per_phase=ipp)
    model = WarpModel(C, S)

    # pass 1: which block ends last on each SM (the one shown in detail there), and which SM ends last overall
    sim1 = Simulator(sc, Hierarchy(C, S, F), None, None, model)
    sim1.memory.bind(d, w, meta)
    r1 = sim1.launch(progs, meta["threads"], meta["smem_bytes"], False)
    last_on_sm = {}
    for b in sim1.blocks:
        if b.sm not in last_on_sm or b.t_end > sim1.blocks[last_on_sm[b.sm]].t_end:
            last_on_sm[b.sm] = b.idx
    critical_block = max(sim1.blocks, key=lambda b: b.t_end).idx
    watch = set(last_on_sm.values())

    # pass 2: the same run with the recorders (deterministic, so the same blocks land on the same SMs)
    progs, meta = design_programs(d, w, instr_per_phase=ipp)
    mem = RecHierarchy(C, S, F, hz); mem.watch = watch; mem.bind(d, w, meta)
    proxy = IssueProxy(model); proxy.watch = watch
    sim = RecSimulator(sc, mem, proxy, watch)
    r = sim.launch(progs, meta["threads"], meta["smem_bytes"], True)
    assert abs(r["time_s"] - r1["time_s"]) < 1e-12, "the recording run differs from the plain run"
    total = cyc(r["time_s"])
    nwin = int(total // WINDOW) + 1
    sms = int(C["sms"])

    # warp-model phases, each re-run once with the recorder
    templates = {}
    for args, key in proxy.keys.items():
        wpb, blocks = args[0], args[1]
        rec = PhaseRecorder(wpb * blocks)
        cycles, hist = model.trace_phase(rec, *args)
        regime, fam = compute_regime(hist)
        templates[key] = {"key": key, "cycles": cycles, "warps_per_block": wpb, "blocks": blocks, "k_steps": args[2],
                          "ffma_per_step": args[3], "lds32_per_step": args[4], "lds128_per_step": args[5], "other_per_phase": args[6],
                          "hist": hist, "families": fam, "regime": regime,
                          "fma_cost": model.fma_cost, "mio_issue": model.mio_issue, "lsu_floor": model.floor(wpb * blocks),
                          "classes": CLS, "states": STATES,
                          "issues": rec.issues, "loads": rec.loads, "warp_states": rec.states}
    key_regime = {k: t["regime"] for k, t in templates.items()}

    # chip strip: per SM per window, the state that holds most warp-time (compute split by its phase's regime)
    acc = [[defaultdict(float) for _ in range(nwin)] for _ in range(sms)]
    CODE = {"compute_scheduler": "F", "compute_load": "M", "copy": "C", "barrier": "B", "issue": "I"}
    for sm_, t0, t1, kind, blk, ph in sim.trace.intervals:
        if t1 <= t0:
            continue
        a, z = cyc(t0), cyc(t1)
        if kind == "compute":
            cat = "compute_load" if key_regime.get(proxy.key_of.get((blk, ph))) == "load-bound" else "compute_scheduler"
        elif kind == "wait":
            cat = "copy"
        elif kind == "barrier":
            cat = "barrier"
        else:
            cat = "issue"
        i0, i1 = int(a // WINDOW), int(z // WINDOW)
        for i in range(i0, min(i1, nwin - 1) + 1):
            lo, hi = max(a, i * WINDOW), min(z, (i + 1) * WINDOW)
            if hi > lo:
                acc[sm_][i][cat] += hi - lo
    strip = []
    for s_ in range(sms):
        row = []
        for i in range(nwin):
            d_ = acc[s_][i]
            row.append(CODE[max(d_, key=d_.get)] if d_ else ".")
        strip.append("".join(row))

    # L2 / DRAM over time (chip-wide)
    l2 = {"hits": [0] * nwin, "misses": [0] * nwin, "bytes": [0.0] * nwin, "depth": [0.0] * nwin}
    dram = {"bytes": [0.0] * nwin, "depth": [0.0] * nwin}
    for t, hit, nb in mem.lookups:
        i = min(nwin - 1, int(cyc(t) // WINDOW))
        if hit:
            l2["hits"][i] += 1
        else:
            l2["misses"][i] += 1; dram["bytes"][i] += nb
    for log, dst in ((mem.l2_log, l2), (mem.dram_log, dram)):
        # requests in the queue (arrived, not yet served): averaged over each window
        for now, start, end in log:
            a, z = cyc(now), cyc(end)
            i0, i1 = int(a // WINDOW), int(z // WINDOW)
            for i in range(i0, min(i1, nwin - 1) + 1):
                lo, hi = max(a, i * WINDOW), min(z, (i + 1) * WINDOW)
                if hi > lo:
                    dst["depth"][i] += (hi - lo) / WINDOW
    for k in ("depth", "bytes"):
        for dct in (l2, dram):
            if k in dct:
                dct[k] = [round(x, 2) for x in dct[k]]

    # per SM: the detailed block's events in the note's vocabulary (§4) and its phases
    out = Path(out or OUT / design_id(workload, d)); out.mkdir(parents=True, exist_ok=True)
    sm_files = {}
    for sm_, blk in sorted(last_on_sm.items()):
        b = sim.blocks[blk]
        evs = []
        last = {}
        for (t, s_, bi, ww, kind, pay) in sim.trace.kept[blk]:
            tc = round(cyc(t), 1)
            if kind == "block_start":
                evs.append({"t": tc, "res": f"chip.sm[{sm_}]", "sm": sm_, "kind": "block_start", "block": blk})
                for x in range(len(b.program.warps)):
                    last[x] = tc
                continue
            if kind == "block_end":
                evs.append({"t": tc, "res": f"chip.sm[{sm_}]", "sm": sm_, "kind": "block_end", "block": blk}); continue
            if kind == "barrier_release":
                evs.append({"t": tc, "res": "sm.barrier", "sm": sm_, "kind": "barrier_release", "barrier": pay["barrier"], "phase": pay["phase"]}); continue
            if kind == "copy_done":
                evs.append({"t": tc, "res": "sm.copy", "sm": sm_, "warp": ww, "kind": "copy_done", "group": pay["group"], "phase": pay["group"]}); continue
            t0 = last.get(ww, tc); last[ww] = tc
            if kind == "issue_copy":
                evs.append({"t": round(t0, 1), "res": "sm.copy", "sm": sm_, "warp": ww, "kind": "copy_issue", "dur": round(tc - t0, 1),
                            "group": pay["group"], "slot": pay["slot"], "tile": pay["tile"], "bytes": pay["bytes"], "phase": pay["tile"], "state": "issuing"})
            elif kind == "wait":
                evs.append({"t": round(t0, 1), "res": f"sm.warp[{ww}]", "sm": sm_, "warp": ww, "kind": "wait", "dur": round(tc - t0, 1),
                            "group": pay["group"], "phase": pay["group"], "state": "copy"})
            elif kind == "barrier":
                evs.append({"t": round(t0, 1), "res": "sm.barrier", "sm": sm_, "warp": ww, "kind": "barrier_arrive", "dur": round(tc - t0, 1),
                            "barrier": pay["barrier"], "phase": pay["phase"], "state": "barrier"})
            elif kind == "compute":
                args, ts = proxy.calls.get((blk, pay["phase"]), (None, None))
                evs.append({"t": round(t0, 1), "res": f"sm.warp[{ww}]", "sm": sm_, "warp": ww, "kind": "phase_compute", "dur": round(tc - t0, 1),
                            "phase": pay["phase"], "slot": pay["slot"], "template": proxy.keys.get(args) if args else None})
            elif kind == "store":
                evs.append({"t": round(t0, 1), "res": f"sm.warp[{ww}]", "sm": sm_, "warp": ww, "kind": "store", "dur": round(tc - t0, 1), "state": "issuing"})
        for q in mem.detail.get(blk, []):
            tc = round(cyc(q["t"]), 1)
            if q["hit"]:
                evs.append({"t": tc, "res": "l2", "sm": sm_, "warp": q["warp"], "kind": "l2_hit", "tile": q["key"], "bytes": q["bytes"]})
            else:
                evs.append({"t": tc, "res": "l2", "sm": sm_, "warp": q["warp"], "kind": "l2_miss", "tile": q["key"], "bytes": q["bytes"]})
                evs.append({"t": tc, "res": "dram", "sm": sm_, "warp": q["warp"], "kind": "dram_read", "tile": q["key"], "bytes": q["bytes"]})
        evs.sort(key=lambda e: (e["t"], e["kind"] != "block_start"))
        # phases of the detailed block: the compute span, the template, and the regime by the simulator's critical-path rule
        phases = []
        misses = defaultdict(int)
        for q in mem.detail.get(blk, []):
            if not q["hit"]:
                try:
                    misses[int(q["key"][-1])] += 1
                except ValueError:
                    pass
        for p in range(b.program.phases):
            args, ts = proxy.calls.get((blk, p), (None, None))
            comp = [e for e in evs if e["kind"] == "phase_compute" and e["phase"] == p]
            waits = b.phase_wait.get(p, {})
            crit = max(waits, key=waits.get) if waits else "compute"
            key = proxy.keys.get(args) if args else None
            if crit in ("compute", "compute_queue", "smem"):
                regime = key_regime.get(key, "scheduler-bound")
            elif crit == "copy":
                regime = "DRAM-bound" if misses.get(p) else "copy-bound"
            else:
                regime = "barrier-bound"
            phases.append({"phase": p, "t0": min((e["t"] for e in comp), default=None), "t1": max((e["t"] + e["dur"] for e in comp), default=None),
                           "template": key, "critical": crit, "waits_cycles": {k: round(cyc(v), 1) for k, v in waits.items()}, "regime": regime})
        doc = {"design": design_id(workload, d), "sm": sm_, "block": blk, "block_coords": list(b.program.meta.get("block", ())),
               "t_start": round(cyc(b.t_start), 1), "t_end": round(cyc(b.t_end), 1), "warps": len(b.program.warps), "slots": b.program.slots,
               "phases": phases, "events": evs}
        f = out / f"sm{sm_:02d}.json"; f.write_text(json.dumps(doc, separators=(",", ":")))
        sm_files[sm_] = {"file": f.name, "block": blk, "t_end": doc["t_end"]}

    (out / "templates.json").write_text(json.dumps(templates, separators=(",", ":")))
    summary = {"id": design_id(workload, d), "label": label or f"{d.tm}×{d.tn}×{d.tk}, S={d.stages}, P={d.prefetch}, β={int(d.trailing_barrier)} on {workload}",
               "workload": workload, "design": {"tm": d.tm, "tn": d.tn, "tk": d.tk, "stages": d.stages, "prefetch": d.prefetch, "beta": int(d.trailing_barrier)},
               "config": config, "clock_hz": hz, "cycles": total, "time_us": r["time_us"], "blocks": r["blocks"], "phases": meta["KT"],
               "sass_overhead": ipp is not None, "instr_per_phase": ipp, "window": WINDOW, "sms": sms,
               "default_sm": sim.blocks[critical_block].sm, "critical_block": critical_block, "sm_files": sm_files,
               "strip": strip, "strip_codes": {"F": "FMA / issue (scheduler-bound compute)", "M": "MIO (load-bound compute)", "C": "copy wait",
                                                "B": "barrier", "I": "issuing copies", ".": "idle"},
               "l2": l2, "dram": dram, "result": {k: v for k, v in r.items() if k not in ("trace", "memory")},
               "memory": r.get("memory"), "templates": {k: {kk: t[kk] for kk in ("cycles", "blocks", "regime", "hist", "families")} for k, t in templates.items()}}
    (out / "summary.json").write_text(json.dumps(summary, separators=(",", ":"), default=str))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="ragged_1000x1000x1000")
    ap.add_argument("--tile", default="128x128x32"); ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--prefetch", type=int, default=1); ap.add_argument("--beta", type=int, default=0)
    ap.add_argument("--config", default="spark-6951-gb10"); ap.add_argument("--label", default=None)
    a = ap.parse_args()
    m, n, k = (int(x) for x in a.tile.split("x"))
    p = build(a.workload, TilingDesign(m, n, k, a.stages, a.prefetch, bool(a.beta)), a.config, label=a.label)
    print("wrote", p)


if __name__ == "__main__":
    main()
