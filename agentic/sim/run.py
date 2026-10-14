"""The timed reader (§3.3): a discrete-event simulator of one kernel over the shared event model.

Mechanisms (each a resource with a queue and a service rule; parameters from the measured contract,
structure from sim/config/<host>.yaml):
  M1 Warp        a warp executes its events in order; blocks on waits and barriers; issue cost per op
  M3 Shared mem  bank conflicts serialize a warp's access by the stride's penalty (G3)
  M4 Async copy  per-SM queue of outstanding chunks; completion after memory latency + queueing (G3)
  M5 Hierarchy   L2 as an LRU over regions, DRAM as a bandwidth-limited queue (G2)
  M6 Barrier     release at the last arrival + barrier_s[threads]
  M7 Compute     per-SM FP32 unit: a warp's step occupies it for flops / rate
  M8 Scheduler   greedy: a block starts on the first SM with free capacity (smem, threads, measured residency)
  M9 Launch      a fixed cost before the first block
  M10 Clock      cycles <-> seconds at the sustained clock
Output: kernel time, the trace, per-resource activity (bytes DRAM actually served, etc.), and the
critical path per phase (what the slowest warp waited on: copy, barrier, compute, scheduler, ...).
"""
from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .events import BarrierArrive, Compute, IssueCopy, Program, Store, Wait


# ------------------------------------------------------------------------ engine
class Engine:
    def __init__(self):
        self.t = 0.0
        self.heap: list = []
        self.seq = 0

    def at(self, t: float, fn, *args):
        self.seq += 1
        heapq.heappush(self.heap, (t, self.seq, fn, args))

    def run(self):
        while self.heap:
            t, _, fn, args = heapq.heappop(self.heap)
            self.t = t
            fn(*args)


class Fifo:
    """A single-server queue: service(seconds) requests, FIFO, work-conserving."""
    def __init__(self):
        self.free_at = 0.0
        self.busy = 0.0

    def serve(self, now: float, seconds: float) -> tuple[float, float]:
        start = max(now, self.free_at)
        self.free_at = start + seconds
        self.busy += seconds
        return start, self.free_at


# ------------------------------------------------------------------------ memory (G2) and copies (G3) interfaces
class Memory:
    """M5 placeholder for G1: every copy completes after the measured cp.async latency for its size.
    G2 replaces it with the L2/DRAM hierarchy; G3 adds the per-SM copy queue."""
    def __init__(self, contract: dict, structural: dict, fixed: dict):
        self.c, self.s, self.f = contract, structural, fixed
        self.dram_bytes = 0
        self.l2_bytes = 0

    def copy_latency(self, bytes_: int) -> float:
        lat = self.c.get("async_copy_latency_s") or {}
        pts = sorted((int(k), float(v)) for k, v in lat.items())
        if not pts:
            return 3e-7
        if bytes_ <= pts[0][0]:
            return pts[0][1]
        for (b0, t0), (b1, t1) in zip(pts, pts[1:]):
            if b0 <= bytes_ <= b1:
                return t0 + (t1 - t0) * (bytes_ - b0) / (b1 - b0)
        b0, t0 = pts[-2]; b1, t1 = pts[-1]
        return t1 + (t1 - t0) * (bytes_ - b1) / (b1 - b0)

    def request(self, engine: Engine, sm: int, src, bytes_: int, done):
        """A copy of `bytes_` from region `src` into an SM's shared memory; call done(t) at completion."""
        engine.at(engine.t + self.copy_latency(bytes_), done)


# ------------------------------------------------------------------------ the simulator
@dataclass
class SimConfig:
    contract: dict
    structural: dict
    fixed: dict
    name: str = "?"

    @property
    def clock_hz(self) -> float:
        return float(self.contract.get("clock_hz_sustained") or self.contract.get("clock_khz", 2.4e6) * 1e3)

    def barrier_s(self, threads: int) -> float:
        b = self.contract["barrier_s"]
        if isinstance(b, dict):
            keys = sorted(int(k) for k in b)
            k = min(keys, key=lambda x: abs(x - threads))
            return float(b[str(k)])
        return float(b)

    def blocks_per_sm(self, smem_per_block: int, threads: int) -> int:
        meas = self.contract.get("max_blocks_per_sm_measured") or {}
        by_smem = {int(k): v for k, v in meas.items() if k.isdigit() and int(k) >= 1024 or k == "0"}
        if by_smem:
            keys = sorted(by_smem)
            # the largest measured smem size not exceeding this block's need gives the residency; threads limit too
            cand = [by_smem[k] for k in keys if k >= smem_per_block] or [1]
            resident = cand[0]
        else:
            resident = max(1, self.contract["smem_bytes_per_sm"] // max(1, smem_per_block))
        by_threads = max(1, self.contract.get("max_threads_per_sm", 1536) // threads)
        return max(1, min(resident, by_threads, self.contract.get("max_blocks_per_sm", 32)))


@dataclass
class Block:
    idx: int
    program: Program
    threads: int
    smem_bytes: int
    sm: int = -1
    t_start: float = 0.0
    t_end: float = 0.0
    warps_pc: list = field(default_factory=list)
    warp_time: list = field(default_factory=list)
    groups_done: dict = field(default_factory=dict)      # group -> completion time
    groups_issued: dict = field(default_factory=dict)     # group -> issue time
    barrier_arrivals: dict = field(default_factory=dict)  # barrier -> count
    barrier_release: dict = field(default_factory=dict)   # barrier -> release time
    waiting: dict = field(default_factory=dict)           # warp -> (reason, since)
    phase_wait: dict = field(default_factory=dict)        # phase -> {reason: seconds} for the slowest warp
    done_warps: int = 0


class Simulator:
    def __init__(self, cfg: SimConfig, memory: Memory | None = None, smem=None, amplify: tuple | None = None):
        self.cfg = cfg
        self.amplify = amplify                               # (warp, cycles): delay that warp before every compute
        self.engine = Engine()
        self.memory = memory or Memory(cfg.contract, cfg.structural, cfg.fixed)
        self.smem = smem                                     # M3 (G3); None = no bank model
        sms = int(cfg.contract["sms"])
        self.compute = [Fifo() for _ in range(sms)]
        self.sm_free_slots = [0] * sms
        self.sm_blocks: list[set] = [set() for _ in range(sms)]
        self.pending: deque = deque()
        self.trace: list = []
        self.blocks: list[Block] = []
        self.waits = defaultdict(float)                      # reason -> total warp-seconds waited
        self.issue_s = {k: v / cfg.clock_hz for k, v in (cfg.structural.get("issue_cycles") or {}).items()}

    # ---- M8 scheduler
    def launch(self, programs: list[Program], threads: int, smem_bytes: int, record_trace: bool = True):
        self.record = record_trace
        self.threads = threads
        per_sm = self.cfg.blocks_per_sm(smem_bytes, threads)
        self.sm_free_slots = [per_sm] * len(self.compute)
        self.blocks = [Block(i, p, threads, smem_bytes) for i, p in enumerate(programs)]
        self.pending = deque(self.blocks)
        t0 = float(self.cfg.contract.get("launch_s") or 0.0)
        self.engine.at(t0, self._schedule)
        self.engine.run()
        return self.result()

    def _schedule(self):
        # breadth-first: each pending block goes to the SM with the most free capacity (ties: lowest id), so
        # a grid of 48 blocks spreads over 48 SMs before any SM takes a second block (B9 fixes the capacity)
        while self.pending:
            sm = max(range(len(self.compute)), key=lambda i: (self.sm_free_slots[i], -i))
            if self.sm_free_slots[sm] <= 0:
                break
            if True:
                b = self.pending.popleft()
                self.sm_free_slots[sm] -= 1
                b.sm = sm; b.t_start = self.engine.t
                b.warps_pc = [0] * len(b.program.warps)
                b.warp_time = [self.engine.t] * len(b.program.warps)
                self.sm_blocks[sm].add(b.idx)
                if self.record:
                    self.trace.append((self.engine.t, sm, b.idx, None, "block_start", {}))
                for w in range(len(b.program.warps)):
                    self.engine.at(self.engine.t, self._step, b, w)

    def _finish_block(self, b: Block):
        b.t_end = self.engine.t
        self.sm_free_slots[b.sm] += 1
        self.sm_blocks[b.sm].discard(b.idx)
        if self.record:
            self.trace.append((self.engine.t, b.sm, b.idx, None, "block_end", {}))
        self._schedule()

    # ---- M1 warp step
    def _step(self, b: Block, w: int):
        seq = b.program.warps[w]
        if b.warps_pc[w] >= len(seq):
            b.done_warps += 1
            if b.done_warps == len(seq and b.program.warps):
                self._finish_block(b)
            return
        e = seq[b.warps_pc[w]]
        now = self.engine.t
        name = type(e).__name__
        cost = self.issue_s.get(name, 0.0)
        if isinstance(e, IssueCopy):
            # cp.async groups are per thread: each warp commits and waits for its own share of the tile
            b.groups_issued[(w, e.group)] = now
            self.memory.request(self.engine, b.sm, e.src, e.bytes, lambda t=None, b=b, w=w, g=e.group: self._copy_done(b, w, g))
            self._advance(b, w, now + cost, "issue_copy", e)
        elif isinstance(e, Wait):
            if (w, e.group) in b.groups_done:
                self._advance(b, w, now + cost, "wait", e)
            else:
                b.waiting[w] = ("copy", now, e)
        elif isinstance(e, BarrierArrive):
            b.barrier_arrivals[e.barrier] = b.barrier_arrivals.get(e.barrier, 0) + 1
            if b.barrier_arrivals[e.barrier] == len(b.program.warps):
                rel = now + self.cfg.barrier_s(b.threads)
                b.barrier_release[e.barrier] = rel
                if self.record:
                    self.trace.append((rel, b.sm, b.idx, None, "barrier_release", {"barrier": e.barrier, "phase": e.phase}))
                # release every waiting warp (and this one)
                for ww, (reason, since, ev) in list(b.waiting.items()):
                    if reason == "barrier" and ev.barrier == e.barrier:
                        del b.waiting[ww]
                        self._account(b, ww, "barrier", rel - since, ev.phase)
                        self._advance(b, ww, rel, "barrier", ev)
                self._advance(b, w, rel, "barrier", e)
            else:
                b.waiting[w] = ("barrier", now, e)
        elif isinstance(e, Compute):
            if self.amplify and w == self.amplify[0]:
                now += self.amplify[1] / self.cfg.clock_hz           # the race amplifier: this warp spins before its compute
            service = self._compute_service(b, e)
            start, end = self.compute[b.sm].serve(now, service)
            end_smem = start
            if self.smem is not None and e.smem_wavefronts:
                # the FMA pipe and the shared-memory pipe run in parallel per SM; the step ends when both have served it
                s0, end_smem = self.smem.serve(b.sm, now, e.smem_wavefronts, e.smem_stride)
                self._account(b, w, "smem", end_smem - now, e.phase)
            if start > now:
                self._account(b, w, "compute_queue", start - now, e.phase)
            self._account(b, w, "compute", service, e.phase)
            self._advance(b, w, max(end, end_smem), "compute", e)
        elif isinstance(e, Store):
            # a write-back: the warp continues after issue (stores are fire-and-forget), the bytes go to DRAM
            if hasattr(self.memory, "request_write"):
                self.memory.request_write(self.engine, b.sm, ("store", b.idx, w), e.bytes)
            self._advance(b, w, now + cost, "store", e)
        else:
            self._advance(b, w, now + cost, name.lower(), e)

    def _compute_service(self, b: Block, e: Compute) -> float:
        rate = float(self.cfg.contract["fp32_flops_per_s_per_sm"]) if e.unit == "fp32" else float(
            (self.cfg.contract.get("tensor_flops_per_s_per_sm") or {}).get(e.unit) or self.cfg.contract["fp32_flops_per_s_per_sm"])
        return e.flops / rate

    def _advance(self, b: Block, w: int, t: float, kind: str, e):
        if self.record:
            self.trace.append((t, b.sm, b.idx, w, kind, _brief(e)))
        b.warps_pc[w] += 1
        b.warp_time[w] = t
        self.engine.at(t, self._step, b, w)

    def _copy_done(self, b: Block, w: int, group: int):
        now = self.engine.t
        b.groups_done[(w, group)] = now
        if self.record:
            self.trace.append((now, b.sm, b.idx, w, "copy_done", {"group": group}))
        if w in b.waiting:
            reason, since, ev = b.waiting[w]
            if reason == "copy" and (w, ev.group) in b.groups_done:
                del b.waiting[w]
                self._account(b, w, "copy", now - since, ev.group)
                self._advance(b, w, now + self.issue_s.get("Wait", 0.0), "wait", ev)

    def _account(self, b: Block, w: int, reason: str, seconds: float, phase):
        self.waits[reason] += seconds
        if phase is not None:
            d = b.phase_wait.setdefault(phase, {})
            d[reason] = max(d.get(reason, 0.0), seconds)

    # ---- outputs
    def result(self) -> dict:
        t_end = max((b.t_end for b in self.blocks), default=0.0)
        hz = self.cfg.clock_hz
        # critical path per phase: the wait reason with the largest slowest-warp time, over all blocks
        phases: dict[int, dict] = {}
        for b in self.blocks:
            for p, d in b.phase_wait.items():
                acc = phases.setdefault(p, {})
                for r, s in d.items():
                    acc[r] = max(acc.get(r, 0.0), s)
        critical = {p: max(d, key=d.get) for p, d in phases.items()}
        counts = defaultdict(int)
        for r in critical.values():
            counts[r] += 1
        return {"time_s": t_end, "time_us": t_end * 1e6, "cycles": t_end * hz, "clock_hz": hz,
                "blocks": len(self.blocks), "sms": len(self.compute),
                "waves": None, "waits_s": dict(self.waits), "critical_path": critical, "critical_counts": dict(counts),
                "dram_bytes": self.memory.dram_bytes, "l2_bytes": self.memory.l2_bytes,
                "compute_busy_s": sum(c.busy for c in self.compute),
                "smem_busy_s": self.smem.busy() if self.smem is not None else 0.0,
                "memory": self.memory.stats() if hasattr(self.memory, "stats") else {},
                "trace": self.trace if self.record else None}


def _brief(e) -> dict:
    if isinstance(e, IssueCopy):
        return {"tile": e.tile, "slot": e.dst_slot, "group": e.group, "bytes": e.bytes}
    if isinstance(e, Wait):
        return {"group": e.group}
    if isinstance(e, BarrierArrive):
        return {"barrier": e.barrier, "phase": e.phase}
    if isinstance(e, Compute):
        return {"tile": e.tile, "phase": e.phase, "slot": e.slot_reads[0] if e.slot_reads else None}
    return {}


# ------------------------------------------------------------------------ convenience
def simulate(programs: list[Program], cfg: SimConfig, threads: int, smem_bytes: int, memory: Memory | None = None,
             smem=None, record_trace: bool = False, amplify: tuple | None = None) -> dict:
    return Simulator(cfg, memory, smem, amplify).launch(programs, threads, smem_bytes, record_trace)
