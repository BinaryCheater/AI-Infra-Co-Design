"""The tensor path (round 2, group 1; notes/hardware_calibration_round2_design.md §1): the tensor pipe as a resource per
sub-partition with an occupancy per mma and a result latency, and ldmatrix as a shared-memory instruction through the
B30 scheduler-to-LSU queue with its own latency. Every number comes from the contract (B31-B34); nothing is fitted.

Mechanisms, each with the benchmark that set it:
  - one mma occupies its sub-partition's tensor pipe for `mma_issue_cycles[shape]` (B31: 16.08 cycles for fp16, bf16,
    tf32 and fp8 alike; a second warp on the sub-partition doubles the time per warp, so the pipe is the resource);
  - an mma that reads an accumulator written by an earlier mma waits `mma_latency_cycles[shape]` (B31 (a): 30.1);
  - the warp that issued an mma issues nothing else until the pipe has taken it, `mma_issue_cycles` later (B34 (a):
    16 FFMAs between two mma add their full 16+ cycles, overlap 0; structural `mma_holds_warp`, B34 (a2) tests whether
    another warp's FFMAs fill those cycles);
  - ldmatrix.xN dispatches through the sub-partition's shared-memory slot (`mio_issue_cycles_per_load`), costs N
    wavefronts at the SM's LSU drain (B32 ncu: x4 reads four times x1's wavefront rate), and returns after
    `ldmatrix_latency_cycles[xN]` less the XOR-and-loop that the benchmark's link carries (one ALU latency);
  - FFMA, ALU, LDS.128 and cp.async as in sim/warp.py (B2, B27, B30, copy_path).

The simulator is a per-cycle in-order issue model over explicit per-warp programs (Instr lists) with a scoreboard on named
registers, four schedulers, block barriers and cp.async stage waits, so the same loop serves B31-B34's kernels and the C1
composite without a template in between.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .warp import WarpModel, LDS128, LDS32, FFMA, ALU, SFU, SHFL, CVT

MMA, LDSM, COPY, BAR, WAIT, GLOAD, GSTORE = 10, 11, 12, 13, 14, 15, 16
CLASS_NAME = {MMA: "mma", LDSM: "ldsm", FFMA: "ffma", ALU: "alu", LDS128: "lds128", LDS32: "lds32", COPY: "copy", BAR: "bar", WAIT: "wait", SFU: "sfu", SHFL: "shfl", CVT: "cvt", GLOAD: "gload", GSTORE: "gstore"}


@dataclass
class Instr:
    cls: int
    reads: tuple = ()
    writes: object = None
    shape: str = "bf16_m16n8k16"
    x: int = 4                 # ldmatrix width
    wavefronts: float = 0.0    # for LDS128/LDS32 when not the width table
    stage: int = 0             # for COPY/WAIT: the cp.async stage the chunk belongs to / the wait needs
    chunks: int = 1            # for COPY: 16 B chunks per thread issued by this instruction
    stage_bytes: int = 0       # for COPY: the whole block's bytes in this stage (sets the B8 latency and the chip-wide bandwidth share)
    op: str = ""               # for SFU / CVT / SHFL: the op or mode (ex2, rcp, f2fp_bf16x2, bfly, ...)


class TensorModel:
    def __init__(self, contract: dict, structural: dict):
        self.wm = WarpModel(contract, structural)
        c = contract
        hz = float(c.get("clock_hz_sustained") or 2.4e9)
        self.mma_issue = {k: float(v) for k, v in (c.get("mma_issue_cycles") or {}).items()}
        self.mma_lat = {k: float(v) for k, v in (c.get("mma_latency_cycles") or {}).items()}
        lat = c.get("ldmatrix_latency_cycles") or c.get("ldmatrix_cycles_per_instruction") or {}
        alu = float((c.get("scoreboard_latency_cycles") or {}).get("fma") or 4.0)
        # the benchmark's link is ldmatrix -> XOR -> loop -> ldmatrix: one dependent ALU latency belongs to the XOR
        self.ldsm_lat = {int(k[1:]): max(1.0, float(v) - alu) for k, v in lat.items()} or {4: float(c.get("ldmatrix_issue_cycles") or 12.0)}
        # B32 (a)'s chain ldmatrix.x4 -> mma -> address -> ldmatrix.x4 is the cleaner latency reading: one link less the mma latency (B31) and
        # one ALU latency for the address. The XOR link above reads 30 cycles more for x4 (notes/sim_gaps.md #26); the chain value is used
        dep = c.get("ldmatrix_mma_dependent_cycles")
        if dep and self.mma_lat:
            self.ldsm_lat[4] = max(1.0, float(dep) - self.latency("bf16_m16n8k16") - alu)
            self.ldsm_lat_source = "chain"
        else:
            self.ldsm_lat_source = "xor-link"
        self.ldsm_wf = {1: 1.0, 2: 2.0, 4: 4.0}
        # group 3: the XU pipe per sub-partition (B38), shuffles (B39: through the LSU queue only if B39 (c) said so), conversions;
        # global loads at the L2 latency through the LSU (lane-consecutive 16 B = 4 wavefronts), stores fire and forget
        self.l2_lat_cycles = float(c["latency_ns"]["l2"]) * 1e-9 * hz
        self.mma_holds_warp = bool(structural.get("mma_holds_warp", True))
        self.copy_cycles_per_chunk = float(c.get("cp_async_issue_cycles_per_chunk") or 6.8)
        # a stage's data is complete this long after its chunks were issued: B8's cp.async group latency from L2 by group size (every SM
        # copying at once), interpolated in the stage's bytes; and the chip's cp.async bandwidth at that size (B8) is a queue the stages of
        # every SM share (`sms` of them run the same program)
        self.hz = hz; self.sms = int(c.get("sms") or 48)
        self.copy_lat_s = {int(k): float(v) for k, v in (c.get("async_copy_latency_s") or {}).items()} or {4096: float(c["latency_ns"]["l2"]) * 1e-9}
        self.copy_bw = {int(k): float(v) for k, v in (c.get("async_copy_bytes_per_s") or {}).items()}
        self.copy_latency_cycles = self._interp(self.copy_lat_s, 4096) * hz

    @staticmethod
    def _interp(tab: dict, x: float) -> float:
        ks = sorted(tab)
        if x <= ks[0]: return tab[ks[0]]
        if x >= ks[-1]: return tab[ks[-1]]
        for a, b in zip(ks, ks[1:]):
            if a <= x <= b: return tab[a] + (x - a) / (b - a) * (tab[b] - tab[a])
        return tab[ks[-1]]

    def stage_cycles(self, stage_bytes: int) -> tuple[float, float]:
        """(latency cycles after the last chunk, chip-wide service cycles for `sms` SMs each moving stage_bytes) for one stage."""
        lat = self._interp(self.copy_lat_s, stage_bytes) * self.hz
        bw = self._interp(self.copy_bw, stage_bytes) if self.copy_bw else None
        service = (self.sms * stage_bytes / bw) * self.hz if bw else 0.0
        return lat, service

    def issue_cycles(self, shape: str) -> float:
        return self.mma_issue.get(shape) or next(iter(self.mma_issue.values()), 16.0)

    def latency(self, shape: str) -> float:
        return self.mma_lat.get(shape) or next(iter(self.mma_lat.values()), 30.0)

    def run(self, programs: list[list[Instr]], warps_per_block: int, trace=None) -> tuple[float, dict]:
        """programs: one Instr list per warp on the SM (warp w belongs to block w // warps_per_block, scheduler w % 4).
        Returns (cycles until every warp finished, histogram of warp-cycles by state)."""
        wm = self.wm; S = wm.schedulers; W = len(programs)
        pc = [0] * W; ready = [dict() for _ in range(W)]; hold = [0.0] * W; last_issue = [-1.0] * W
        tensor_free = [0.0] * S; mio_free = [0.0] * S; fma_free = [0.0] * S; lsu_free = 0.0; xu_free = [0.0] * S; shfl_free = [0.0] * S
        nblocks = (W + warps_per_block - 1) // warps_per_block
        bar_count = [dict() for _ in range(nblocks)]           # block -> barrier index -> arrivals
        bar_idx = [0] * W                                        # per warp: how many barriers it has passed
        stage_done = [dict() for _ in range(nblocks)]            # block -> stage -> cycle its data is complete
        stage_last = [dict() for _ in range(nblocks)]            # block -> stage -> last chunk issue cycle so far
        # resource times are fractional cycles; an issue at integer cycle t is allowed when the resource frees within the half cycle (as sim/warp.py)
        hist = {k: 0 for k in ("issued", "scoreboard", "tensor", "mio", "fma", "xu", "not_selected", "barrier", "copy_wait", "hold", "done")}
        floor = wm.floor(W); alive = W; t = 0; n = [len(p) for p in programs]; bw_free = 0.0; bw_seen = [set() for _ in range(nblocks)]
        while alive > 0 and t < 20_000_000:
            for s in range(S):
                issued = False
                for w in range(s, W, S):
                    if pc[w] >= n[w]:
                        continue
                    if last_issue[w] >= t or hold[w] > t + 0.5:
                        hist["hold"] += 1; continue
                    ins = programs[w][pc[w]]; b = w // warps_per_block
                    if any(ready[w].get(r, 0.0) > t + 0.5 for r in ins.reads):
                        hist["scoreboard"] += 1; continue
                    if ins.cls == BAR:
                        k = bar_idx[w]; arr = bar_count[b].setdefault(k, set()); arr.add(w)
                        if len(arr) < min(warps_per_block, W - b * warps_per_block):
                            hist["barrier"] += 1; continue
                        bar_idx[w] += 1; pc[w] += 1; last_issue[w] = t; hist["issued"] += 1
                        if pc[w] >= n[w]: alive -= 1
                        continue
                    if ins.cls == WAIT:
                        if stage_done[b].get(ins.stage, 0.0) > t + 0.5:
                            hist["copy_wait"] += 1; continue
                        pc[w] += 1; last_issue[w] = t; hist["issued"] += 1
                        if pc[w] >= n[w]: alive -= 1
                        continue
                    if ins.cls == MMA and tensor_free[s] > t + 0.5:
                        hist["tensor"] += 1; continue
                    if ins.cls in (LDSM, LDS128, LDS32) and mio_free[s] > t + 0.5:
                        hist["mio"] += 1; continue
                    if ins.cls == FFMA and fma_free[s] > t + 0.5:
                        hist["fma"] += 1; continue
                    if ins.cls in (SFU, CVT) and xu_free[s] > t + 0.5:
                        hist["xu"] += 1; continue
                    if ins.cls == SHFL and (shfl_free[s] > t + 0.5 or (wm.shfl_in_lsu_queue and mio_free[s] > t + 0.5)):
                        hist["mio"] += 1; continue
                    if ins.cls == GLOAD and mio_free[s] > t + 0.5:
                        hist["mio"] += 1; continue
                    if issued:
                        hist["not_selected"] += 1; continue
                    # issue
                    if ins.cls == MMA:
                        occ = self.issue_cycles(ins.shape); tensor_free[s] = t + occ
                        if ins.writes is not None: ready[w][ins.writes] = t + self.latency(ins.shape)
                        if self.mma_holds_warp: hold[w] = t + occ
                    elif ins.cls == LDSM:
                        mio_free[s] = t + wm.mio_issue; wf = self.ldsm_wf[ins.x]
                        start = max(float(t), lsu_free); lsu_free = start + max(floor, wf / wm.lsu_drain_wf)
                        if ins.writes is not None:
                            for tag in (ins.writes if isinstance(ins.writes, tuple) else (ins.writes,)):
                                ready[w][tag] = lsu_free + self.ldsm_lat.get(ins.x, self.ldsm_lat[max(self.ldsm_lat)])
                    elif ins.cls in (LDS128, LDS32):
                        mio_free[s] = t + wm.mio_issue; wf = ins.wavefronts or wm.wavefronts[ins.cls]
                        start = max(float(t), lsu_free); lsu_free = start + max(floor, wf / wm.lsu_drain_wf)
                        if ins.writes is not None: ready[w][ins.writes] = lsu_free + wm.lat[ins.cls]
                    elif ins.cls == FFMA:
                        fma_free[s] = max(fma_free[s], float(t)) + wm.fma_cost
                        if ins.writes is not None: ready[w][ins.writes] = t + wm.lat[FFMA]
                    elif ins.cls == ALU:
                        if ins.writes is not None: ready[w][ins.writes] = t + wm.lat[ALU]
                    elif ins.cls in (SFU, CVT):
                        table = wm.sfu_cost if ins.cls == SFU else wm.cvt_cost
                        cost = table.get(ins.op) or next(iter(table.values()), 8.0) or 8.0
                        xu_free[s] = max(xu_free[s], float(t)) + cost
                        lat = wm.sfu_lat.get(ins.op) or (next(iter(wm.sfu_lat.values()), 20.0) if ins.cls == SFU else wm.lat[ALU])
                        if ins.writes is not None: ready[w][ins.writes] = t + lat
                        if ins.cls == SFU and wm.sfu_hold: hold[w] = t + wm.sfu_hold
                    elif ins.cls == SHFL:
                        cost = wm.shfl_cost.get(ins.op) or next(iter(wm.shfl_cost.values()), 2.0)
                        shfl_free[s] = max(shfl_free[s], float(t)) + cost
                        if wm.shfl_in_lsu_queue:
                            mio_free[s] = t + wm.mio_issue; start = max(float(t), lsu_free); lsu_free = start + max(floor, wm.shfl_wavefronts / wm.lsu_drain_wf)
                        lat = wm.shfl_lat.get(ins.op) or next(iter(wm.shfl_lat.values()), 20.0)
                        if ins.writes is not None: ready[w][ins.writes] = t + lat
                        if wm.shfl_hold: hold[w] = t + wm.shfl_hold
                    elif ins.cls == GLOAD:
                        mio_free[s] = t + wm.mio_issue; wf = ins.wavefronts or 4.0
                        start = max(float(t), lsu_free); lsu_free = start + max(floor, wf / wm.lsu_drain_wf)
                        if ins.writes is not None: ready[w][ins.writes] = lsu_free + self.l2_lat_cycles
                    elif ins.cls == GSTORE:
                        mio_free[s] = t + wm.mio_issue
                    elif ins.cls == COPY:
                        # copy_path "separate": the chunks leave through their own path; the warp is held for the issue of its chunks
                        hold[w] = t + self.copy_cycles_per_chunk * ins.chunks
                        last = t + self.copy_cycles_per_chunk * ins.chunks
                        stage_last[b][ins.stage] = max(stage_last[b].get(ins.stage, 0.0), last)
                        if ins.stage_bytes:
                            lat, service = self.stage_cycles(ins.stage_bytes)
                            # the chip's copy bandwidth (B8): every SM runs this program, so each block-stage is one of `sms` identical
                            # stages started together; the shared FIFO serves them in issue order, `service` cycles per stage-set
                            if ins.stage not in bw_seen[b]:
                                bw_seen[b].add(ins.stage); bw_free = max(bw_free, float(t)) + service
                            stage_done[b][ins.stage] = max(stage_last[b][ins.stage] + lat, bw_free)
                        else:
                            stage_done[b][ins.stage] = stage_last[b][ins.stage] + self.copy_latency_cycles
                    pc[w] += 1; last_issue[w] = t; hist["issued"] += 1; issued = True
                    if trace: trace(t, w, s, ins)
                    if pc[w] >= n[w]: alive -= 1
            hist["done"] += sum(1 for w in range(W) if pc[w] >= n[w])
            t += 1
        return float(t), hist


# ---------------------------------------------------------------------------------------------- the benchmarks as programs
def prog_b31(shape: str, a: int, iters: int, int8: bool = False) -> list[Instr]:
    p = []
    for _ in range(iters):
        for j in range(a):
            p.append(Instr(MMA, reads=(f"acc{j}",), writes=f"acc{j}", shape=shape))
        p.append(Instr(ALU))
    return p


def prog_b32_pipelined(depth: int, trips: int, steps: int = 16) -> list[Instr]:
    """B32 (b): per step two ldmatrix.x4 (A and B fragments for step k + depth into slot (k + depth) % (depth + 1)) then four mma."""
    p = []
    for _ in range(trips):
        def load(k, slot):
            p.append(Instr(LDSM, writes=(f"fa{slot}",), x=4)); p.append(Instr(LDSM, writes=(f"fb{slot}",), x=4))
        for d in range(depth):
            load(d, d)
        for k in range(steps):
            cur = k % (depth + 1)
            if k + depth < steps:
                load(k + depth, (k + depth) % (depth + 1))
            for j in range(4):
                p.append(Instr(MMA, reads=(f"fa{cur}", f"fb{cur}", f"acc{j}"), writes=f"acc{j}"))
        p.append(Instr(ALU))
    return p


def prog_b32_dependent(links: int) -> list[Instr]:
    p = []
    for _ in range(links):
        p.append(Instr(ALU, reads=("acc",), writes="addr"))                       # the address from the accumulator
        p.append(Instr(LDSM, reads=("addr",), writes=("fa",), x=4))
        p.append(Instr(MMA, reads=("fa", "acc"), writes="acc"))
    return p


def prog_b32_ldsm_latency(x: int, iters: int) -> list[Instr]:
    p = []
    for _ in range(iters):
        p.append(Instr(ALU, writes="addr"))
        p.append(Instr(LDSM, reads=("addr",), writes=("f",), x=x))
        p.append(Instr(ALU, reads=("f", "r"), writes="r"))                        # the XOR into the accumulator
    return p


def prog_b33_load(role: str, trips: int, steps: int = 16) -> list[Instr]:
    """B33's load block: per step an LDS.128 (lane-consecutive, 4 wavefronts) feeding one FADD chain, and/or one cp.async chunk."""
    p = []
    for _ in range(trips):
        for k in range(steps):
            if role in ("lds128", "both"):
                p.append(Instr(LDS128, writes=f"v{k % 4}", wavefronts=4.0)); p.append(Instr(FFMA, reads=(f"v{k % 4}", "acc"), writes="acc"))
            if role in ("cp_async", "both"):
                p.append(Instr(COPY, stage=k // 4, chunks=1))
                if k % 4 == 3:
                    p.append(Instr(WAIT, stage=max(0, k // 4 - 1)))
        p.append(Instr(ALU))
    return p


def prog_b34_ratio(ff: int, iters: int) -> list[Instr]:
    p = []
    for _ in range(iters):
        for j in range(4):
            p.append(Instr(MMA, reads=(f"acc{j}",), writes=f"acc{j}"))
            for f in range(ff):
                p.append(Instr(FFMA, reads=(f"f{f % 8}",), writes=f"f{f % 8}"))
        p.append(Instr(ALU))
    return p


def prog_ffma_only(chains: int, iters: int) -> list[Instr]:
    p = []
    for _ in range(iters):
        for f in range(chains):
            p.append(Instr(FFMA, reads=(f"f{f}",), writes=f"f{f}"))
        p.append(Instr(ALU))
    return p


def prog_c1_warp(steps: int, passes: int, tk16: int = 2, m_tiles: int = 4, n_pairs: int = 2, chunks_per_stage: int = 4, stage_bytes: int = 16384,
                 copies: bool = True, compute: bool = True) -> list[Instr]:
    """C1: per k-step, the next stage's cp.async chunks (4 per thread), the wait, a barrier, then per k16: 4 A and 2 B ldmatrix.x4
    and 16 mma, then the trailing barrier. Stage s is double-buffered: slot s % 2. copies=False / compute=False are the kernel's
    no-copy and copy-only variants."""
    p = []
    stage = 0
    for _ in range(passes):
        if copies:
            p.append(Instr(COPY, stage=stage, chunks=chunks_per_stage, stage_bytes=stage_bytes)); p.append(Instr(ALU))
        for s in range(steps):
            if copies and s + 1 < steps:
                p.append(Instr(COPY, stage=stage + 1, chunks=chunks_per_stage, stage_bytes=stage_bytes))
            if copies:
                p.append(Instr(WAIT, stage=stage))
            p.append(Instr(BAR))
            for kk in range(tk16 if compute else 0):
                for i in range(m_tiles):
                    p.append(Instr(ALU)); p.append(Instr(LDSM, writes=(f"a{i}",), x=4))
                for j in range(n_pairs):
                    p.append(Instr(ALU)); p.append(Instr(LDSM, writes=(f"b{j}",), x=4))
                for i in range(m_tiles):
                    for j in range(2 * n_pairs):
                        p.append(Instr(MMA, reads=(f"a{i}", f"b{j // 2}", f"acc{i}_{j}"), writes=f"acc{i}_{j}"))
            p.append(Instr(BAR)); p.append(Instr(ALU))
            stage += 1
    return p


def prog_c3_row(rows: int) -> list[Instr]:
    """C3: one warp per row of 128 keys: a 16 B global load per lane, 3 fmax, a 5-step butterfly (shfl + fmax), 4 fsub+fmul, 4 ex2,
    3 fadd, a 5-step butterfly (shfl + fadd), rcp, 4 fmul, 2 packed bf16 converts, 2 stores, loop and address arithmetic."""
    p = []
    for r in range(rows):
        p.append(Instr(ALU, writes="addr")); p.append(Instr(GLOAD, reads=("addr",), writes="v", wavefronts=4.0))
        p.append(Instr(ALU, reads=("v",), writes="m"))
        for _ in range(2): p.append(Instr(ALU, reads=("m",), writes="m"))
        for d in range(5):
            p.append(Instr(SHFL, reads=("m",), writes="mx", op="bfly")); p.append(Instr(ALU, reads=("m", "mx"), writes="m"))
        for i in range(4):
            p.append(Instr(FFMA, reads=("v", "m"), writes=f"e{i}")); p.append(Instr(SFU, reads=(f"e{i}",), writes=f"e{i}", op="ex2"))
        p.append(Instr(FFMA, reads=("e0", "e1"), writes="sum")); p.append(Instr(FFMA, reads=("sum", "e2"), writes="sum")); p.append(Instr(FFMA, reads=("sum", "e3"), writes="sum"))
        for d in range(5):
            p.append(Instr(SHFL, reads=("sum",), writes="sx", op="bfly")); p.append(Instr(FFMA, reads=("sum", "sx"), writes="sum"))
        p.append(Instr(SFU, reads=("sum",), writes="inv", op="rcp"))
        for i in range(4): p.append(Instr(FFMA, reads=(f"e{i}", "inv"), writes=f"e{i}"))
        p.append(Instr(CVT, reads=("e0", "e1"), writes="p0", op="f2fp_bf16x2")); p.append(Instr(CVT, reads=("e2", "e3"), writes="p1", op="f2fp_bf16x2"))
        p.append(Instr(GSTORE, reads=("p0",))); p.append(Instr(GSTORE, reads=("p1",))); p.append(Instr(ALU))
    return p
