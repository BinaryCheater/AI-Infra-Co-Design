"""M1/M7 warp issue model (B27_plus.md §2): the compute phase as in-order warps with a scoreboard on four schedulers,
sharing one LSU per SM and one FMA pipe per scheduler. Replaces sim/issue.py's additive occupancy.

Per warp: a program counter over the phase's instruction sequence and a scoreboard (cycle at which each pending result
is ready). The sequence is what the template's events carry: `other` overhead instructions (ALU class) at the phase
start, then per k-step: the A loads (ld.shared.128, two addresses per warp), the B loads (ld.shared.32), the FFMAs that
consume that step's loads, one loop/address instruction. The compiler hoists nothing across k-steps in this reading
(in-order issue; the SASS of the generated kernels shows loads then FFMAs per step), so a warp stalls on its own
scoreboard until its loads return and other warps issue in its place.

Schedulers: four per SM, warp id mod 4 (across all resident blocks); each cycle a scheduler issues one instruction from
the oldest eligible warp (operands ready; the structural resource free).
Shared resources (B30): each sub-partition dispatches one shared-memory instruction every `mio_issue_cycles_per_load`
cycles into the queue (a warp whose load cannot dispatch is not eligible: `mio_throttle`); the SM's LSU drains the queue
in order at `lsu_drain_wavefronts_per_cycle_per_sm` with at least `lsu_min_cycles_per_instruction` per instruction, an
entry costing its wavefronts (width x conflict degree, the B30 ncu table); the FMA pipe accepts one FFMA per scheduler
per the B2 rate. cp.async chunks do not use this queue (B30.4); ldmatrix would.
Latencies: scoreboard_latency_cycles from B27 (LDS 27, FFMA/IMAD 4), confirming B17.

Outputs per phase: cycles and the wait-family histogram over warp-cycles: issued, scoreboard (operands pending),
not_selected (eligible, another warp chosen), lsu (blocked by the LSU queue), done (finished early). One phase per
(design, residency) is simulated and cached; the event-driven reader multiplies by the phase count.
"""
from __future__ import annotations

import math
from functools import lru_cache

LDS128, LDS32, FFMA, ALU, COPY = 0, 1, 2, 3, 4
SFU, SHFL, CVT = 5, 6, 7          # round 2, group 3 (B38, B39): the XU pipe, warp shuffles, conversions


class WarpModel:
    _rec = None                                       # a recorder (sim/microtrace.py) while trace_phase runs; None otherwise

    def __init__(self, contract: dict, structural: dict):
        hz = float(contract.get("clock_hz_sustained") or 2.4e9)
        sb = contract.get("scoreboard_latency_cycles") or {}
        self.lat = {LDS128: float(sb.get("lds") or (contract.get("op_latency_cycles") or {}).get("ld_shared") or 27),
                    LDS32: float(sb.get("lds") or (contract.get("op_latency_cycles") or {}).get("ld_shared") or 27),
                    FFMA: float(sb.get("fma") or 4.0), ALU: float(sb.get("fma") or 4.0)}
        self.schedulers = int(contract.get("schedulers_per_sm") or 4)
        # B30: the queue between a sub-partition's scheduler and the LSU. A sub-partition dispatches one shared-memory instruction
        # every mio_issue_cycles_per_load cycles (4.0); no depth limit appears within 64 back-to-back loads from one warp; the LSU,
        # shared by the SM, drains lsu_drain_wavefronts_per_cycle_per_sm (1.0) with a floor per instruction that B30's occupancy
        # sweep gives as a function of resident warps per SM (2.4 cycles at 4 warps, 2.0 from 16 up); wavefronts per instruction by width from the ncu table (32-bit 1, 64-bit 2, 128-bit 4, times the stride degree)
        self.mio_issue = float(contract.get("mio_issue_cycles_per_load") or 4.0)
        self.lsu_drain_wf = float(contract.get("lsu_drain_wavefronts_per_cycle_per_sm") or 1.0)
        # the floor by occupancy (B30 occupancy sweep): SM cycles per one-wavefront load vs resident warps per SM, interpolated
        self.floor_by_warps = {int(k): float(v) for k, v in (contract.get("lsu_instr_floor_cycles_by_warps_per_sm") or {}).items()} or {1: float(contract.get("lsu_min_cycles_per_instruction") or 2.0)}
        self.lsu_min_cycles = self.floor_by_warps[max(self.floor_by_warps)]
        wft = contract.get("lds_wavefronts_per_instruction") or {}
        self.wavefronts = {LDS32: float(wft.get("width=4,stride=1") or 1.0), LDS128: float(wft.get("width=16,stride=1") or 4.0)}
        # the template's A loads (ld.shared.128 with two distinct 16 B addresses per warp) move two 16 B segments: one wavefront
        # by the quarter-warp rule, taken as one (B30 measured 32 distinct addresses only; recorded in notes/sim_gaps.md)
        self.wavefronts_broadcast128 = 1.0
        self.mio_queue_depth = contract.get("mio_queue_depth_per_smsp")        # None: no depth limit observed (B30.1)
        # cp.async chunks (16 B per lane, 512 B per warp instruction) go through the same sub-partition slot and LSU entry as shared
        # loads: B30.4's 6.8 cycles per chunk per warp on the slot, four 128 B wavefronts in the LSU (B30_plus round 4, §1)
        self.copy_issue = float(contract.get("cp_async_issue_cycles_per_chunk") or 6.8)
        self.copy_wavefronts = 4.0
        # structural: "shared" (B30_plus round 4 §1: chunks queue with the loads) or "separate" (B30.4 measured cp.async issue unaffected
        # by a preceding burst of loads); with "separate" the copies do not enter the phase's instruction stream
        self.copy_path = str(structural.get("copy_path", "shared"))
        self.fma_cost = 1.0 / min(1.0, float(contract["fp32_flops_per_s_per_sm"]) / 64.0 / hz / self.schedulers)
        self.penalty = {int(k): float(v) for k, v in (contract.get("bank_conflict_penalty") or {}).items()}
        # B38: the SFU (XU) pipe per sub-partition: issue cost in cycles per warp instruction = 32 lanes / (thread-ops per cycle per SM / 4),
        # latency from the dependent chain less the loop's cycle; B39: shuffles the same way; conversions on their own rate
        def per_warp_cost(ops_per_s):
            return 32.0 / max(1e-9, float(ops_per_s) / hz / self.schedulers) if ops_per_s else None
        sfu_tp = contract.get("sfu_throughput_per_sm") or {}; sfu_lat = contract.get("sfu_latency_cycles") or {}
        self.sfu_cost = {op: per_warp_cost(v) for op, v in sfu_tp.items()}
        # the counter reading of the XU pipe (B38 ncu: one MUFU per 8 cycles per sub-partition) overrides the clock64 table, whose
        # eight-concurrent-blocks assumption the counter contradicts
        xu = contract.get("sfu_warp_instr_per_cycle_per_smsp")
        if xu:
            self.sfu_cost = {op: 1.0 / float(xu) for op in (list(sfu_tp) or ["ex2", "rcp", "rsqrt", "lg2", "tanh"])}
        self.sfu_lat = {op: max(1.0, float(v) - 1.0) for op, v in sfu_lat.items()}
        cvt_tp = contract.get("cvt_throughput_per_sm") or {}
        self.cvt_cost = {op: per_warp_cost(v) for op, v in cvt_tp.items()}
        if xu:
            self.cvt_cost = {op: (1.0 / float(xu) if op.startswith("f2") else c) for op, c in self.cvt_cost.items()}
        shfl_tp = contract.get("shfl_throughput_per_sm") or {}; shfl_lat = contract.get("shfl_latency_cycles") or {}
        self.shfl_cost = {m: (1.0 / max(1e-9, float(v) / hz / self.schedulers)) for m, v in shfl_tp.items()}       # warp instructions per cycle per sub-partition
        sh = contract.get("shfl_warp_instr_per_cycle_per_smsp")
        if sh:
            self.shfl_cost = {m: 1.0 / float(sh) for m in (list(shfl_tp) or ["bfly", "up", "idx"])}
        self.shfl_wavefronts = float(contract.get("shfl_wavefronts_per_instruction") or 1.0)
        self.shfl_lat = {m: max(1.0, float(v) - 1.0) for m, v in shfl_lat.items()}
        self.shfl_in_lsu_queue = bool(contract.get("shfl_in_lsu_queue", False))
        # B38 / B39: one warp issues an SFU or shuffle instruction only every so many cycles (8.3 for ex2, 8.7 for shfl.bfly), while
        # the pipe's aggregate rate above needs several warps; the warp is held for that interval (as the mma hold, B34)
        self.sfu_hold = float(contract.get("sfu_issue_cycles_one_warp") or 0.0)
        self.shfl_hold = float(contract.get("shfl_issue_cycles_one_warp") or 0.0)
        self.sfu_ffma_overlap = contract.get("sfu_ffma_overlap")
        # B40 (c) / B9: the register budget: blocks per SM by registers per thread (measured table), else the allocation rule
        self.blocks_by_regs = {int(k): int(v["measured"] if isinstance(v, dict) else v) for k, v in (contract.get("max_blocks_per_sm_by_regs") or {}).items()}
        g = contract.get("reg_alloc_granularity")
        self.reg_granularity = int(g) if isinstance(g, (int, float)) else int((g or {}).get("granularity") or 8)
        self.regs_per_sm = int(contract.get("regs_per_sm") or contract.get("registers_per_sm") or 65536)

        # the instruction-level input (notes/instruction_stream.md): "counts" builds each phase from per-step instruction counts;
        # "stream" runs the kernel's SASS stream (sim/sass.py) through the same schedulers, scoreboard, LSU queue and pipes
        self.input = str(structural.get("input", "counts"))
        self._stream = None
        self._costs = None

    def set_stream(self, block_streams: list, costs) -> None:
        """The per-warp phase streams of one block (sim/sass.gemm_phase_stream) and the pipe costs (sim/isa.PipeCosts)."""
        self._stream = block_streams
        self._costs = costs

    def run_stream(self, block_streams: list, warps_per_block: int, blocks: int, costs, *, mode: str = "phase", a_wavefronts: tuple = (),
                   b_wavefronts: tuple = (), deg: float = 1.0, copy_rule=None, stage_bytes: int = 0, trace=None) -> tuple[float, dict]:
        """The stream front end. Each warp issues its instructions in order: an instruction issues when the registers it reads (and
        the ones it writes) are ready, its sub-partition's pipe is free, and the scheduler picks it (one per sub-partition per cycle,
        oldest warp first). Pipes per sub-partition: FMA (FFMA, FADD, FMUL, IMAD... at the B21 rate relative to FFMA), ALU (IADD3,
        LOP3, SHF, ISETP... at their B21 rates), XU (MUFU and the F2F/F2FP converts, one per 8 cycles, B38), tensor (16.1 cycles per
        HMMA, B31), the shuffle unit (one per 8 cycles, B39); shared and global memory instructions and shuffles dispatch through the
        B30 queue (one per 4.2 cycles per sub-partition) and drain at the SM's LSU, one wavefront per cycle with the occupancy floor.
        Latencies by scoreboard class from isa.PipeCosts. mode "phase": barriers, copy waits and copy issue are issue slots only (the
        event-level simulator does them); mode "kernel": BAR waits for the block, LDGSTS/LDGDEPBAR/DEPBAR keep copy groups whose
        completion comes from `copy_rule(stage_bytes) -> (latency cycles, chip service cycles)`."""
        from .isa import FMA as P_FMA, ALU as P_ALU, LSU as P_LSU, TENSOR as P_TEN, XU as P_XU, COPY as P_COPY, BAR as P_BAR, WAIT as P_WAIT, CONST as P_CONST
        S = self.schedulers; W = warps_per_block * blocks
        streams = [block_streams[w % warps_per_block] for w in range(W)]
        n = [len(x) for x in streams]
        lat = costs.latency
        pc = [0] * W; ready = [dict() for _ in range(W)]; hold = [0.0] * W; last = [-1] * W
        fma_free = [0.0] * S; alu_free = [0.0] * S; xu_free = [0.0] * S; ten_free = [0.0] * S; shfl_free = [0.0] * S; mio_free = [0.0] * S; lsu_free = 0.0
        floor = self.floor(W)
        hist = {"issued": 0, "scoreboard": 0, "not_selected": 0, "lsu": 0, "tensor": 0, "xu": 0, "hold": 0, "barrier": 0, "copy_wait": 0, "done": 0}
        bar_arr = [dict() for _ in range(blocks)]; bar_idx = [0] * W
        grp_open = [0] * W; grp_last = [dict() for _ in range(blocks)]; grp_done = [dict() for _ in range(blocks)]; bw_seen = [set() for _ in range(blocks)]; bw_free = 0.0
        width_wf = {1: 1.0, 2: 1.0, 4: 1.0, 8: 2.0, 16: 4.0}
        alive = W; t = 0
        while alive > 0 and t < 20_000_000:
            for s in range(S):
                issued = False
                for w in range(s, W, S):
                    if pc[w] >= n[w]:
                        continue
                    if last[w] >= t or hold[w] > t + 0.5:
                        hist["hold"] += 1 if hold[w] > t + 0.5 else 0; continue
                    ins = streams[w][pc[w]]; b = w // warps_per_block; rw = ready[w]
                    if any(rw.get(r, 0.0) > t + 0.5 for r in ins.src) or any(rw.get(r, 0.0) > t + 0.5 for r in ins.dst):
                        hist["scoreboard"] += 1; continue
                    p = ins.pipe
                    if mode == "kernel" and p == P_BAR:
                        k = bar_idx[w]; arr = bar_arr[b].setdefault(k, set()); arr.add(w)
                        if len(arr) < min(warps_per_block, W - b * warps_per_block):
                            hist["barrier"] += 1; continue
                    if mode == "kernel" and p == P_WAIT:
                        g = grp_open[w] - ins.count - 1
                        if g >= 0 and grp_done[b].get(g, 0.0) > t + 0.5:
                            hist["copy_wait"] += 1; continue
                    if p == P_FMA and fma_free[s] > t + 0.5:
                        hist["not_selected"] += 1; continue
                    if p == P_ALU and alu_free[s] > t + 0.5:
                        hist["not_selected"] += 1; continue
                    if p == P_XU and xu_free[s] > t + 0.5:
                        hist["xu"] += 1; continue
                    if p == P_TEN and ten_free[s] > t + 0.5:
                        hist["tensor"] += 1; continue
                    if p == P_LSU and (mio_free[s] > t + 0.5 or (ins.sb_class == "shfl" and shfl_free[s] > t + 0.5)):
                        hist["lsu"] += 1; continue
                    if issued:
                        hist["not_selected"] += 1; continue
                    # issue
                    cls = ins.sb_class
                    if p == P_FMA:
                        fma_free[s] = max(fma_free[s], float(t)) + ins.cost
                    elif p == P_ALU:
                        alu_free[s] = max(alu_free[s], float(t)) + ins.cost
                    elif p == P_XU:
                        xu_free[s] = max(xu_free[s], float(t)) + ins.cost
                    elif p == P_TEN:
                        ten_free[s] = t + ins.cost
                    if p == P_LSU:
                        if cls == "shfl":
                            shfl_free[s] = max(shfl_free[s], float(t)) + ins.cost; wf = 1.0
                        elif cls == "ldsm":
                            wf = 4.0 if ".X4" in ins.opcode else (2.0 if ".X2" in ins.opcode else 1.0)
                        elif ins.pattern == "template:A" and a_wavefronts:
                            wf = a_wavefronts[ins.tidx % len(a_wavefronts)] * deg
                        elif ins.pattern == "template:B" and b_wavefronts:
                            wf = b_wavefronts[ins.tidx % len(b_wavefronts)] * deg
                        elif cls == "mbar":
                            wf = 1.0
                        else:
                            wf = width_wf.get(ins.width_bytes or 4, 1.0) * (deg if cls in ("lds", "sts") else 1.0)
                        mio_free[s] = t + self.mio_issue
                        start = max(float(t), lsu_free); lsu_free = start + max(floor, wf / self.lsu_drain_wf)
                        done_at = lsu_free + lat.get(cls, 0.0)
                    else:
                        done_at = t + lat.get(cls, lat["alu"])
                    for r in ins.dst:
                        rw[r] = done_at
                    if ins.hold and not (p == P_COPY and mode == "phase"):
                        hold[w] = t + ins.hold
                    if mode == "kernel":
                        if p == P_BAR:
                            bar_idx[w] += 1
                        elif p == P_COPY:
                            g = grp_open[w]; grp_last[b][g] = max(grp_last[b].get(g, 0.0), t + ins.hold)
                        elif ins.opcode.startswith("LDGDEPBAR"):
                            g = grp_open[w]; grp_open[w] += 1
                            lat_c, service = copy_rule(stage_bytes) if copy_rule else (0.0, 0.0)
                            if g not in bw_seen[b]:
                                bw_seen[b].add(g); bw_free = max(bw_free, float(t)) + service
                            grp_done[b][g] = max(grp_last[b].get(g, float(t)) + lat_c, bw_free)
                    pc[w] += 1; last[w] = t; hist["issued"] += 1; issued = True
                    if trace: trace(t, w, s, ins)
                    if pc[w] >= n[w]:
                        alive -= 1
            hist["done"] += sum(1 for w in range(W) if pc[w] >= n[w])
            t += 1
        hist["copy_dispatch_cycles"] = 0.0; hist["copy_dispatch_by_block"] = ()
        return float(t), hist

    def blocks_per_sm_by_regs(self, regs_per_thread: int, threads_per_block: int) -> int:
        """The residency the register file allows (B40 (c): the measured table where it has a point, else the allocation rule with
        the B19 granularity), before shared memory and the block limit are applied."""
        if self.blocks_by_regs:
            keys = sorted(self.blocks_by_regs)
            at = [k for k in keys if k >= regs_per_thread]
            if at:
                return self.blocks_by_regs[at[0]]
        g = self.reg_granularity
        per_warp = ((regs_per_thread * 32 + g * 32 - 1) // (g * 32)) * g * 32
        return max(1, self.regs_per_sm // (per_warp * (threads_per_block // 32)))

    def floor(self, warps_per_sm: int) -> float:
        keys = sorted(self.floor_by_warps)
        if warps_per_sm <= keys[0]:
            return self.floor_by_warps[keys[0]]
        for a, b in zip(keys, keys[1:]):
            if a <= warps_per_sm <= b:
                f = (warps_per_sm - a) / (b - a)
                return self.floor_by_warps[a] + f * (self.floor_by_warps[b] - self.floor_by_warps[a])
        return self.floor_by_warps[keys[-1]]

    def degree(self, stride_words: int) -> float:
        if stride_words <= 1 or not self.penalty:
            return 1.0
        keys = sorted(self.penalty)
        return self.penalty[max([x for x in keys if x <= stride_words] or [keys[0]])]

    @lru_cache(maxsize=8192)
    def phase(self, warps_per_block: int, blocks: int, k_steps: int, ffma: int, lds32: int, lds128: int, other: int = 0, alu: int = 1, stride_words: int = 1,
              a_wavefronts: tuple = (), b_wavefronts: tuple = (), copies: int = 0):
        """Cycles for one phase with `blocks` resident blocks of `warps_per_block` warps starting together, and the wait histogram.
        a_wavefronts: one entry per A load (128-bit, one per RM row per unrolled body of four steps); b_wavefronts: per B load per step
        (32-bit). When absent, the width table applies (lds128 loads per step at 4 wavefronts, lds32 at 1)."""
        if self.input == "stream" and self._stream is not None:
            return self.run_stream(self._stream, warps_per_block, blocks, self._costs, mode="phase", a_wavefronts=tuple(a_wavefronts),
                                   b_wavefronts=tuple(b_wavefronts), deg=self.degree(stride_words))
        S = self.schedulers
        # the compiler's schedule (SASS of the generated kernels, #pragma unroll 4): per body of U k-steps, a burst of the A loads
        # (ld.shared.128, one per 4 steps per RM row) plus step 0's B loads, then the FFMAs of each step with the next step's B
        # loads spread among them; an FFMA depends on its step's loads
        U = 4
        prog = []                                     # (class, dep_step): the k-step whose loads this FFMA needs, else -1
        if self.copy_path != "shared":
            copies = 0
        for _ in range(copies):                       # the next tile's cp.async chunks, issued at the phase start (program order)
            prog.append((COPY, -1))
        for _ in range(other):
            prog.append((ALU, -1))
        a_wf = list(a_wavefronts) if a_wavefronts else [self.wavefronts[LDS128]] * (lds128 * U)
        b_wf = list(b_wavefronts) if b_wavefronts else [self.wavefronts[LDS32]] * lds32
        wfs = []                                      # wavefronts per program entry (loads only)
        for k0 in range(0, k_steps, U):
            steps = list(range(k0, min(k0 + U, k_steps)))
            prog += [(LDS128, steps[0])] * len(a_wf) + [(LDS32, steps[0])] * len(b_wf)
            wfs += list(a_wf) + list(b_wf)
            for idx, k in enumerate(steps):
                nxt = steps[idx + 1] if idx + 1 < len(steps) else None
                pending = [(LDS32, nxt)] * len(b_wf) if nxt is not None else []
                pending_wf = list(b_wf) if nxt is not None else []
                gap = max(1, ffma // max(1, len(pending))) if pending else ffma + 1
                for f in range(ffma):
                    prog.append((FFMA, k))
                    if pending and (f + 1) % gap == 0:
                        prog.append(pending.pop(0)); wfs.append(pending_wf.pop(0))
                prog += pending; wfs += pending_wf; prog += [(ALU, -1)] * alu
        n = len(prog)
        load_wf = iter(wfs)                           # wavefronts in load program order (identical for every warp)
        per_warp_wf = [list(wfs) for _ in range(warps_per_block * blocks)]
        W = warps_per_block * blocks
        pc = [0] * W; sched = [w % S for w in range(W)]
        step_loads_ready = [[0.0] * k_steps for _ in range(W)]           # per warp per step: when the last load the step depends on returns
        last_issue = [0.0] * W                                            # the warp's previous instruction's issue cycle (in-order: one per cycle at most)
        last_ready = [0.0] * W                                            # when the warp's previous FFMA/ALU result is ready (chains of F >= 4 never bind; ALU before loads does)
        lsu_free = 0.0                                                    # the SM's LSU drain (wavefronts, in order)
        mio_free = [0.0] * S                                              # per sub-partition: when the next shared-memory instruction may dispatch
        mio_pending = [[] for _ in range(S)]                              # per sub-partition: drain times of dispatched entries (for the optional depth limit)
        fma_free = [0.0] * S
        hist = {"issued": 0, "scoreboard": 0, "not_selected": 0, "lsu": 0, "done": 0}
        copy_done = [0.0] * W                                             # per warp: when its last cp.async chunk left the LSU
        t = 0; alive = W; deg = self.degree(stride_words); floor = self.floor(W)
        age = list(range(W))
        rec = self._rec
        while alive > 0 and t < 50_000_000:
            for s in range(S):
                issued = False; eligible_seen = 0
                for w in age:
                    if sched[w] != s:
                        continue
                    if pc[w] >= n:
                        if rec: rec.state(t, w, "done")
                        continue
                    cls, dep = prog[pc[w]]
                    # in-order: a warp issues at most one instruction per cycle
                    if last_issue[w] >= t + 0.5 and pc[w] > 0:
                        hist["scoreboard"] += 0; continue
                    if cls == FFMA and step_loads_ready[w][dep] > t:
                        hist["scoreboard"] += 1
                        if rec: rec.state(t, w, "scoreboard")
                        continue
                    if cls in (LDS128, LDS32, COPY) and (mio_free[s] > t + 0.5 or (self.mio_queue_depth and sum(1 for c in mio_pending[s] if c > t) >= self.mio_queue_depth)):
                        hist["lsu"] += 1
                        if rec: rec.state(t, w, "mio")
                        continue                           # mio_throttle: the sub-partition's dispatch slot (or queue) is not free
                    if cls == FFMA and fma_free[s] > t + 0.5:
                        hist["not_selected"] += 1
                        if rec: rec.state(t, w, "ready")
                        continue
                    eligible_seen += 1
                    if issued:
                        hist["not_selected"] += 1
                        if rec: rec.state(t, w, "ready")
                        continue
                    # issue
                    if cls == COPY:
                        mio_free[s] = t + self.copy_issue
                        start = max(float(t), lsu_free); lsu_free = start + max(floor, self.copy_wavefronts / self.lsu_drain_wf)
                        copy_done[w] = lsu_free
                        if self.mio_queue_depth:
                            mio_pending[s] = [c for c in mio_pending[s] if c > t] + [lsu_free]
                        if rec: rec.load(t, w, s, cls, -1, self.copy_wavefronts, start, lsu_free, lsu_free)
                    elif cls in (LDS128, LDS32):
                        mio_free[s] = t + self.mio_issue
                        wf = (per_warp_wf[w].pop(0) if per_warp_wf[w] else self.wavefronts[cls]) * deg
                        start = max(float(t), lsu_free); lsu_free = start + max(floor, wf / self.lsu_drain_wf)
                        done_at = lsu_free + self.lat[cls]
                        if self.mio_queue_depth:
                            mio_pending[s] = [c for c in mio_pending[s] if c > t] + [lsu_free]
                        if rec: rec.load(t, w, s, cls, dep, wf, start, lsu_free, done_at)
                        if dep >= 0:
                            # an A load serves its whole unrolled body: every step of the body depends on it
                            targets = range(dep, min(dep + 4, k_steps)) if cls == LDS128 else (dep,)
                            for kk in targets:
                                step_loads_ready[w][kk] = max(step_loads_ready[w][kk], done_at)
                    elif cls == FFMA:
                        fma_free[s] = max(fma_free[s], float(t)) + self.fma_cost
                    pc[w] += 1; last_issue[w] = float(t); hist["issued"] += 1; issued = True
                    if rec: rec.issue(t, w, s, cls)
                    if pc[w] >= n:
                        alive -= 1
                # oldest-first within a scheduler: rotate the issued warp to the back so a stuck warp does not starve others
                if issued:
                    pass
            # every warp that finished its phase waits at the barrier
            hist["done"] += sum(1 for w in range(W) if pc[w] >= n)
            t += 1
        hist["copy_dispatch_cycles"] = max(copy_done) if copies else 0.0  # when the last chunk of the phase's copies left the LSU (block 0's view)
        hist["copy_dispatch_by_block"] = tuple(max(copy_done[b * warps_per_block:(b + 1) * warps_per_block]) for b in range(blocks)) if copies else ()
        return float(t), hist

    def trace_phase(self, rec, *args):
        """phase() run once more, uncached, with every issue, load and warp state handed to `rec` (the microarchitecture view's
        trace, notes/microarchitecture_view_design.md §4). Same arguments and result as phase()."""
        self._rec = rec
        try:
            return WarpModel.phase.__wrapped__(self, *args)
        finally:
            self._rec = None

    def phase_cycles(self, warps_per_block: int, blocks: int, k_steps: int, ffma: int, lds32: int, lds128: int, other: int = 0, stride_words: int = 1) -> float:
        return self.phase(warps_per_block, blocks, k_steps, ffma, lds32, lds128, other, 1, stride_words)[0]

    @staticmethod
    def counts(rm: int, rn: int) -> tuple[int, int, int]:
        return rm * rn, rn, int(math.ceil(rm / 4))

    def wait_families(self, warps_per_block: int, blocks: int, k_steps: int, ffma: int, lds32: int, lds128: int, other: int = 0, stride_words: int = 1) -> dict:
        return self.phase(warps_per_block, blocks, k_steps, ffma, lds32, lds128, other, 1, stride_words)[1]
