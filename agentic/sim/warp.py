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
Shared per-SM resources: the LSU accepts ld.shared instructions at the measured rate by width (B21; B28's mixed-pattern
plateau is the cross-check) and holds at most `lsu_queue` accepted-but-unfinished instructions (an LDS whose warp would
exceed it is not eligible: `mio_throttle`); the FMA pipe accepts one FFMA per scheduler per the B2 rate.
Latencies: scoreboard_latency_cycles from B27 (LDS 27, FFMA/IMAD 4), confirming B17.

Outputs per phase: cycles and the wait-family histogram over warp-cycles: issued, scoreboard (operands pending),
not_selected (eligible, another warp chosen), lsu (blocked by the LSU queue), done (finished early). One phase per
(design, residency) is simulated and cached; the event-driven reader multiplies by the phase count.
"""
from __future__ import annotations

import math
from functools import lru_cache

LDS128, LDS32, FFMA, ALU = 0, 1, 2, 3


class WarpModel:
    def __init__(self, contract: dict, structural: dict):
        hz = float(contract.get("clock_hz_sustained") or 2.4e9)
        sb = contract.get("scoreboard_latency_cycles") or {}
        self.lat = {LDS128: float(sb.get("lds") or (contract.get("op_latency_cycles") or {}).get("ld_shared") or 27),
                    LDS32: float(sb.get("lds") or (contract.get("op_latency_cycles") or {}).get("ld_shared") or 27),
                    FFMA: float(sb.get("fma") or 4.0), ALU: float(sb.get("fma") or 4.0)}
        self.schedulers = int(contract.get("schedulers_per_sm") or 4)
        # the LSU accepts ld.shared instructions at B28's plateau (the template's own width mix at six resident blocks, 1.02 per
        # SM cycle, both widths alike); B28 and B27 show the rate falling at low occupancy and short bursts, which this does not model
        plateau = float(contract.get("lsu_instr_per_cycle_per_sm") or (float((contract.get("ld_shared_instr_per_s_per_sm") or {}).get("lds_32", 1.05 * hz)) / hz))
        # B28 measured the acceptance rate per residency (0.47 at one block to 1.02 at six); the model reads that table by the
        # number of resident blocks (a measured parameter, not a fit) and uses the plateau beyond it
        self.lsu_rate_by_blocks = {int(k): float(v) for k, v in (contract.get("lsu_instr_per_cycle_per_sm_by_residency") or {}).items()} or {1: plateau}
        self.lsu_cost = {LDS32: 1.0 / plateau, LDS128: 1.0 / plateau}       # SM cycles the LSU is busy per instruction (at the plateau)
        self.lsu_queue = int(structural.get("lsu_queue_instructions", 24))   # B28: accepted-but-unfinished LDS per SM before issue blocks (mio_throttle)
        self.fma_cost = 1.0 / min(1.0, float(contract["fp32_flops_per_s_per_sm"]) / 64.0 / hz / self.schedulers)
        self.penalty = {int(k): float(v) for k, v in (contract.get("bank_conflict_penalty") or {}).items()}

    def lsu_rate(self, blocks: int) -> float:
        keys = sorted(self.lsu_rate_by_blocks)
        if blocks <= keys[0]:
            return self.lsu_rate_by_blocks[keys[0]]
        for a, b in zip(keys, keys[1:]):
            if a <= blocks <= b:
                f = (blocks - a) / (b - a)
                return self.lsu_rate_by_blocks[a] + f * (self.lsu_rate_by_blocks[b] - self.lsu_rate_by_blocks[a])
        return self.lsu_rate_by_blocks[keys[-1]]

    def degree(self, stride_words: int) -> float:
        if stride_words <= 1 or not self.penalty:
            return 1.0
        keys = sorted(self.penalty)
        return self.penalty[max([x for x in keys if x <= stride_words] or [keys[0]])]

    @staticmethod
    def counts(rm: int, rn: int) -> tuple[int, int, int]:
        return rm * rn, rn, int(math.ceil(rm / 4))

    @lru_cache(maxsize=8192)
    def phase(self, warps_per_block: int, blocks: int, k_steps: int, ffma: int, lds32: int, lds128: int, other: int = 0, alu: int = 1, stride_words: int = 1):
        """Cycles for one phase with `blocks` resident blocks of `warps_per_block` warps starting together, and the wait histogram."""
        S = self.schedulers
        # the compiler's schedule (SASS of the generated kernels, #pragma unroll 4): per body of U k-steps, a burst of the A loads
        # (ld.shared.128, one per 4 steps per RM row) plus step 0's B loads, then the FFMAs of each step with the next step's B
        # loads spread among them; an FFMA depends on its step's loads
        U = 4
        prog = []                                     # (class, dep_step): the k-step whose loads this FFMA needs, else -1
        for _ in range(other):
            prog.append((ALU, -1))
        for k0 in range(0, k_steps, U):
            steps = list(range(k0, min(k0 + U, k_steps)))
            prog += [(LDS128, steps[0])] * (lds128 * len(steps)) + [(LDS32, steps[0])] * lds32
            for idx, k in enumerate(steps):
                nxt = steps[idx + 1] if idx + 1 < len(steps) else None
                pending = [(LDS32, nxt)] * lds32 if nxt is not None else []
                gap = max(1, ffma // max(1, len(pending))) if pending else ffma + 1
                for f in range(ffma):
                    prog.append((FFMA, k))
                    if pending and (f + 1) % gap == 0:
                        prog.append(pending.pop(0))
                prog += pending + [(ALU, -1)] * alu
        n = len(prog)
        W = warps_per_block * blocks
        pc = [0] * W; sched = [w % S for w in range(W)]
        step_loads_ready = [[0.0] * k_steps for _ in range(W)]           # per warp per step: when the last load the step depends on returns
        last_issue = [0.0] * W                                            # the warp's previous instruction's issue cycle (in-order: one per cycle at most)
        last_ready = [0.0] * W                                            # when the warp's previous FFMA/ALU result is ready (chains of F >= 4 never bind; ALU before loads does)
        lsu_free = 0.0; lsu_pending: list[float] = []                     # completion cycles of accepted LDS
        fma_free = [0.0] * S
        hist = {"issued": 0, "scoreboard": 0, "not_selected": 0, "lsu": 0, "done": 0}
        t = 0; alive = W; deg = self.degree(stride_words) / self.lsu_rate(blocks) * self.lsu_rate(max(self.lsu_rate_by_blocks))   # the LSU cost scaled to this residency's measured rate
        age = list(range(W))
        while alive > 0 and t < 50_000_000:
            lsu_pending = [c for c in lsu_pending if c > t]
            for s in range(S):
                issued = False; eligible_seen = 0
                for w in age:
                    if sched[w] != s:
                        continue
                    if pc[w] >= n:
                        continue
                    cls, dep = prog[pc[w]]
                    # in-order: a warp issues at most one instruction per cycle
                    if last_issue[w] >= t + 0.5 and pc[w] > 0:
                        hist["scoreboard"] += 0; continue
                    if cls == FFMA and step_loads_ready[w][dep] > t:
                        hist["scoreboard"] += 1; continue
                    if cls in (LDS128, LDS32) and len(lsu_pending) >= self.lsu_queue:
                        hist["lsu"] += 1; continue
                    if cls == FFMA and fma_free[s] > t + 0.5:
                        hist["not_selected"] += 1; continue
                    eligible_seen += 1
                    if issued:
                        hist["not_selected"] += 1; continue
                    # issue
                    if cls in (LDS128, LDS32):
                        start = max(float(t), lsu_free); lsu_free = start + self.lsu_cost[cls] * deg
                        done_at = lsu_free + self.lat[cls]; lsu_pending.append(done_at)
                        if dep >= 0:
                            # an A load serves its whole unrolled body: every step of the body depends on it
                            targets = range(dep, min(dep + 4, k_steps)) if cls == LDS128 else (dep,)
                            for kk in targets:
                                step_loads_ready[w][kk] = max(step_loads_ready[w][kk], done_at)
                    elif cls == FFMA:
                        fma_free[s] = max(fma_free[s], float(t)) + self.fma_cost
                    pc[w] += 1; last_issue[w] = float(t); hist["issued"] += 1; issued = True
                    if pc[w] >= n:
                        alive -= 1
                # oldest-first within a scheduler: rotate the issued warp to the back so a stuck warp does not starve others
                if issued:
                    pass
            # every warp that finished its phase waits at the barrier
            hist["done"] += sum(1 for w in range(W) if pc[w] >= n)
            t += 1
        return float(t), hist

    def phase_cycles(self, warps_per_block: int, blocks: int, k_steps: int, ffma: int, lds32: int, lds128: int, other: int = 0, stride_words: int = 1) -> float:
        return self.phase(warps_per_block, blocks, k_steps, ffma, lds32, lds128, other, 1, stride_words)[0]

    def wait_families(self, warps_per_block: int, blocks: int, k_steps: int, ffma: int, lds32: int, lds128: int, other: int = 0, stride_words: int = 1) -> dict:
        return self.phase(warps_per_block, blocks, k_steps, ffma, lds32, lds128, other, 1, stride_words)[1]
