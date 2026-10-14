"""M7 issue model (G4): a tile phase's inner loop as instruction-issue occupancy on the SM's schedulers (gap 6).

What B17, B21 and the ncu counters established:
  - the generated inner loop is bound by instruction issue, not by the FMA pipe (ncu: `not_selected` dominant,
    FP32 pipes 62 % busy at 46 % of the FFMA bound);
  - per-block step time nearly doubles when the warps per SM double (B21 at 1-4 blocks per SM), so the cost is
    additive per scheduler: every warp-instruction occupies its scheduler for a fixed number of cycles;
  - the costs are the measured issue rates: FFMA 1 / (B2 rate per scheduler) ~ 1.07 cycles, ld.shared.32
    schedulers / (B21 lds_32 rate per SM) ~ 3.9 cycles, ld.shared.128 ~ 8.3 cycles (the MIO interface accepts
    one shared-memory instruction per sub-partition every few cycles), bank conflicts multiply the LDS cost
    by B6's penalty.
Per k-step a warp issues F = RM x RN FFMA, RN ld.shared.32 (the B row, one wavefront), ceil(RM / 4)
ld.shared.128 (the A column, four k-steps at once under the compiler's unroll) and about one ALU instruction.
A block's step time is the busiest scheduler's occupancy over the warps resident on that SM (blocks on one SM
share schedulers round-robin), floored by a single warp's dependent path (loads, the LDS latency, its FFMAs)
when the SM is nearly empty. Nothing is fitted: B21's inner loops at 1-4 blocks per SM are the test.
"""
from __future__ import annotations

import math


class IssueModel:
    def __init__(self, contract: dict, structural: dict):
        self.schedulers = int(contract.get("schedulers_per_sm") or structural.get("schedulers_per_sm") or 4)
        self.lds_latency = float((contract.get("op_latency_cycles") or {}).get("ld_shared") or 33)
        hz = float(contract.get("clock_hz_sustained") or 2.4e9)
        lds = contract.get("ld_shared_bytes_per_s_per_sm") or {}
        inst = contract.get("ld_shared_instr_per_s_per_sm") or {}
        # warp-level ld.shared instructions per SM cycle by width (B21: warp-instructions/s / Hz)
        r32 = (float(inst["lds_32"]) / hz) if "lds_32" in inst else ((float(lds["lds_32"]) / 128 / hz) if "lds_32" in lds else 1.0)
        r128 = (float(inst["lds_128"]) / hz) if "lds_128" in inst else ((float(lds["lds_128"]) / 512 / hz) if "lds_128" in lds else r32 / 2)
        # the template's A loads are ld.shared.128 with two distinct addresses per warp (threads 0-15 one row, 16-31 the next): B21
        # measures that broadcast pattern separately (lds_128_broadcast2, ~0.8 per SM cycle against 0.49 for 32 distinct addresses)
        r128b = (float(inst["lds_128_broadcast2"]) / hz) if "lds_128_broadcast2" in inst else r128
        # beyond four warps per scheduler the schedulers overlap part of the LDS occupancy (B21 at 3 and 4 blocks per SM: per-block
        # phase time 2.1x and 3.0x the one-block time instead of 3x and 4x); phenomenological, from those two points
        self.overlap = {int(k): float(v) for k, v in (structural.get("issue_overlap_by_warps_per_scheduler") or {1: 1.0, 2: 1.0, 4: 1.0, 6: 0.70, 8: 0.75}).items()}
        self.c_lds32 = self.schedulers / r32          # scheduler cycles one ld.shared.32 occupies
        # B21's inner loops (the same A-load pattern) are reproduced with the 32-address rate and under-read by 13-20 % with the
        # broadcast rate, so the 32-address rate stays; the broadcast measurement is kept in the contract (notes/sim_gaps.md #8)
        self.c_lds128 = self.schedulers / r128
        self.c_lds128_broadcast = self.schedulers / r128b
        ffma_per_sm_cycle = float(contract["fp32_flops_per_s_per_sm"]) / 64.0 / hz      # warp-FFMA per SM cycle (B2)
        self.c_ffma = self.schedulers / ffma_per_sm_cycle
        self.c_alu = 1.0
        self.penalty = {int(k): float(v) for k, v in (contract.get("bank_conflict_penalty") or {}).items()}
        # a lone warp issues one instruction per ~1.44 cycles (B17.4 at one warp per SM, the ncu issue counter)
        one = ((contract.get("counter_crosscheck") or {}).get("issue_by_warps") or {}).get("1", {}).get("inst_issued_per_active_cycle_per_sm")
        self.issue_interval = (1.0 / float(one)) if one else 1.0

    def degree(self, stride_words: int) -> float:
        if stride_words <= 1 or not self.penalty:
            return 1.0
        keys = sorted(self.penalty)
        k = max([x for x in keys if x <= stride_words] or [keys[0]])
        return self.penalty[k]

    def overlap_factor(self, per_sched: int) -> float:
        keys = sorted(self.overlap)
        if per_sched <= keys[0]:
            return self.overlap[keys[0]]
        for a, b in zip(keys, keys[1:]):
            if a <= per_sched <= b:
                f = (per_sched - a) / (b - a)
                return self.overlap[a] + f * (self.overlap[b] - self.overlap[a])
        return self.overlap[keys[-1]]

    def warp_step_cycles(self, ffma: int, lds32: int, lds128: int, alu: int = 1, stride_words: int = 1) -> float:
        """Scheduler cycles one warp's k-step occupies."""
        deg = self.degree(stride_words)
        return ffma * self.c_ffma + (lds32 * self.c_lds32 + lds128 * self.c_lds128) * deg + alu * self.c_alu

    def cycles_per_step(self, warps_per_sm: int, ffma: int, lds32: int, lds128: int, alu: int = 1, stride_words: int = 1) -> float:
        """SM cycles per k-step for a block when `warps_per_sm` warps share the SM (round-robin over the schedulers)."""
        if warps_per_sm <= 0:
            return 0.0
        per_sched = math.ceil(warps_per_sm / self.schedulers)
        occupancy = per_sched * self.overlap_factor(per_sched) * self.warp_step_cycles(ffma, lds32, lds128, alu, stride_words)
        # a lone warp cannot overlap its loads' latency with anything: dependent path = issue the loads, wait, issue the FFMAs
        lone = (lds32 * self.c_lds32 + lds128 * self.c_lds128) * self.degree(stride_words) + self.lds_latency + (ffma + alu) * self.issue_interval
        return max(occupancy, lone if per_sched == 1 else 0.0)

    def phase_cycles(self, warps_per_sm: int, k_steps: int, ffma: int, lds32: int, lds128: int, other_per_phase: int = 0, stride_words: int = 1) -> float:
        """SM cycles for one phase of a block: k_steps of the inner loop plus the per-phase overhead instructions, shared by the SM's warps."""
        per_sched = math.ceil(max(1, warps_per_sm) / self.schedulers)
        return self.cycles_per_step(warps_per_sm, ffma, lds32, lds128, 1, stride_words) * k_steps + per_sched * self.overlap_factor(per_sched) * other_per_phase * self.c_alu

    @staticmethod
    def counts(rm: int, rn: int) -> tuple[int, int, int]:
        """Per-warp, per-k-step instruction counts of the generated inner loop: (FFMA, ld.shared.32, ld.shared.128)."""
        return rm * rn, rn, int(math.ceil(rm / 4))
