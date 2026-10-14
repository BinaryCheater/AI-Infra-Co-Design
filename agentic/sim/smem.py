"""M3, shared memory as a resource: `banks` banks of `bank_bytes`, one wavefront per warp-level access when
conflict-free; a stride of s words across the warp touches ceil(s * 4 / bank_bytes) words per bank and
serializes by that degree (B6: strides 1 and 2 conflict-free, penalty = stride / 2 from 4 on, so banks
are 8 bytes wide). The pipe's rate is the measured stride-1 bandwidth per SM in wavefronts of 128 B.
"""
from __future__ import annotations

from .run import Fifo


class SharedMemory:
    def __init__(self, contract: dict, structural: dict, fixed: dict):
        self.banks = int(structural.get("banks", 32))
        self.bank_bytes = int(structural.get("bank_bytes", 8))
        self.wavefront_bytes = self.banks * 4                 # 32 words per warp access
        self.rate = float(contract["smem_bytes_per_s_per_sm"]) / self.wavefront_bytes    # wavefronts per second per SM
        self.pipes: dict[int, Fifo] = {}

    def degree(self, stride_words: int) -> int:
        if stride_words <= 1:
            return 1
        return max(1, (stride_words * 4) // self.bank_bytes)

    def service(self, wavefronts: int, stride_words: int) -> float:
        return wavefronts * self.degree(stride_words) / self.rate

    def serve(self, sm: int, now: float, wavefronts: int, stride_words: int) -> tuple[float, float]:
        pipe = self.pipes.setdefault(sm, Fifo())
        return pipe.serve(now, self.service(wavefronts, stride_words))

    def busy(self) -> float:
        return sum(p.busy for p in self.pipes.values())
