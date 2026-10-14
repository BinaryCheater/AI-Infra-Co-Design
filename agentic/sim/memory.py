"""M5, the memory hierarchy (§2): L1 (bypassed by cp.async.cg), L2 as an LRU over regions with the
measured effective capacity, DRAM as a bandwidth-limited queue with the measured read/write rates.

Phenomenological, and labeled so (§9.2): the GB10's L2 set/slice structure is not public; an LRU over
regions with capacity `l2_bytes_effective` reproduces B4's plateaus and places the knee at the
capacity, which is what the tiling template's reuse pattern needs (every block in a row panel reads
the same A tile at the same phase). A region is one request's footprint (a tile for the template, a
chunk for a streaming kernel), so sizes stay exact and the LRU stays fast.

Timing of one request: lookup -> (hit) L2 latency + bytes through the L2 bandwidth queue, or (miss)
DRAM latency + bytes through the DRAM queue, then through L2. Every byte served by DRAM is counted,
which is the simulated traffic the regime label is computed from.
"""
from __future__ import annotations

from collections import OrderedDict

from .run import Engine, Fifo, Memory


class Hierarchy(Memory):
    def __init__(self, contract: dict, structural: dict, fixed: dict):
        super().__init__(contract, structural, fixed)
        c = contract
        self.l2_capacity = int(c.get("l2_bytes_effective") or c["l2_bytes"])
        self.lat_l2 = float(c["latency_ns"]["l2"]) * 1e-9
        self.lat_dram = float(c["latency_ns"]["dram"]) * 1e-9
        self.l2_bw = float(c["l2_bytes_per_s"])
        self.dram_bw_read = float(c["dram_bytes_per_s"])
        self.dram_bw_write = float(c.get("dram_bytes_per_s_write") or c["dram_bytes_per_s"])
        self.l2 = OrderedDict()                 # region key -> bytes (LRU order)
        self.l2_used = 0
        self.l2_q = Fifo()
        self.dram_q = Fifo()
        self.hits = 0
        self.misses = 0
        self.policy = structural.get("l2_policy", "lru-fully-associative")
        # M4: a per-SM window of outstanding copy chunks (B8: the per-SM copy rate at the L2 latency fixes its depth);
        # a request of n chunks is served in ceil(n / depth) rounds, each paying the level latency, plus the measured
        # completion overhead of a cp.async group (B8's intercept above the L2 latency at the smallest tile)
        self.copy_depth = int(structural.get("copy_queue_depth", 256))
        self.copy_chunk = int(structural.get("copy_chunk_bytes", 16))
        lat4k = (c.get("async_copy_latency_s") or {}).get("4096")
        window_rate_l2 = self.copy_depth * self.copy_chunk / self.lat_l2
        self.copy_overhead = max(0.0, float(lat4k) - self.lat_l2 - 4096 / window_rate_l2) if lat4k else 0.0
        self.windows: dict[int, Fifo] = {}

    # ---- the LRU
    def lookup(self, key, bytes_: int) -> bool:
        if key in self.l2:
            self.l2.move_to_end(key)
            self.hits += 1
            return True
        self.misses += 1
        self.l2[key] = bytes_
        self.l2_used += bytes_
        while self.l2_used > self.l2_capacity and len(self.l2) > 1:
            _, b = self.l2.popitem(last=False)
            self.l2_used -= b
        return False

    # ---- a request
    def request(self, engine: Engine, sm: int, src, bytes_: int, done, write: bool = False):
        now = engine.t
        key = src if not isinstance(src, tuple) or src[0] != "block" else None
        if isinstance(src, tuple) and src[0] == "block":
            # the tiling template: ("block", i, j, tile) reads A tile (i, tile) and B tile (tile, j); each warp
            # requests its share (bytes_ of the tile). The first warp to miss brings the whole tile from DRAM
            # (every warp asks for its part at once); later warps hit in L2 and move only their share.
            _, i, j, tile = src
            frac = bytes_ / max(1, self.tile_bytes)
            window = self.windows.setdefault(sm, Fifo())
            t_end = now
            for k, nb in ((("A", i, tile), self.a_bytes), (("B", tile, j), self.b_bytes)):
                share = nb * frac
                self.l2_bytes += share
                hit = self.lookup(k, nb)
                if not hit:
                    self.dram_bytes += nb
                lat = self.lat_l2 if hit else self.lat_dram
                q, bw, nbytes = (self.l2_q, self.l2_bw, share) if hit else (self.dram_q, self.dram_bw_read, nb)
                # the SM's copy window: `copy_depth` chunks in flight per SM, so the window's own rate is
                # depth * chunk / latency of the level; the share flows through the window and the level's bandwidth queue
                window_rate = self.copy_depth * self.copy_chunk / lat
                t0, t_win = window.serve(now, share / window_rate)
                _, t = q.serve(max(t_win, now) + lat, nbytes / bw if not hit else share / bw)
                t_end = max(t_end, t + self.copy_overhead)
            engine.at(t_end, done)
            return
        engine.at(self._serve(now, key, bytes_, write), done)

    def _serve(self, now: float, key, bytes_: int, write: bool) -> float:
        self.l2_bytes += bytes_
        if write:
            self.dram_bytes += bytes_
            _, t = self.dram_q.serve(now + self.lat_l2, bytes_ / self.dram_bw_write)
            return t
        if self.lookup(key, bytes_):
            _, t = self.l2_q.serve(now + self.lat_l2, bytes_ / self.l2_bw)
            return t
        # a miss is bounded by the DRAM queue; the L2 pipe (6x faster) is not charged for it, which keeps
        # hits issued after a miss from waiting behind its DRAM return (no head-of-line blocking across levels)
        self.dram_bytes += bytes_
        _, t_dram = self.dram_q.serve(now + self.lat_dram, bytes_ / self.dram_bw_read)   # lat_dram is the measured end-to-end miss latency
        return t_dram

    def request_write(self, engine: Engine, sm: int, key, bytes_: int):
        self._serve(engine.t, key, bytes_, write=True)

    def bind(self, d, w, meta: dict):
        """Per-design facts the template's requests need (set by sim.tiling.simulate_design)."""
        self.a_bytes = meta["a_tile_bytes"]
        self.b_bytes = meta["b_tile_bytes"]
        self.tile_bytes = meta["tile_bytes"]

    def stats(self) -> dict:
        return {"l2_hits": self.hits, "l2_misses": self.misses, "l2_capacity": self.l2_capacity, "copy_depth": self.copy_depth,
                "copy_overhead_s": self.copy_overhead,
                "dram_bytes": self.dram_bytes, "l2_bytes": self.l2_bytes, "dram_busy_s": self.dram_q.busy, "l2_busy_s": self.l2_q.busy}


# ------------------------------------------------------------------------ benchmark replays (B4, B5)
def stream_bandwidth(contract: dict, structural: dict, fixed: dict, working_set: int, passes: int = 3,
                     chunk: int = 4096, write: bool = False) -> dict:
    """B4 as a mechanism test: `passes` cyclic sweeps over `working_set` bytes in `chunk` requests from
    every SM at once; bandwidth = bytes / simulated time (first pass warms the LRU, like the benchmark's warm-up)."""
    h = Hierarchy(contract, structural, fixed)
    eng = Engine()
    sms = int(contract["sms"]); outstanding = 8               # blocks per SM in the benchmark
    n = max(1, working_set // chunk)
    done = {"t": 0.0}
    def finish(): done["t"] = max(done["t"], eng.t)
    # warm pass (not timed)
    for i in range(n):
        h.request(eng, i % sms, ("chunk", i), chunk, finish, write)
    eng.run()
    t0 = eng.t; h.dram_bytes = 0; h.l2_bytes = 0; done["t"] = t0
    # the SMs keep requests streaming (8 blocks per SM, grid-stride loops): issue every chunk of every pass
    # in order and let the bandwidth queues serialize them; the first request pays one latency
    for _ in range(passes):
        for i in range(n):
            h.request(eng, i % sms, ("chunk", i), chunk, finish, write)
        eng.run()
    t = done["t"] - t0
    return {"bytes": passes * n * chunk, "seconds": t, "bytes_per_s": passes * n * chunk / t if t else 0.0, **h.stats()}


def pointer_chase(contract: dict, structural: dict, fixed: dict, working_set: int, steps: int = 2000, line: int = 64) -> float:
    """B5: dependent loads over a random cycle of lines; returns seconds per step (latency at that rung)."""
    h = Hierarchy(contract, structural, fixed)
    lat_l1 = float(contract["latency_ns"]["l1"]) * 1e-9
    l1_bytes = 64 * 1024                                     # per-SM L1: a request within it is served at the L1 latency
    if working_set <= l1_bytes:
        return lat_l1
    eng = Engine(); n = max(1, working_set // line)
    for i in range(n):                                       # warm: touch every line once
        h.lookup(("line", i), line)
    t0 = 0.0; t = [0.0]
    def step(i):
        def done(): t[0] = eng.t
        h.request(eng, 0, ("line", i), line, done)
    for i in range(steps):
        eng.t = t[0]
        step((i * 7919) % n); eng.run()
    return (t[0] - t0) / steps
