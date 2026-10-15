"""M5, the memory hierarchy (§2): L1 (bypassed by cp.async.cg), L2 as an LRU over regions with the
measured capacity, DRAM as a bandwidth-limited queue with the measured read/write rates.

The L2 capacity is the measured one: B13's single-warp chase keeps 91 % hits at 24 MiB and collapses at
32, and the counter cross-check reads 87 % hits through 24 MiB (`l2_chase_capacity_bytes`). B4's 16 MiB
streaming knee is then a prediction of the bandwidth queue under many concurrent streams, not an input
(B27_plus.md §3.1). What stays phenomenological (§9.2) is the set/slice structure: B13 found hashed sets
and no slice latency spread, so the LRU is over regions, fully associative. A region is one request's
footprint (a tile for the template, a chunk for a streaming kernel), so sizes stay exact and the LRU fast.

Timing of one request: lookup -> (hit) L2 latency + bytes through the L2 bandwidth queue, or (miss)
DRAM latency + bytes through the DRAM queue, then through L2. Every byte served by DRAM is counted,
which is the simulated traffic the regime label is computed from.
"""
from __future__ import annotations

from collections import OrderedDict

from .run import Engine, Fifo, Memory


class TmaEngine:
    """Round 2, group 2 (B35-B37): TMA as a second copy engine on the per-SM path. One bulk copy per tile completes after B35's
    measured p50 for that tile size and residency (every SM copying at once, so the chip's share is inside the number); a per-SM
    rate cap from the same trials; 2-D tensor-map copies scaled by B36 (c); the issue is one thread's two instructions plus the
    mbarrier round trip (B37). Nothing is fitted: the tables are the contract's."""
    def __init__(self, contract: dict):
        c = contract; self.hz = float(c.get("clock_hz_sustained") or 2.4e9)
        self.lat = {}                                     # (tile, resident) -> p50 cycles
        for k, v in (c.get("tma_latency_dist") or {}).items():
            kv = dict(x.split("=") for x in k.split(",")); self.lat[(int(kv["tile"]), int(kv["resident"]))] = float(v["p50_cycles"])
        self.rate = {}
        for k, v in (c.get("tma_bytes_per_s_per_sm_by_residency") or {}).items():
            kv = dict(x.split("=") for x in k.split(",")); self.rate[(int(kv["tile"]), int(kv["resident"]))] = float(v)
        self.ratio_2d = float(c.get("tma_2d_vs_1d_ratio") or 1.0)
        mb = c.get("mbarrier_cycles") or {}
        self.mbarrier_cycles = float(mb.get("arrivals=1") or (float(c["mbarrier_s"]) * self.hz if c.get("mbarrier_s") else 100.0))
        self.single = {"16384": float(c["tma_bulk_latency_s"]["16384"]) * self.hz} if c.get("tma_bulk_latency_s") else {}
        self.chip_rate = {}
        for k, v in (c.get("tma_latency_dist") or {}).items():
            kv = dict(x.split("=") for x in k.split(",")); self.chip_rate[(int(kv["tile"]), int(kv["resident"]))] = float(v.get("chip_bytes_per_s") or 0)
        self.chip_q = Fifo()

    def _interp(self, table: dict, tile: int, resident: int):
        if not table:
            return None
        res = sorted({r for _, r in table})
        r = max([x for x in res if x <= resident] or [res[0]])
        pts = sorted((t, v) for (t, rr), v in table.items() if rr == r)
        if tile <= pts[0][0]: return pts[0][1] * tile / pts[0][0] if len(pts) == 1 else pts[0][1]
        if tile >= pts[-1][0]: return pts[-1][1]
        for (a, va), (b, vb) in zip(pts, pts[1:]):
            if a <= tile <= b: return va + (vb - va) * (tile - a) / (b - a)
        return pts[-1][1]

    def latency_cycles(self, tile_bytes: int, resident: int, two_d: bool = False) -> float:
        lat = self._interp(self.lat, tile_bytes, resident)
        if lat is None:
            lat = self.single.get("16384", 1200.0) * tile_bytes / 16384
        return lat / (self.ratio_2d if two_d else 1.0)

    def rate_bytes_per_s(self, tile_bytes: int, resident: int) -> float | None:
        return self._interp(self.rate, tile_bytes, resident)

    def chip_bytes_per_s(self) -> float | None:
        """B35: the chip's TMA copy bandwidth, the largest chip rate the trials reached (tiles of 16 KB and up at any residency)."""
        rates = [v for (t, r), v in self.chip_rate.items() if t >= 16384]
        return max(rates) if rates else None

    def complete(self, sm: int, now: float, tile_bytes: int, resident: int, two_d: bool = False, sms: int = 48) -> float:
        """Completion time of a tile issued at `now` on SM `sm`. B35's per-SM rate at a residency is the chip's share of its copy
        bandwidth (16 KB x 48 SMs per 1176 cycles = 1.5 TB/s at one block, 1.9 at two), so the rule is: the tile lands after the
        one-block latency for its size, and no sooner than the chip's copy queue has moved every SM's tiles issued before it
        (`sms` identical SMs run the same program, so each tile costs sms x bytes of the queue). The C2 copy-only variant is the
        check: 922 / 1833 / 2076 cycles per step measured against this rule."""
        bw = self.chip_bytes_per_s()
        _, served = self.chip_q.serve(now, sms * tile_bytes / bw if bw else 0.0)
        return max(now + self.latency_cycles(tile_bytes, 1, two_d) / self.hz, served)


class Hierarchy(Memory):
    def __init__(self, contract: dict, structural: dict, fixed: dict):
        super().__init__(contract, structural, fixed)
        c = contract
        self.l2_capacity = int(c.get("l2_capacity_bytes") or c.get("l2_chase_capacity_bytes") or c["l2_bytes"])     # measured (counters, B13), not B4's streaming knee
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
        # M4: a per-SM throughput window of copy chunks in flight (B8's latency-vs-size slope after the measured per-chunk
        # issue cost is taken out, at the L2 latency; B27_plus.md §3.3-3.4). The issue cost is charged explicitly: each warp
        # issues its chunks serially at B15.3's cycles per 16 B. Completion of a tile = max(issue, window streaming) + the level
        # latency + the measured completion overhead of a cp.async group (B8's 4 KB intercept under this formula).
        self.copy_depth = int(structural.get("copy_throughput_window_chunks") or structural.get("copy_queue_depth", 256))
        self.copy_chunk = int(structural.get("copy_chunk_bytes", 16))
        hz = float(c.get("clock_hz_sustained") or 2.4e9)
        # B30.4 measured the cp.async issue itself at 6.8 cycles per chunk per warp (B15.3's 127 included the address and guard code,
        # which the warp model charges as per-phase overhead instructions), so the explicit issue term is small
        self.copy_issue_s_per_chunk_per_warp = float(c.get("cp_async_issue_cycles_per_chunk") or c.get("cp_async_issue_cycles_per_16B") or 0.0) / hz
        self.threads_per_block = int(c.get("threads_per_block") or 256)
        lat4k = (c.get("async_copy_latency_s") or {}).get("4096")
        self.copy_overhead = 0.0
        if lat4k:
            self.copy_overhead = max(0.0, float(lat4k) - self.copy_time(4096, self.lat_l2))
        self.windows: dict[int, Fifo] = {}
        self.tma = TmaEngine(c)

    def request_tma(self, engine: Engine, sm: int, src, bytes_: int, resident: int, done, two_d: bool = True):
        """A TMA tile copy (round 2, group 2): L2/DRAM traffic is counted as for cp.async; completion comes from the TMA engine."""
        now = engine.t
        if isinstance(src, tuple) and src[0] == "block":
            _, i, j, tile = src
            for k, nb in ((("A", i, tile), self.a_bytes), (("B", tile, j), self.b_bytes)):
                self.l2_bytes += nb
                if not self.lookup(k, nb):
                    self.dram_bytes += nb
        else:
            self.l2_bytes += bytes_
        engine.at(self.tma.complete(sm, now, bytes_, resident, two_d, int(self.c.get("sms") or 48)), done)

    def copy_issue_s(self, bytes_block: int) -> float:
        """Serial issue time of one warp's share of a block-wide copy: chunks per thread x the measured issue cost per chunk."""
        chunks_per_thread = bytes_block / (self.threads_per_block * self.copy_chunk)
        return chunks_per_thread * self.copy_issue_s_per_chunk_per_warp

    def lsu_entry_s(self, bytes_block: int) -> float:
        """The LSU entry the chunks occupy on their way to the window: one warp instruction per 512 B, four 128 B wavefronts each at one
        per cycle (B30), per SM."""
        hz = float(self.c.get("clock_hz_sustained") or 2.4e9)
        return (bytes_block / 512.0) * 4.0 / hz

    def copy_time(self, bytes_block: int, lat: float) -> float:
        """One block's tile of `bytes_block` from a level of latency `lat` with the SM's window to itself (B8's setting): the per-chunk
        issue on the sub-partition slot, the LSU entry, then the chunks stream through the window (B8's slope is their sum)."""
        window_rate = self.copy_depth * self.copy_chunk / lat
        return self.copy_issue_s(bytes_block) + self.lsu_entry_s(bytes_block) + bytes_block / window_rate + lat + self.copy_overhead

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
    def request(self, engine: Engine, sm: int, src, bytes_: int, done, write: bool = False, skip_issue: bool = False, delay: float = 0.0):
        """skip_issue: the warp model already dispatched the chunks (delay = when the last one left the LSU)."""
        now = engine.t + delay
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
                # depth * chunk / latency of the level; the share flows through the window and the level's bandwidth queue;
                # the warp's own chunk issue (serial, B15.3) must also have finished before the last chunk can be in flight
                window_rate = self.copy_depth * self.copy_chunk / lat
                t0, t_win = window.serve(now, share / window_rate)
                issue_end = now if skip_issue else now + self.copy_issue_s(share * (self.tile_bytes / max(1, bytes_)))   # the block's whole tile, per warp share
                _, t = q.serve(max(t_win, now) + (issue_end - now) + lat, nbytes / bw if not hit else share / bw)
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
