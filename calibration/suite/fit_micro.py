"""Fitters for the structure suite B13-B26 (calibration/suite/B13_plus.md), one per benchmark.

Each fitter takes the raw records (quantity -> record) and returns (fields, sources): the contract
fields it fills and, per field, the raw quantities and the method it used. B13 and B18 infer structure;
their inference tables stay in `sources[...]["inferred_from"]` so a reader can check the inference.
Absent records leave fields out; a probe that could not run becomes {"measured": false, "reason": ...}.
"""
from __future__ import annotations

import re


def keyed(recs, prefix):
    out = {}
    for q, r in recs.items():
        m = re.fullmatch(re.escape(prefix) + r"\[(.+)\]", q)
        if m:
            out[m.group(1)] = r
    return out


def _mmm(series):
    xs = [x for x in (series or []) if isinstance(x, (int, float))]
    return [min(xs), sorted(xs)[len(xs) // 2], max(xs)] if xs else None


def _val(r):
    return None if r is None else r.get("value")


def _nm(r, q, method, **extra):
    d = {"from": q, "method": method}
    if r:
        d["ci95"] = r.get("ci95"); d["n"] = r.get("n")
    d.update(extra)
    return d


def _not_measured(recs, q):
    r = recs.get(q)
    if r and not r.get("samples") and r.get("detail", {}).get("measured") is False:
        return {"measured": False, "reason": r["detail"].get("reason")}
    return None


# ---------------------------------------------------------------- B13 L2 structure
def fit_b13(recs):
    f, s = {}, {}
    sec = recs.get("l2_sector_bytes")
    if sec:
        f["l2_sector_bytes"] = int(sec["value"])
        s["l2_sector_bytes"] = _nm(sec, "l2_sector_bytes", "B13.1: bytes per request at stride s (time x B4 DRAM rate / requests); the stride where it stops growing",
                                   inferred_from=sec["detail"].get("table_stride_seconds_bytesPerRequest"))
        f["l2_line_bytes"] = 128                            # the chase and set-conflict probes use 128 B lines; B14 measures the L1 line directly
        s["l2_line_bytes"] = {"from": "l2_sector_bytes", "method": "assumed 128 B (4 x the measured sector); not independently measured"}
    sc = recs.get("l2_set_conflicts")
    if sc:
        d = sc["detail"]
        ways = d.get("inferred_ways", -1)
        f["l2_ways"] = ways if ways > 0 else None
        f["l2_sets"] = d.get("inferred_sets_if_128B_lines") if d.get("inferred_sets_if_128B_lines", -1) > 0 else None
        if ways <= 0:
            f["l2_set_conflicts_observed"] = False
            f["l2_ways_lower_bound"] = 64      # the largest n probed: no stride up to 8 MiB conflicted with 64 lines
        s["l2_ways"] = _nm(sc, "l2_set_conflicts", "B13.2: n lines at stride s chased (ld.cg); at the largest stride the first n whose latency crosses the L2/DRAM midpoint gives the associativity",
                           inferred_from={"grid": d.get("grid"), "lat_l2_ns": d.get("lat_l2_ns"), "lat_dram_ns": d.get("lat_dram_ns"), "set_stride_bytes": d.get("inferred_set_stride_bytes"),
                                          "capacity_from_structure": d.get("inferred_capacity_bytes"), "declared_l2_bytes": d.get("declared_l2_bytes")})
        s["l2_sets"] = {"from": "l2_set_conflicts", "method": "B13.2: smallest conflicting stride / 128 B line; null when no power-of-two stride conflicts (a hashed set index spreads them)"}
        s["l2_set_conflicts_observed"] = {"from": "l2_set_conflicts", "method": "B13.2: whether any n <= 64 lines at any power-of-two stride up to 8 MiB chased at DRAM latency"}
        s["l2_ways_lower_bound"] = {"from": "l2_set_conflicts", "method": "B13.2: 64 lines at every power-of-two stride stayed at L2 latency, so the associativity is at least 64 or the set index is hashed"}
        cap = d.get("inferred_capacity_bytes", -1)
        f["l2_structure_confirmed"] = bool(cap and cap > 0 and d.get("declared_l2_bytes") and abs(cap - d["declared_l2_bytes"]) / d["declared_l2_bytes"] < 0.5)
        s["l2_structure_confirmed"] = {"from": "l2_set_conflicts", "method": "ways x sets x 128 B within 50 % of the declared L2 size"}
    sh = recs.get("l2_slice_hash")
    if sh:
        d = sh["detail"]
        nf = d.get("near_far_ns") or [0, 0]
        spread = (nf[1] - nf[0]) if len(nf) == 2 else 0
        bits = (d.get("inferred_slice_bits") or []) if spread >= 10 else []      # under 10 ns the per-bit differences are within the chase noise
        f["l2_slice_hash_bits"] = bits
        f["l2_slices"] = (1 << len(bits)) if bits else None
        f["l2_slice_latency_spread_ns"] = round(spread, 2)
        f["l2_latency_near_far_ns"] = d.get("near_far_ns")
        s["l2_slice_hash_bits"] = _nm(sh, "l2_slice_hash", "B13.3: two-line chase base vs base XOR 2^bit; a bit is slice-select when its flip moves the latency by more than half the near/far spread (spread > 5 ns)",
                                      inferred_from={"latency_ns_by_flipped_bit": d.get("latency_ns_by_flipped_bit"), "aggregate_bw_by_region_bit": d.get("aggregate_bw_by_region_bit")})
        s["l2_slices"] = {"from": "l2_slice_hash", "method": "2^(number of slice-select bits); null when no bit moves the latency by a clear margin (then the slices, if any, are not visible from one SM's chase)"}
        s["l2_slice_latency_spread_ns"] = {"from": "l2_slice_hash", "method": "B13.3: max - min of the two-line chase latency over the flipped bits 7..29 (a near/far slice difference would show as a bimodal spread)"}
        s["l2_latency_near_far_ns"] = {"from": "l2_slice_hash", "method": "min and max two-line chase latency over the flipped bits"}
    rp = recs.get("l2_replacement")
    if rp:
        d = rp["detail"]
        curve = d.get("hit_rate_by_footprint_mb_seq_rand") or []
        past = [(mb, sq, rd) for mb, sq, rd in curve if mb * (1 << 20) > 1.2 * 24 * (1 << 20)]
        at = [(mb, sq, rd) for mb, sq, rd in curve if mb == 24]
        if past and all(sq < 0.1 and rd < 0.1 for _, sq, rd in past):
            f["l2_replacement"] = "lru-like (a cyclic chase larger than the cache misses entirely in both orders)"
        elif past and any(rd > 0.2 for _, _, rd in past):
            f["l2_replacement"] = "protected/random-like (random order keeps hits past capacity)"
        else:
            f["l2_replacement"] = d.get("classification")
        f["l2_chase_capacity_bytes"] = max((mb for mb, sq, rd in curve if sq > 0.8), default=None)
        if f["l2_chase_capacity_bytes"]:
            f["l2_chase_capacity_bytes"] *= 1 << 20
        s["l2_chase_capacity_bytes"] = {"from": "l2_replacement", "method": "B13.4: largest footprint whose sequential cyclic chase still hits above 80 % (a single warp sees the whole cache; B4's bandwidth knee is the streaming view)"}
        s["l2_replacement"] = _nm(rp, "l2_replacement", "B13.4: hit rate from mean chase latency vs footprint, sequential vs random cyclic order; classified from the points past 1.2x the declared size",
                                  inferred_from={"hit_rate_by_footprint_mb_seq_rand": d.get("hit_rate_by_footprint_mb_seq_rand")})
    return f, s


# ---------------------------------------------------------------- B14 L1
def fit_b14(recs):
    f, s = {}, {}
    by = keyed(recs, "l1_bytes_by_carveout")
    if by:
        f["l1_bytes_by_carveout"] = {k.split("=")[-1]: int(r["value"]) for k, r in sorted(by.items(), key=lambda kv: int(kv[0].split("=")[-1]))}
        s["l1_bytes_by_carveout"] = {"from": [r["quantity"] for r in by.values()], "method": "B14.1: single-thread chase at 128 B stride; capacity = largest footprint under 1.5x the 4 KiB latency, per dynamic smem request"}
    lat = recs.get("l1_latency_ns")
    if lat:
        d = lat["detail"]
        f["l1_latency_ns"] = lat["value"]
        lines = d.get("line_cycles_by_stride_16KiB") or []
        # line size: the smallest stride at which the cycles per step stop falling as the stride grows (every step a new line)
        f["l1_line_bytes"] = None
        if lines and len(lines) >= 2:
            # a flat tail from stride s on means every step touches a new line from s; a monotone fall to the last stride resolves nothing
            flat = [st for st, cyc in lines if abs(cyc - lines[-1][1]) <= 0.1 * lines[-1][1]]
            if flat and flat[0] != lines[-1][0]:
                f["l1_line_bytes"] = int(flat[0])
        s["l1_latency_ns"] = _nm(lat, "l1_latency_ns", "B14.1: 4 KiB chase hit latency with no shared memory requested", inferred_from=d.get("table"))
        s["l1_line_bytes"] = {"from": "l1_latency_ns", "method": "B14.2: chase at strides 32..256 over 16 KiB; the stride from which the per-step cost is flat (every step touches a new line); null when the cost still falls at the largest stride (unresolved by this probe)", "inferred_from": lines}
    for q, k in (("cp_async_cg_allocates_l1", "cp_async_cg_allocates_l1"), ("cp_async_ca_allocates_l1", "cp_async_ca_allocates_l1"), ("tma_allocates_l1", "tma_allocates_l1")):
        r = recs.get(q)
        if r:
            f[k] = bool(r["value"]); s[k] = _nm(r, q, "B14.3: 32 KB copied global->smem, then the same lines chased with ld.global.ca; L1-hit latency means the copy allocated", chase_cycles_after_copy=r["detail"].get("chase_cycles_after_copy"), l1_hit_cycles=r["detail"].get("l1_hit_cycles"))
    pool = recs.get("l1_smem_pool_bytes")
    if pool:
        f["l1_smem_pool_bytes"] = int(pool["value"]); s["l1_smem_pool_bytes"] = _nm(pool, "l1_smem_pool_bytes", "B14.4: max over settings of measured L1 capacity + requested smem", inferred_from=pool["detail"].get("l1_plus_smem_by_setting"))
    return f, s


# ---------------------------------------------------------------- B15 copy queue
def fit_b15(recs):
    f, s = {}, {}
    g = recs.get("cp_async_groups_in_flight")
    if g:
        f["cp_async_groups_in_flight"] = {"per_warp": int(g["value"]) if g["value"] and g["value"] > 0 else None, "note": "per warp = per block here (one block of 256 threads per SM, each group 4 KB); -1/null = no issue stall within 22 pending groups"}
        s["cp_async_groups_in_flight"] = _nm(g, "cp_async_groups_in_flight", "B15.1: clock64 cost to issue the k-th 4 KB group with k-1 pending; the first k whose increment exceeds 3x the median is the window", inferred_from=g["detail"].get("issue_cycles_by_k"))
    ic = recs.get("cp_async_issue_cycles_per_16B")
    if ic:
        f["cp_async_issue_cycles_per_16B"] = ic["value"]; s["cp_async_issue_cycles_per_16B"] = _nm(ic, "cp_async_issue_cycles_per_16B", "B15.3: cycles per cp.async 16 B instruction per warp, issue only (no wait)")
    ld = keyed(recs, "cp_async_latency_dist")
    if ld:
        f["cp_async_latency_dist"] = {k: {kk: r["detail"].get(kk) for kk in ("p1_cycles", "p50_cycles", "p99_cycles", "max_cycles", "p1_s", "p50_s", "p99_s", "max_s") if kk in r["detail"]} for k, r in sorted(ld.items())}
        s["cp_async_latency_dist"] = {"from": [r["quantity"] for r in ld.values()], "method": "B15.2: one tile per trial, 2000 trials x 48 SMs (x 6 blocks when resident=6), issue-to-wait_group-return in clock64 cycles"}
    th = keyed(recs, "copy_throughput")
    if th:
        cp, tma, mix = _val(th.get("cp_async")), _val(th.get("tma_bulk")), _val(th.get("mixed_half_half"))
        f["copy_throughput_bytes_per_s"] = {k: _val(r) for k, r in th.items()}
        if cp and tma and mix:
            # a shared path: the mixed run is bounded by the same aggregate; separate paths: the mix exceeds the smaller single-path rate by a clear margin
            f["tma_cp_async_shared_path"] = bool(mix <= 1.15 * max(cp, tma))
            s["tma_cp_async_shared_path"] = {"from": [r["quantity"] for r in th.values()], "method": "B15.4: half the SMs stream with cp.async and half with TMA; shared path when the mix does not exceed the better single-path aggregate by more than 15 %", "inferred_from": {"cp_async": cp, "tma_bulk": tma, "mixed": mix}}
        s["copy_throughput_bytes_per_s"] = {"from": [r["quantity"] for r in th.values()], "method": "B15.4: 48 blocks streaming 16 KB tiles, aggregate bytes per second", "ci95": {k: r.get("ci95") for k, r in th.items()}}
    mb = recs.get("mbarrier_s")
    if mb:
        f["mbarrier_s"] = mb["value"]; s["mbarrier_s"] = _nm(mb, "mbarrier_s", "B15.5: mbarrier arrive + try_wait round with nothing pending, 256 threads")
    t2 = recs.get("tma_2d_latency_s")
    if t2:
        nm = _not_measured(recs, "tma_2d_latency_s")
        f["tma_2d_latency_s"] = nm or t2["value"]; s["tma_2d_latency_s"] = _nm(t2, "tma_2d_latency_s", "B15.6: cuTensorMapEncodeTiled 128x32 fp32 tile, cp.async.bulk.tensor.2d + mbarrier, median SM")
    return f, s


# ---------------------------------------------------------------- B16 block scheduler
def fit_b16(recs):
    f, s = {}, {}
    tabs = keyed(recs, "block_order_table")
    dist = keyed(recs, "block_distinct_sms")
    if tabs:
        order = {}
        for k, r in sorted(tabs.items()):
            d = r["detail"]; sm = d.get("sm_by_block") or []; t = d.get("entry_ns_by_block") or []
            sms_n = len(set(sm)); rr = all(sm[i] != sm[i - 1] for i in range(1, min(len(sm), sms_n))) if sm else False
            cls = "round-robin-first-wave" if rr and sms_n >= 2 else "unclassified"
            if sm and sms_n >= 2:
                # even/odd or strided patterns: compare the first-wave SM sequence to simple generators
                first = sm[:sms_n]
                if first == sorted(first): cls = "ascending-sm-id"
                elif first == list(range(0, sms_n, 2)) + list(range(1, sms_n, 2)): cls = "even-then-odd"
                elif all((first[i] - first[i - 1]) % sms_n == (first[1] - first[0]) % sms_n for i in range(1, len(first))): cls = f"stride-{(first[1] - first[0]) % sms_n}"
                elif len(first) >= 4 and all(first[i + 1] == first[i] + 1 for i in range(0, len(first) - 1, 2)):
                    # pairs (2i, 2i+1) in sequence: the pair stride tells how the groups are visited
                    pstride = first[2] - first[0]
                    groups = sms_n // pstride if pstride and sms_n % pstride == 0 else None
                    cls = f"sm-pairs round-robin over {groups} groups of {pstride} SMs" if groups else f"sm-pairs with pair stride {pstride}"
            order[k] = {"class": cls, "identical_orderings_fraction": dist.get(k, {}).get("detail", {}).get("identical_orderings_fraction") if dist else None,
                        "sms_with_two_blocks_median": dist.get(k, {}).get("detail", {}).get("sms_with_two_blocks_median") if dist else None}
            if k in ("grid=48,threads=256,smem=0", "grid=96,threads=256,smem=61440", "grid=192,threads=128,smem=0"):   # representative tables; the rest stay in raw/
                order[k]["sm_sequence"] = sm[:96]; order[k]["entry_ns"] = [x - t[0] for x in t[:96]] if t else []
        f["block_dispatch_order"] = order
        s["block_dispatch_order"] = {"from": [r["quantity"] for r in tabs.values()] + [r["quantity"] for r in dist.values()], "method": "B16.1: %smid and globaltimer at entry of every block, sorted by entry time; class from the first-wave SM sequence; the full table is in the raw record"}
        det = [v["identical_orderings_fraction"] for v in order.values() if v["identical_orderings_fraction"] is not None]
        if det:
            f["block_dispatch_deterministic"] = min(det); s["block_dispatch_deterministic"] = {"from": [r["quantity"] for r in dist.values()], "method": "B16.1: fraction of repeated launches whose block->SM assignment is identical to the first (min over configurations)"}
    r = recs.get("block_dispatch_rate_per_s")
    if r:
        f["block_dispatch_rate_per_s"] = r["value"]; s["block_dispatch_rate_per_s"] = _nm(r, "block_dispatch_rate_per_s", "B16.2: 48 x 32 empty-ish blocks, entry timestamps; blocks per second over the dispatch span")
    r = recs.get("block_refill_latency_s")
    if r:
        f["block_refill_latency_s"] = r["value"]; s["block_refill_latency_s"] = _nm(r, "block_refill_latency_s", "B16.3: exit of a first-wave block to entry of the block that replaces it on the same SM, median")
    st = keyed(recs, "wave_staircase")
    if st:
        f["wave_staircase"] = {k.split("=")[-1]: r["value"] for k, r in sorted(st.items(), key=lambda kv: float(kv[0].split("=")[-1]))}
        s["wave_staircase"] = {"from": [r["quantity"] for r in st.values()], "method": "B16.4: kernel time vs grid size in units of one wave (r x 48 blocks); a staircase means waves are serialized", "ci95": {k.split("=")[-1]: r.get("ci95") for k, r in st.items()}}
    return f, s


# ---------------------------------------------------------------- B17 latency hiding
def fit_b17(recs, contract):
    f, s = {}, {}
    bw = keyed(recs, "bandwidth_vs_warps")
    if bw:
        curve = {}
        for k, r in bw.items():
            lvl, w = k.split(",")[0], int(k.split("=")[-1]); curve.setdefault(lvl, {})[w] = r["value"]
        f["bandwidth_vs_warps"] = {lvl: {str(w): v for w, v in sorted(c.items())} for lvl, c in curve.items()}
        s["bandwidth_vs_warps"] = {"from": [r["quantity"] for r in bw.values()], "method": "B17.1: streaming read bandwidth with w resident warps per SM (dram: 1 GiB, l2: 8 MiB)"}
    for lvl in ("dram", "l2"):
        r = recs.get(f"warps_to_hide_latency[{lvl}]")
        if r:
            f[f"warps_to_hide_{lvl}_latency"] = r["value"]; s[f"warps_to_hide_{lvl}_latency"] = _nm(r, f"warps_to_hide_latency[{lvl}]", "B17.1: smallest warp count reaching 90 % of the plateau bandwidth")
    r = recs.get("per_warp_outstanding_loads")
    if r:
        f["per_warp_outstanding_loads"] = r["value"]; s["per_warp_outstanding_loads"] = _nm(r, "per_warp_outstanding_loads", "B17.2: independent loads per warp before the issue time per load rises (the per-warp MLP limit)", inferred_from=r["detail"])
    # per-SM outstanding requests by Little's law: bandwidth per SM x latency / bytes per request (128 B lines)
    lat = contract.get("latency_ns") or {}
    for lvl, lkey in (("dram", "dram"), ("l2", "l2")):
        c = f.get("bandwidth_vs_warps", {}).get(lvl)
        if c and lat.get(lkey) and contract.get("sms"):
            peak = max(c.values()); per_sm = peak / contract["sms"]
            f.setdefault("per_sm_outstanding_requests", {})[lvl] = round(per_sm * lat[lkey] * 1e-9 / 128, 1)
            s["per_sm_outstanding_requests"] = {"from": [r["quantity"] for r in bw.values()], "method": "B17.3 (derived): Little's law, peak bandwidth per SM x unloaded latency (B5) / 128 B per request; the loaded latency is higher, so this is a lower bound on the queue depth"}
    r = recs.get("issue_width_per_sm")
    if r:
        import math
        f["issue_rate_warp_instr_per_cycle_per_sm"] = r["value"]
        f["schedulers_per_sm"] = int(math.ceil(r["value"] - 0.05)) if r["value"] else None
        f["issue_width_per_scheduler"] = 1
        s["issue_rate_warp_instr_per_cycle_per_sm"] = _nm(r, "issue_width_per_sm", "B17.4: FFMA warp-instructions issued per SM cycle with w warps of 8 independent chains each, saturating value (block-wide clock64 window)", inferred_from=r["detail"])
        s["schedulers_per_sm"] = {"from": "issue_width_per_sm", "method": "B17.4 (derived): smallest integer at or above the saturating issue rate; the FFMA probe reaches 3.3 of the 4 x 1 the family declares"}
        s["issue_width_per_scheduler"] = {"from": "issue_width_per_sm", "method": "assumed 1 instruction per scheduler per cycle (B17.4 measures the product)"}
    ops = keyed(recs, "op_latency_cycles")
    if ops:
        f["op_latency_cycles"] = {k: r["value"] for k, r in ops.items()}
        s["op_latency_cycles"] = {"from": [r["quantity"] for r in ops.values()], "method": "B17.5: dependent chain of one instruction, clock64 per step, single warp", "ci95": {k: r.get("ci95") for k, r in ops.items()}}
    return f, s


# ---------------------------------------------------------------- B18 DRAM
def fit_b18(recs):
    f, s = {}, {}
    r = recs.get("dram_interleave")
    if r:
        d = r["detail"]
        bits = d.get("inferred_interleave_bits") or []
        f["dram_channels"] = (1 << len(bits)) if bits else None
        f["dram_interleave_bits"] = bits
        tab = d.get("table_forcedBit_bytesPerS") or []; top = d.get("top_bytes_per_s") or 0
        worst = min(((bw / top) if top else 1.0, bit) for bit, bw in tab) if tab else (1.0, None)
        f["dram_channel_select"] = "no single address bit selects a channel (every forced bit keeps >= %d %% of the unmasked bandwidth): the channel function hashes several bits; the count is not inferable from user space by this probe" % round(100 * worst[0]) if not bits else f"bits {bits}"
        s["dram_channels"] = _nm(r, "dram_interleave", "B18.1: random 128 B requests over 1 GiB with one address bit forced to 0; a bit whose forcing drops the bandwidth below 70 % selects a channel; channels = 2^(bits found); null when no bit does", inferred_from={k: v for k, v in d.items() if k != "method"})
        s["dram_interleave_bits"] = {"from": "dram_interleave", "method": "B18.1: the address bits whose forcing drops the random-request bandwidth"}
        s["dram_channel_select"] = {"from": "dram_interleave", "method": "B18.1: reading of the per-bit table (min bandwidth ratio over bits 7..29)"}
    loc = keyed(recs, "dram_locality")
    if loc:
        f["dram_locality_bytes_per_s"] = {k: r["value"] for k, r in loc.items()}
        seq, rnd = _val(loc.get("seq_128B")), _val(loc.get("rand_128B"))
        if seq and rnd:
            f["dram_seq_random_ratio"] = round(seq / rnd, 3)
        s["dram_locality_bytes_per_s"] = {"from": [r["quantity"] for r in loc.values()], "method": "B18.2: read bandwidth for sequential 128 B lines vs random lines of 128 B .. 16 KB runs over 1 GiB", "ci95": {k: r.get("ci95") for k, r in loc.items()}}
        s["dram_seq_random_ratio"] = {"from": ["dram_locality[seq_128B]", "dram_locality[rand_128B]"], "method": "B18.2: sequential / random-128B bandwidth"}
    r = recs.get("dram_page_bytes_effective")
    if r:
        f["dram_page_bytes_effective"] = int(r["value"]) if r["value"] else None; s["dram_page_bytes_effective"] = _nm(r, "dram_page_bytes_effective", "B18.2: smallest random run length reaching 90 % of the sequential bandwidth", inferred_from=r["detail"])
    r = recs.get("dram_rw_turnaround_penalty")
    if r:
        f["dram_rw_turnaround_penalty"] = r["detail"].get("curve") or r["value"]; s["dram_rw_turnaround_penalty"] = _nm(r, "dram_rw_turnaround_penalty", "B18.3: (read-only + write-only time) / mixed time at read:write ratios; > 1 = turnaround cost", inferred_from=r["detail"])
    r = recs.get("dram_bytes_per_s_under_cpu_load")
    if r:
        f["dram_bytes_per_s_under_cpu_load"] = r["value"]; s["dram_bytes_per_s_under_cpu_load"] = _nm(r, "dram_bytes_per_s_under_cpu_load", "B18.4: GPU streaming read while all CPU cores run a memcpy-style stream", inferred_from=r["detail"])
    r = recs.get("dram_drift_60s")
    if r:
        f["dram_drift_60s"] = r["value"]; s["dram_drift_60s"] = _nm(r, "dram_drift_60s", "B18.5: (max - min) / max of per-second read-bandwidth samples over 60 s", inferred_from={k: v for k, v in r["detail"].items() if "series" not in k})
    return f, s


# ---------------------------------------------------------------- B19 registers / local
def fit_b19(recs):
    f, s = {}, {}
    sp = keyed(recs, "spill_flops_per_s_per_sm")
    if sp:
        curve = {}
        for k, r in sp.items():
            cap = int(k.split("=")[-1]); d = r["detail"]
            curve[str(cap)] = {"flops_per_s_per_sm": r["value"], "regs_used": d.get("regs_used"), "spill_bytes": d.get("spill_bytes_ptxas"), "local_bytes": d.get("local_bytes")}
        f["spill_penalty_curve"] = dict(sorted(curve.items(), key=lambda kv: -int(kv[0])))
        s["spill_penalty_curve"] = {"from": [r["quantity"] for r in sp.values()], "method": "B19.1: the same 64-accumulator FMA kernel built at each -maxrregcount; throughput per SM with the ptxas spill bytes", "ci95": {k.split("=")[-1]: r.get("ci95") for k, r in sp.items()}}
    for q in ("local_mem_latency_ns", "local_mem_bytes_per_s_per_sm"):
        r = recs.get(q)
        if r:
            f[q] = r["value"]; s[q] = _nm(r, q, "B19.2: " + ("single-thread dependent chase over a local array" if "latency" in q else "8 blocks per SM streaming a 256 B local array, bytes per second per SM"))
    occ = keyed(recs, "max_blocks_per_sm")
    regs = sorted(((int(k.split("=")[-1]), int(r["value"])) for k, r in occ.items() if k.startswith("regs=") and r.get("value")), key=lambda t: t[0])
    if regs:
        # allocation granularity: the smallest register step at which the resident-block count changes implies a per-warp rounding; report the per-thread step between distinct occupancy levels
        steps = [b - a for (a, _), (b, _) in zip(regs, regs[1:])]
        f["reg_alloc_granularity"] = {"regs_per_thread_steps_tested": [r for r, _ in regs], "blocks_per_sm": {str(r): b for r, b in regs}, "note": "from B9's __launch_bounds__ variants; the hardware rounds per warp to 256-register units (derived, not independently measured)"}
        s["reg_alloc_granularity"] = {"from": [r["quantity"] for k, r in occ.items() if k.startswith("regs=")], "method": "B19.4 (derived from B9): resident blocks per SM at each register budget"}
    nm = _not_measured(recs, "reg_bank_conflict_penalty")
    if nm:
        f["reg_bank_conflict_penalty"] = nm; s["reg_bank_conflict_penalty"] = {"from": "reg_bank_conflict_penalty", "method": "B19.3: not measured"}
    return f, s


# ---------------------------------------------------------------- B20 tensor detail
def fit_b20(recs):
    f, s = {}, {}
    rate = keyed(recs, "tensor_rate")
    if rate:
        per_sm = {}; supported = []; unsupported = {}
        for k, r in rate.items():
            if r.get("samples"):
                per_sm[k] = r["value"]; supported.append(k)
            else:
                unsupported[k] = r["detail"].get("reason")
        f["tensor_flops_per_s_per_sm_by_shape"] = per_sm
        f["tensor_shapes_supported"] = sorted(supported)
        if unsupported:
            f["tensor_shapes_unsupported"] = unsupported
        s["tensor_flops_per_s_per_sm_by_shape"] = {"from": [r["quantity"] for r in rate.values()], "method": "B20.2: mma.sync per shape, 4 chains x 4 warps x 4 blocks per SM, clock64 cycles", "ci95": {k: r.get("ci95") for k, r in rate.items()}}
        s["tensor_shapes_supported"] = {"from": [r["quantity"] for r in rate.values()], "method": "B20.2: shapes that compiled and ran on sm_121"}
    lat = keyed(recs, "tensor_op_latency_cycles")
    if lat:
        f["tensor_op_latency_cycles"] = {k: r["value"] for k, r in lat.items()}; s["tensor_op_latency_cycles"] = {"from": [r["quantity"] for r in lat.values()], "method": "B20.3: dependent mma.sync chain, accumulator feeds the next, single warp"}
    ldm = keyed(recs, "ldmatrix_bytes_per_s_per_sm")
    if ldm:
        f["ldmatrix_bytes_per_s_per_sm"] = {k: r["value"] for k, r in ldm.items()}; s["ldmatrix_bytes_per_s_per_sm"] = {"from": [r["quantity"] for r in ldm.values()], "method": "B20.4: ldmatrix.x4 from a 16 KB smem tile, swizzled vs row-major (128 B rows conflict)", "ci95": {k: r.get("ci95") for k, r in ldm.items()}}
    chip = keyed(recs, "tensor_flops_per_s_chip")
    fp8 = {k: (r["value"] if r.get("samples") else {"measured": False, "reason": r["detail"].get("reason")}) for k, r in chip.items() if k.startswith("fp8")}
    if fp8:
        f["tensor_flops_per_s_chip_fp8"] = fp8; s["tensor_flops_per_s_chip_fp8"] = {"from": [r["quantity"] for k, r in chip.items() if k.startswith("fp8")], "method": "B20.1: cublasLtMatmul 8192^2, FP8 inputs, FP32 accumulate"}
    return f, s


# ---------------------------------------------------------------- B21 compute block
def fit_b21(recs):
    f, s = {}, {}
    it = keyed(recs, "inst_throughput_per_sm")
    if it:
        f["inst_throughput_per_sm"] = {k: r["value"] for k, r in it.items() if not k.startswith("lds") and not k.startswith("sts")}
        f["ld_shared_bytes_per_s_per_sm"] = {k: r["value"] * {"lds_32": 4, "lds_64": 8, "lds_128": 16, "sts_32": 4}[k] for k, r in it.items() if k in ("lds_32", "lds_64", "lds_128", "sts_32")}
        f["ld_shared_instr_per_s_per_sm"] = {k: r["value"] / 32 for k, r in it.items() if k.startswith(("lds", "sts"))}   # warp instructions per second per SM
        s["ld_shared_instr_per_s_per_sm"] = {"from": [r["quantity"] for k, r in it.items() if k.startswith(("lds", "sts"))], "method": "B21.1: ld.shared / st.shared warp-instructions per second per SM by width; lds_128_broadcast2 = 16 B loads with two distinct addresses per warp (the template's A-load pattern)"}
        s["inst_throughput_per_sm"] = {"from": [r["quantity"] for r in it.values()], "method": "B21.1: B2 pattern per opcode class (ILP 8, 8 x 256 threads per SM), instructions per second per SM", "ci95": {k: r.get("ci95") for k, r in it.items()}}
        s["ld_shared_bytes_per_s_per_sm"] = {"from": [r["quantity"] for k, r in it.items() if k.startswith(("lds", "sts"))], "method": "B21.1: ld.shared / st.shared throughput by access width, bytes per second per SM"}
    eff = keyed(recs, "compute_block_efficiency")
    if eff:
        f["compute_block_efficiency"] = {k: {kk: r["detail"].get(kk) for kk in ("TM", "TN", "TK", "R", "cycles_per_phase", "fma_bound_cycles", "efficiency", "lds_per_thread_per_phase", "ffma_per_thread_per_phase", "sass_mix")} for k, r in eff.items()}
        s["compute_block_efficiency"] = {"from": [r["quantity"] for r in eff.values()], "method": "B21.2-3: the template's inner loop with the smem tile pre-filled, one block per SM; cycles per TK step over the FFMA-only bound; SASS opcode mix from cuobjdump"}
    return f, s


# ---------------------------------------------------------------- B22 clocks
def fit_b22(recs):
    f, s = {}, {}
    for q in ("clock_warmup_s", "clock_idle_hz"):
        r = recs.get(q)
        if r:
            f[q] = r["value"]; s[q] = _nm(r, q, {"clock_warmup_s": "B22.1: time from an idle start until the in-kernel clock reaches 98 % of its sustained value", "clock_idle_hz": "B22.1: SM clock reported by nvidia-smi at idle"}[q], inferred_from={k: v for k, v in r["detail"].items() if "series" not in k})
    by = keyed(recs, "clock_hz_sustained_by_load")
    if by:
        f["clock_hz_sustained_by_load"] = {k: r["value"] for k, r in by.items()}
        def limit(d):
            cap = str(d.get("sw_power_cap_and_thermal_last") or "")
            pw = [x for x in (d.get("power_w_series") or []) if isinstance(x, (int, float))]; tc = [x for x in (d.get("temperature_c_series") or []) if isinstance(x, (int, float))]
            parts = [x.strip() for x in cap.split("/")]
            power = bool(parts) and parts[0] == "Active"; thermal = len(parts) > 1 and parts[1] == "Active"
            if power: return "power"
            if thermal: return "thermal"
            return "none" if pw else None
        caps = {k: limit(r["detail"]) for k, r in by.items()}
        f["clock_governor_limit"] = caps
        s["clock_hz_sustained_by_load"] = {"from": [r["quantity"] for r in by.values()], "method": "B22.2: median SM clock over 30 s of each load type (nvidia-smi once per second, with power and temperature)", "inferred_from": {k: {"smi_clock_mhz_min_median_max": _mmm(r["detail"].get("smi_clock_mhz_series")), "power_w_min_median_max": _mmm(r["detail"].get("power_w_series")), "temperature_c_min_median_max": _mmm(r["detail"].get("temperature_c_series")), "sw_power_cap_and_thermal_last": r["detail"].get("sw_power_cap_and_thermal_last"), "probe_mhz_after": r["detail"].get("probe_mhz_after"), "series": "in raw/"} for k, r in by.items()}}
        s["clock_governor_limit"] = {"from": [r["quantity"] for r in by.values()], "method": "B22.3: nvidia-smi throttle reasons / power vs cap during each load"}
    g = keyed(recs, "clock_after_gap_hz")
    r = recs.get("clock_after_gap_hz")
    if g or r:
        f["clock_after_gap_hz"] = {k: rr["value"] for k, rr in g.items()} if g else ({str(gap): hz for gap, hz in r["detail"].get("gap_ms_vs_clock_hz", [])} or r["value"])
        s["clock_after_gap_hz"] = {"from": [rr["quantity"] for rr in g.values()] if g else "clock_after_gap_hz", "method": "B22.4: in-kernel clock of the first kernel after an idle gap of the given length"}
    return f, s


# ---------------------------------------------------------------- B23 unified memory
def fit_b23(recs):
    f, s = {}, {}
    ft = keyed(recs, "um_first_touch_s")
    if ft:
        f["um_first_touch_s"] = {k.split("=")[-1]: r["value"] for k, r in ft.items()}
        s["um_first_touch_s"] = {"from": [r["quantity"] for r in ft.values()], "method": "B23.1: managed 1 GiB touched by the CPU one byte per page, then by a GPU kernel one byte per page; per-touch cost on first access vs the second pass", "inferred_from": {k.split("=")[-1]: r["detail"].get("steady_touch_s_per_page_median") for k, r in ft.items()}}
    for q in ("um_prefetched_bytes_per_s", "um_host_resident_bytes_per_s", "um_translation_overhead", "um_max_alloc_bytes"):
        r = recs.get(q)
        if r:
            f[q] = r["value"]; s[q] = _nm(r, q, "B23: " + r["detail"].get("method", ""), inferred_from={k: v for k, v in r["detail"].items() if k != "method"})
    return f, s


# ---------------------------------------------------------------- B24 atomics
def fit_b24(recs):
    f, s = {}, {}
    th = keyed(recs, "atomic_throughput_per_s")
    if th:
        f["atomic_throughput_per_s"] = {k: r["value"] for k, r in th.items() if "contention" not in k}
        f["atomic_contention_curve"] = {k.split("=")[-1]: r["value"] for k, r in th.items() if "contention" in k}
        s["atomic_throughput_per_s"] = {"from": [r["quantity"] for r in th.values()], "method": "B24.1-2: atomicAdd f32 by scope (global L2 / shared) and pattern (same address, per warp, per thread), atom (value used) vs red (value unused); atomics per second over the slowest block", "ci95": {k: r.get("ci95") for k, r in th.items()}}
        s["atomic_contention_curve"] = {"from": [r["quantity"] for k, r in th.items() if "contention" in k], "method": "B24.3: same-address global atomics vs number of blocks"}
    lat = keyed(recs, "atomic_latency_ns")
    if lat:
        f["atomic_latency_ns"] = {k: r["value"] for k, r in lat.items()}; s["atomic_latency_ns"] = {"from": [r["quantity"] for r in lat.values()], "method": "B24.4: single-thread dependent atomicAdd chain"}
    return f, s


# ---------------------------------------------------------------- B25 multi-kernel
def fit_b25(recs):
    f, s = {}, {}
    for q in ("kernel_gap_s", "graph_kernel_gap_s"):
        r = recs.get(q)
        if r:
            f[q] = r["value"]; s[q] = _nm(r, q, "B25: " + r["detail"].get("method", ""))
    ck = keyed(recs, "concurrent_kernels")
    if ck:
        f["concurrent_kernels"] = {k.split("=")[-1]: {"concurrent": bool(r["detail"].get("concurrent")), "ratio_two_streams_over_one": r["value"]} for k, r in ck.items()}
        s["concurrent_kernels"] = {"from": [r["quantity"] for r in ck.values()], "method": "B25.3: two 2 ms kernels on two streams; ratio ~1 = co-run, ~2 = serialized, by blocks per kernel"}
    for q in ("l2_persists_across_kernels", "l2_persisting_window_supported"):
        r = recs.get(q)
        if r:
            f[q] = bool(r["value"]); s[q] = _nm(r, q, "B25.4: " + r["detail"].get("method", ""), inferred_from={k: v for k, v in r["detail"].items() if k != "method"})
    return f, s


# ---------------------------------------------------------------- B26 functional
def fit_b26(recs):
    f, s = {}, {}
    r = recs.get("cp_async_window_cycles")
    if r:
        d = r["detail"]
        iof = d.get("in_order_fraction_median", d.get("in_order_fraction", 0)) or 0
        f["cp_async_chunk_order"] = "in-order" if iof >= 0.999 else ("mostly-in-order" if iof >= 0.9 else "out-of-order")
        f["cp_async_window_cycles"] = {"p1": d.get("window_cycles_p1"), "p50": d.get("window_cycles_p50"), "p99": d.get("window_cycles_p99"), "first_chunk_p50": d.get("first_chunk_cycles_p50"),
                                       "note": "upper bound: one polling warp sweeps all chunks in order, so the window includes the sweep time"}
        f["cp_async_torn_chunks_observed"] = d.get("torn_chunks_observed")
        s["cp_async_window_cycles"] = _nm(r, "cp_async_window_cycles", "B26.1: cycles from issue to the moment every 16 B chunk of a group is visible to a polling warp; chunk order from the landing order; torn = a chunk seen half-written", inferred_from=d)
        s["cp_async_chunk_order"] = {"from": "cp_async_window_cycles", "method": "B26.1: fraction of groups whose chunks landed in issue order"}
        s["cp_async_torn_chunks_observed"] = {"from": "cp_async_window_cycles", "method": "B26.1: count over all polled chunks"}
    r = recs.get("wait_group_overcomplete_observed")
    if r:
        f["wait_group_overcomplete_observed"] = int(r["value"]); s["wait_group_overcomplete_observed"] = _nm(r, "wait_group_overcomplete_observed", "B26.2: wait_group N returned while more than N groups were still incomplete (count)", inferred_from=r["detail"])
    r = recs.get("threadfence_visibility_cycles")
    if r:
        f["threadfence_visibility_cycles"] = r["detail"]; s["threadfence_visibility_cycles"] = _nm(r, "threadfence_visibility_cycles", "B26.3: a writer warp's smem store seen by a reader after syncthreads / named barrier / fence only; counts of stale reads and the visibility delay")
    r = recs.get("tma_early_bytes_observed")
    if r:
        f["tma_early_bytes_observed"] = int(r["value"]); s["tma_early_bytes_observed"] = _nm(r, "tma_early_bytes_observed", "B26.4: bytes of a TMA tile visible before the mbarrier phase completed (count)", inferred_from=r["detail"])
    for q in ("fma_contraction_default", "ftz_default"):
        r = recs.get(q)
        if r:
            f[q] = bool(r["value"]); s[q] = _nm(r, q, "B26.5: " + r["detail"].get("method", ""), inferred_from={k: v for k, v in r["detail"].items() if k != "method"})
    ulp = keyed(recs, "tensor_accumulate_ulp_bound")
    r = recs.get("tensor_accumulate_ulp_bound")
    if ulp or r:
        f["tensor_accumulate_ulp_bound"] = {k: rr["value"] for k, rr in ulp.items()} if ulp else r["detail"]
        s["tensor_accumulate_ulp_bound"] = {"from": [rr["quantity"] for rr in ulp.values()] if ulp else "tensor_accumulate_ulp_bound", "method": "B26.5: max |mma - fp64 reference| in ulps of the fp32 accumulator over random inputs, per precision"}
    r = recs.get("smem_bank_map")
    if r:
        d = r["detail"]; t8 = dict((int(a), b) for a, b in (d.get("cycles_per_access_by_stride_elems_8B") or [])); t16 = dict((int(a), b) for a, b in (d.get("cycles_per_access_by_stride_elems_16B") or []))
        cls = d.get("classification") or d.get("bank_map")
        if t8 and t16 and 1 in t8 and 16 in t8 and 32 in t8 and 8 in t16 and 16 in t16:
            sat8 = abs(t8[32] - t8[16]) <= 0.1 * t8[16]; sat16 = abs(t16[16] - t16[8]) <= 0.1 * t16[8]
            if sat8 and sat16 and t8[2] < 1.6 * t8[1]:
                cls = "32 banks x 4 B; 64-bit accesses served per half-warp (max 16-way), 128-bit per quarter-warp (max 8-way)"
        f["smem_bank_map"] = cls; s["smem_bank_map"] = _nm(r, "smem_bank_map", "B26.6: conflict cycles at 8 B and 16 B strides; saturation at stride 16 (8 B) and 8 (16 B) with no penalty at stride 1 is the 32 x 4 B map with half/quarter-warp wavefronts", inferred_from=d)
    r = recs.get("independent_thread_scheduling")
    if r:
        f["independent_thread_scheduling"] = bool(r["value"]); s["independent_thread_scheduling"] = _nm(r, "independent_thread_scheduling", "B26.7: divergent branches of one warp interleave in time (even lanes start before odd lanes finish)", inferred_from=r["detail"])
    return f, s


# ---------------------------------------------------------------- BX counter cross-checks (ncu, after P0)
def fit_counters(recs):
    """ncu records -> contract['counter_crosscheck']: the same questions as B13/B14/B17/B21/B26 answered from hardware counters."""
    cc, s = {}, {}
    rows = {}
    for q, r in recs.items():
        m = re.fullmatch(r"counter_(\w+)\[a=(-?\d+),b=(-?\d+)\]", q)
        if m and r.get("detail", {}).get("measured"):
            rows.setdefault(m.group(1), {})[(int(m.group(2)), int(m.group(3)))] = {k: v[0] for k, v in r["detail"]["metrics"].items()}
    def g(mt, *names):
        for n in names:
            if n in mt: return mt[n]
        return None
    fill = rows.get("l2_fill")
    if fill:
        tab = {}
        for (st, _), mt in sorted(fill.items()):
            req = g(mt, "lts__t_requests_srcunit_tex.sum"); sec = g(mt, "lts__t_sectors_srcunit_tex_op_read.sum", "lts__t_sectors_srcunit_tex.sum"); miss = g(mt, "lts__t_sectors_op_read_lookup_miss.sum")
            tab[str(st)] = {"l2_sectors_per_request": round(sec / req, 3) if req and sec else None, "dram_bytes_per_request": round(32 * miss / req, 1) if req and miss else None, "hit_rate_pct": g(mt, "lts__t_sector_hit_rate.pct")}
        big = [v["dram_bytes_per_request"] for st, v in tab.items() if int(st) >= 128 and v["dram_bytes_per_request"]]
        cc["l2_fill_by_stride"] = tab
        cc["l2_fill_bytes_per_request_at_128B_plus"] = round(sum(big) / len(big), 1) if big else None
        s["counter_crosscheck.l2_fill"] = {"from": [f"counter_l2_fill[a={a},b={b}]" for (a, b) in fill], "method": "ncu: L2 sectors (32 B) per L2 request and L2-miss sectors per request for a strided 4 B read over 1 GiB; at strides >= 128 B every request is its own line, so miss bytes per request is the DRAM fill granularity"}
    hit = rows.get("l2_hit_rate")
    if hit:
        curve = {str(a): g(mt, "lts__t_sector_hit_rate.pct") for (a, b), mt in sorted(hit.items())}
        cc["l2_hit_rate_pct_by_footprint_mib"] = curve
        # 8 passes, the first fills: a fully resident footprint reads 7/8 = 87.5 % hits; capacity = the last footprint within 3 points of that plateau
        cap = max((int(k) for k, v in curve.items() if v is not None and v >= 87.5 - 3), default=None)
        cc["l2_capacity_mib_from_counters"] = cap
        s["counter_crosscheck.l2_hit_rate"] = {"from": [f"counter_l2_hit_rate[a={a},b={b}]" for (a, b) in hit], "method": "ncu: L2 sector hit rate of an 8-pass per-block cyclic stream, L1 bypassed (first pass fills, 87.5 % = fully resident); capacity = largest footprint within 3 points of that plateau"}
    iss = rows.get("issue")
    if iss:
        tab = {}
        for (w, _), mt in sorted(iss.items()):
            cyc = g(mt, "sm__cycles_active.avg"); issued = g(mt, "sm__inst_issued.sum")
            tab[str(w)] = {"inst_issued_per_active_cycle_per_sm": round(issued / 48 / cyc, 3) if cyc and issued else None, "issue_active_fraction": round(g(mt, "sm__issue_active.avg") / cyc, 3) if cyc and g(mt, "sm__issue_active.avg") is not None else None, "fma_pipe_active_fraction": round(g(mt, "sm__pipe_fma_cycles_active.avg") / cyc / 8, 3) if cyc and g(mt, "sm__pipe_fma_cycles_active.avg") is not None else None}
        cc["issue_by_warps"] = tab
        s["counter_crosscheck.issue"] = {"from": [f"counter_issue[a={a},b={b}]" for (a, b) in iss], "method": "ncu: sm__inst_issued per active SM cycle (48 SMs), the issue-active fraction and the FP32 pipe activity (sm__pipe_fma_cycles_active over active cycles x 8: four sub-partitions x two FP32 pipes) for the B17 FFMA kernel at w warps per SM; compare issue_rate_warp_instr_per_cycle_per_sm"}
    for key, nm in (("inner_loop", "T1_128x256x16"), ("inner_loop_T5", "T5_128x128x32"), ("inner_loop_R2", "R2_32x32x32")):
        il = rows.get(key)
        if il:
            mt = next(iter(il.values())); cyc = g(mt, "sm__cycles_active.avg"); issued = g(mt, "sm__inst_issued.sum")
            stalls = {k.split("stalled_")[1].split("_per")[0]: round(v, 3) for k, v in mt.items() if "issue_stalled" in k}
            fma = g(mt, "sm__inst_executed_pipe_fma.sum"); lsu = g(mt, "sm__inst_executed_pipe_lsu.sum"); alu = g(mt, "sm__inst_executed_pipe_alu.sum")
            tot = (fma or 0) + (lsu or 0) + (alu or 0)
            cc.setdefault("inner_loop", {})[nm] = {"inst_issued_per_active_cycle_per_sm": round(issued / 48 / cyc, 3) if cyc and issued else None, "issue_active_fraction": round(g(mt, "sm__issue_active.avg") / cyc, 3) if cyc and g(mt, "sm__issue_active.avg") is not None else None, "fma_pipe_active_fraction": round(g(mt, "sm__pipe_fma_cycles_active.avg") / cyc / 8, 3) if cyc and g(mt, "sm__pipe_fma_cycles_active.avg") is not None else None,
                                                   "pipe_share": {"fma": round(fma / tot, 3) if tot and fma else None, "lsu": round(lsu / tot, 3) if tot and lsu else None, "alu": round(alu / tot, 3) if tot and alu else None}, "warps_stalled_per_issue_active": stalls}
            s["counter_crosscheck.inner_loop"] = {"from": [f"counter_{k}[a=0,b=0]" for k in ("inner_loop", "inner_loop_T5", "inner_loop_R2") if k in rows], "method": "ncu: the B21 inner loops, one block per SM, tile pre-filled: issue rate, FMA-pipe activity, instructions by pipe and the average warps stalled per issue-active cycle by reason (what keeps the FMA pipe below 100 %)"}
    sb = rows.get("smem_banks")
    if sb:
        tab = {}
        for (st, _), mt in sorted(sb.items()):
            wf = g(mt, "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"); ins = g(mt, "sm__sass_inst_executed_op_shared_ld.sum"); conf = g(mt, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum")
            tab[str(st)] = {"wavefronts_per_instruction": round(wf / ins, 3) if wf and ins else None, "bank_conflicts_per_instruction": round(conf / ins, 3) if ins and conf is not None else None}
        cc["smem_wavefronts_by_stride"] = tab
        s["counter_crosscheck.smem_banks"] = {"from": [f"counter_smem_banks[a={a},b={b}]" for (a, b) in sb], "method": "ncu: shared-memory wavefronts and bank conflicts per 4 B ld.shared warp instruction at word stride s; compare bank_conflict_penalty (B6) and functional.smem_bank_map (B26)"}
    # B29: the generated kernels at the bad corners (stripped of global loads, and real), cycles per k-step per block and the stall breakdown
    gt = {}
    for q, r in recs.items():
        m = re.fullmatch(r"inner_loop_ground_truth\[kernel=([^,]+),variant=(\w+),b=(\d+)\]", q)
        if m and r.get("detail", {}).get("measured"):
            d = r["detail"]; n = d["ncu"]
            gt.setdefault(m.group(1), {}).setdefault(m.group(2), {})[m.group(3)] = {"cycles_per_step_per_block": r["value"], "issue_active_fraction": n.get("issue_active_fraction"), "inst_issued_per_active_cycle_per_sm": n.get("inst_issued_per_active_cycle_per_sm"),
                                                                                    "fma_pipe_active_fraction": n.get("fma_pipe_active_fraction"), "lsu_wavefronts_per_active_cycle_per_sm": n.get("lsu_wavefronts_per_active_cycle_per_sm"), "dominant_stall": n.get("dominant_stall"), "stalls_per_issue_active": n.get("stalls_per_issue_active")}
    if gt:
        cc["inner_loop_ground_truth"] = gt
        s["counter_crosscheck.inner_loop_ground_truth"] = {"from": [q for q in recs if q.startswith("inner_loop_ground_truth[")], "method": "B29: codegen kernels of the four worst G4 designs and the two spark-001 references, stripped (global loads removed) and real, one wave of 48 x b blocks (b beyond the register-limited residency runs serialized), cycles per k-step per block from the wall time at the in-run clock, ncu issue and stall breakdown"}
    l1 = rows.get("l1_hit")
    if l1:
        cc["l1_hit_rate_pct"] = {f"footprint_kib={a},smem={b}": g(mt, "l1tex__t_sector_hit_rate.pct") for (a, b), mt in sorted(l1.items())}
        caps = {}
        for (a, b), mt in sorted(l1.items()):
            hr = g(mt, "l1tex__t_sector_hit_rate.pct")
            if hr is not None and hr >= 90:
                caps[str(b)] = max(caps.get(str(b), 0), a * 1024)
        cc["l1_capacity_bytes_from_counters_by_smem"] = caps   # largest chase footprint still >= 90 % L1 hits, per smem request
        s["counter_crosscheck.l1_capacity"] = {"from": [f"counter_l1_hit[a={a},b={b}]" for (a, b) in l1], "method": "ncu: largest 128 B-stride chase footprint with >= 90 % L1 sector hits per shared-memory request (a cyclic chase past capacity drops to near 0 under LRU)"}
        s["counter_crosscheck.l1_hit"] = {"from": [f"counter_l1_hit[a={a},b={b}]" for (a, b) in l1], "method": "ncu: L1 sector hit rate of a single-thread 128 B-stride chase (ld.global.ca) by footprint and shared-memory request; compare l1_bytes_by_carveout (B14)"}
    return ({"counter_crosscheck": cc} if cc else {}), s


# ---------------------------------------------------------------- B27 dependency / ILP / TLP
def fit_b27(recs):
    f, s = {}, {}
    sb = keyed(recs, "scoreboard_latency_cycles")
    if sb:
        f["scoreboard_latency_cycles"] = {k: r["value"] for k, r in sb.items()}
        s["scoreboard_latency_cycles"] = {"from": [r["quantity"] for r in sb.values()], "method": "B27: fma variant a=1 w=1 (one dependent FFMA chain) and lds variant a=1 w=1 (one shared-memory chase), cycles per instruction; confirms B17.5", "ci95": {k: r.get("ci95") for k, r in sb.items()}}
    g = recs.get("issue_bound_fraction_grid")
    if g:
        grid = g["detail"].get("grid_a_w_L_cyclesPerIter_fraction") or []
        f["issue_bound_fraction"] = {f"a={a},w={w},L={L}": frac for a, w, L, cyc, frac in grid}
        f["dependency_cycles_per_iter"] = {f"a={a},w={w},L={L}": cyc for a, w, L, cyc, frac in grid}
        # the headline: the smallest a*w whose dep loop at L=16 reaches 90 % of the issue bound
        hit = sorted((a * w, a, w) for a, w, L, cyc, frac in grid if L == 16 and frac >= 0.9)
        f["ilp_to_hide_lds"] = {"instructions_in_flight": hit[0][0], "a": hit[0][1], "w": hit[0][2]} if hit else None
        s["issue_bound_fraction"] = {"from": "issue_bound_fraction_grid", "method": "B27 dep variant: per iteration a ld.shared then a dependent FFMA per warp, L iterations unrolled, 4w warps per SM (w per scheduler); fraction = warp-instructions issued per scheduler cycle (bound 1). The grid is a prediction target of the warp model, not an input", "inferred_from": grid}
        s["ilp_to_hide_lds"] = {"from": "issue_bound_fraction_grid", "method": "B27: smallest a x w at L=16 with fraction >= 0.9"}
    return f, s


# ---------------------------------------------------------------- B28 LSU / shared-memory port contention
def fit_b28(recs):
    f, s = {}, {}
    t = recs.get("smem_port_contention_table")
    rows = keyed(recs, "lsu_contention")
    if not rows:
        return f, s
    by = {}
    for k, r in rows.items():
        d = r["detail"]; by.setdefault(d["mix"], {})[int(d["blocks_per_sm"])] = d
    lds = by.get("lds", {}); copy = by.get("copy", {}); both = by.get("both", {})
    if lds:
        agg = {str(b): d["smem_bytes_per_cycle_per_sm"] for b, d in sorted(lds.items())}
        f["lds_rate_by_residency"] = {str(b): {"cycles_per_step_per_block": d["cycles_per_step_per_block"], "smem_bytes_per_cycle_per_sm": d["smem_bytes_per_cycle_per_sm"], "per_block_time_cv": d["per_block_time_cv"]} for b, d in sorted(lds.items())}
        plateau = max(agg.values()); bstar = min(int(b) for b, v in agg.items() if v >= 0.95 * plateau)
        f["smem_port_bytes_per_cycle_per_sm"] = round(plateau, 1); f["smem_port_plateau_from_blocks"] = bstar
        bmax = max(lds); f["lsu_instr_per_cycle_per_sm"] = round(bmax * 8 * 12 / lds[bmax]["cycles_per_step_per_block"], 3)     # 8 warps x 12 ld.shared per step per block
        f["lsu_instr_per_cycle_per_sm_by_residency"] = {str(b): round(b * 8 * 12 / d["cycles_per_step_per_block"], 3) for b, d in sorted(lds.items())}
        s["lsu_instr_per_cycle_per_sm"] = {"from": [r["quantity"] for k, r in rows.items() if "lds" in k], "method": "B28 lds mix at the highest residency: ld.shared warp-instructions (4 x 128-bit + 8 x 32-bit per thread per step) accepted per SM cycle; the per-residency table shows the rate rising with occupancy (0.47 at one block to 1.02 at six)"}
        cvs = [d["per_block_time_cv"] for b, d in lds.items() if b >= 2]
        f["smem_arbiter"] = ("fair-share" if cvs and max(cvs) <= 0.10 else "unequal") if cvs else None
        s["smem_port_bytes_per_cycle_per_sm"] = {"from": [r["quantity"] for r in rows.values()], "method": "B28 lds mix: aggregate shared-memory bytes per SM cycle (b blocks x per-block bytes over the median block's time), plateau over b = 1..6", "inferred_from": agg}
        s["smem_arbiter"] = {"from": [r["quantity"] for k, r in rows.items() if "lds" in k], "method": "B28: spread (cv) of per-block times at b >= 2; <= 10 % = fair share"}
        s["lds_rate_by_residency"] = {"from": [r["quantity"] for k, r in rows.items() if "lds" in k], "method": "B28 lds mix per residency"}
    if copy:
        f["copy_bytes_per_s_per_sm_by_residency"] = {str(b): d["copy_bytes_per_s_per_sm"] for b, d in sorted(copy.items())}
        s["copy_bytes_per_s_per_sm_by_residency"] = {"from": [r["quantity"] for k, r in rows.items() if "copy" in k], "method": "B28 copy mix: 16 B cp.async per thread per phase, aggregate bytes per second per SM by residency"}
    if both and lds and copy:
        # shared path: in the mixed run the lds and copy rates both drop against their pure runs at the same residency
        shared = []
        for b in both:
            if b in lds and b in copy:
                shared.append(both[b]["smem_bytes_per_cycle_per_sm"] < 0.85 * lds[b]["smem_bytes_per_cycle_per_sm"])
        f["lsu_shared_by_copies"] = bool(shared and sum(shared) >= len(shared) / 2)
        f["lsu_copy_share_of_smem_rate"] = {str(b): round(both[b]["smem_bytes_per_cycle_per_sm"] / lds[b]["smem_bytes_per_cycle_per_sm"], 3) for b in sorted(both) if b in lds}
        f["lsu_mixed_by_residency"] = {str(b): {"smem_bytes_per_cycle_per_sm": d["smem_bytes_per_cycle_per_sm"], "copy_bytes_per_s_per_sm": d["copy_bytes_per_s_per_sm"], "cycles_per_step_per_block": d["cycles_per_step_per_block"]} for b, d in sorted(both.items())}
        s["lsu_shared_by_copies"] = {"from": [r["quantity"] for k, r in rows.items() if "both" in k], "method": "B28 both mix vs the lds mix at the same residency: shared when the shared-memory rate falls below 85 % of its lds-only value in at least half the residencies (the copy rate in the mixed run is confounded by the longer phase and is not used)"}
    return f, s


# ---------------------------------------------------------------- B30 the scheduler-to-LSU queue
def fit_b30(recs):
    f, s = {}, {}
    t = recs.get("mio_queue_depth_table")
    if t:
        tab = t["detail"].get("table_n_cyclesPerLoad_1warp_2warps") or []
        big = [c1 for n, c1, c2 in tab if n >= 8]; big2 = [c2 for n, c1, c2 in tab if n >= 8]
        f["mio_issue_cycles_per_load"] = round(sum(big) / len(big), 2) if big else None
        f["mio_issue_cycles_per_load_two_warps_per_smsp"] = round(sum(big2) / len(big2), 2) if big2 else None
        jump = [n for n, c1, c2 in tab if n >= 8 and c1 > 3 * min(big)]
        f["mio_queue_depth_per_smsp"] = (jump[0] if jump else None)
        f["mio_queue_depth_note"] = "no issue stall within 64 back-to-back ld.shared from one warp: the dispatch cost per load, not a queue depth, limits issue" if not jump else None
        f["mio_queue_scope"] = "per-smsp" if (f["mio_issue_cycles_per_load_two_warps_per_smsp"] or 0) > 1.6 * (f["mio_issue_cycles_per_load"] or 1) else "per-warp-or-wider"
        s["mio_issue_cycles_per_load"] = _nm(t, "mio_queue_depth_table", "B30.1: one warp per SM issues n back-to-back ld.shared.32; cycles per issued load (clock64 before the first, after the last issue), mean over n >= 8", inferred_from=tab)
        s["mio_queue_depth_per_smsp"] = {"from": "mio_queue_depth_table", "method": "B30.1: the n at which cycles per issued load exceeds 3x the burst value; null when no stall appears up to 64"}
        s["mio_queue_scope"] = {"from": "mio_queue_depth_table", "method": "B30.1: two warps on the same sub-partition issuing bursts; per-smsp when each warp's cycles per load rise by more than 1.6x"}
    dr = keyed(recs, "lds_instr_per_cycle_per_sm")
    wf = {}
    for q, r in recs.items():
        m = re.fullmatch(r"counter_lds_wavefronts\[a=(\d+),b=(\d+)\]", q)
        if m and r["detail"].get("measured"):
            mt = {k: v[0] for k, v in r["detail"]["metrics"].items()}
            w = mt.get("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"); i = mt.get("sm__sass_inst_executed_op_shared_ld.sum"); cyc = mt.get("sm__cycles_active.avg")
            wf[f"width={m.group(1)},stride={m.group(2)}"] = {"wavefronts_per_instruction": round(w / i, 2) if w and i else None, "instr_per_cycle_per_sm": round(i / 48 / cyc, 3) if i and cyc else None,
                                                               "wavefronts_per_cycle_per_sm": round(w / 48 / cyc, 3) if w and cyc else None, "mio_throttle_per_issue_active": mt.get("smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio")}
    if dr:
        f["lds_instr_per_cycle_per_sm_by_width_stride"] = {k: round(r["value"], 3) for k, r in dr.items() if "span" not in k}
        s["lds_instr_per_cycle_per_sm_by_width_stride"] = {"from": [r["quantity"] for r in dr.values()], "method": "B30.2: 16 warps per SM in bursts of 32 independent ld.shared with independent consumers, instructions per SM cycle by width (bytes) and word stride"}
    if wf:
        f["lds_wavefronts_per_instruction"] = {k: v["wavefronts_per_instruction"] for k, v in wf.items()}
        rates = [v["wavefronts_per_cycle_per_sm"] for v in wf.values() if v["wavefronts_per_instruction"] and v["wavefronts_per_instruction"] >= 2 and v["wavefronts_per_cycle_per_sm"]]
        f["lsu_drain_wavefronts_per_cycle_per_sm"] = round(sum(rates) / len(rates), 3) if rates else None
        one = [v["instr_per_cycle_per_sm"] for v in wf.values() if v["wavefronts_per_instruction"] and v["wavefronts_per_instruction"] < 1.5 and v["instr_per_cycle_per_sm"]]
        f["lsu_min_cycles_per_instruction"] = round(1 / max(one), 2) if one else None
        s["lds_wavefronts_per_instruction"] = {"from": [q for q in recs if q.startswith("counter_lds_wavefronts[")], "method": "B30.3 (ncu): l1tex shared-memory wavefronts per ld.shared instruction by width and stride, 16 warps per SM in bursts of 8", "inferred_from": wf}
        s["lsu_drain_wavefronts_per_cycle_per_sm"] = {"from": [q for q in recs if q.startswith("counter_lds_wavefronts[")], "method": "B30.2 (ncu): wavefronts per active SM cycle, mean over the configurations needing two or more wavefronts per instruction (every one of them drains at 0.96-1.0)"}
        s["lsu_min_cycles_per_instruction"] = {"from": [q for q in recs if q.startswith("counter_lds_wavefronts[")], "method": "B30.2 (ncu): one-wavefront loads cap at 0.48 instructions per SM cycle: at least two LSU cycles per instruction"}
    # occupancy sweep (ncu): shared-load instructions per cycle vs warps per sub-partition, 32-bit and 128-bit conflict-free
    occ = {}
    for q, r in recs.items():
        m = re.fullmatch(r"counter_lds_occupancy\[a=(\d+),b=(\d+)\]", q)
        if m and r["detail"].get("measured"):
            mt = {k: v[0] for k, v in r["detail"]["metrics"].items()}; cyc = mt.get("sm__cycles_active.avg"); ins = mt.get("sm__sass_inst_executed_op_shared_ld.sum"); w = mt.get("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum")
            ia, sc = mt.get("smsp__issue_active.avg"), mt.get("smsp__cycles_active.avg")
            occ.setdefault(m.group(1), {})[int(m.group(2))] = {"instr_per_cycle_per_sm": round(ins / 48 / cyc, 3), "instr_per_cycle_per_smsp": round(ins / 48 / cyc / 4, 4), "wavefronts_per_cycle_per_sm": round(w / 48 / cyc, 3),
                                                                "issue_active": round(ia / sc, 3) if ia and sc else None, "mio_throttle_per_issue_active": mt.get("smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"),
                                                                "short_scoreboard_per_issue_active": mt.get("smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio")}
    clk = keyed(recs, "lds_dispatch_instr_per_cycle_per_sm")
    if occ:
        f["lds_dispatch_rate_by_warps_per_smsp"] = {f"width={wd}": {str(w): v for w, v in sorted(d.items())} for wd, d in occ.items()}
        if "4" in occ:
            f["lsu_instr_floor_cycles_by_warps_per_sm"] = {str(4 * w): round(1.0 / v["instr_per_cycle_per_sm"], 2) for w, v in sorted(occ["4"].items())}
        s["lds_dispatch_rate_by_warps_per_smsp"] = {"from": [q for q in recs if q.startswith("counter_lds_occupancy[")] + [r["quantity"] for r in clk.values()], "method": "B30 occupancy sweep (ncu): w blocks of 4 warps per SM (one warp per sub-partition each), bursts of 32 conflict-free loads with independent consumers; shared-load instructions per SM cycle and per sub-partition, wavefronts per cycle, issue-active and mio_throttle at w = 1, 2, 4, 8, 12 warps per sub-partition", "inferred_from": {k: v for k, v in occ.items()}}
        s["lsu_instr_floor_cycles_by_warps_per_sm"] = {"from": [q for q in recs if q.startswith("counter_lds_occupancy[a=4")], "method": "B30 occupancy sweep: SM cycles per one-wavefront load (the inverse of the 32-bit rate) by resident warps per SM; replaces the single 2.1-cycle floor in the warp model"}
    if clk:
        f["lds_dispatch_instr_per_cycle_per_sm_clock"] = {k: round(r["value"], 3) for k, r in clk.items()}
    sh = keyed(recs, "mio_share_issue_cycles")
    if sh:
        g = lambda k: _val(sh.get(k))
        f["cp_async_issue_cycles_per_chunk"] = g("cp_async_alone"); f["ldmatrix_issue_cycles"] = g("ldmatrix_alone")
        f["mio_shared_with_cp_async"] = bool(g("cp_async_after_lds_burst") and g("cp_async_alone") and g("cp_async_after_lds_burst") > 1.5 * g("cp_async_alone"))
        f["mio_shared_with_ldmatrix"] = bool(g("ldmatrix_after_lds_burst") and g("ldmatrix_alone") and g("ldmatrix_after_lds_burst") > 1.5 * g("ldmatrix_alone"))
        s["mio_shared_with_cp_async"] = {"from": [r["quantity"] for r in sh.values()], "method": "B30.4: issue cycles of 8 cp.async chunks alone vs right after 32 ld.shared from the same warp; shared when the burst delays them by more than 1.5x", "inferred_from": {k: _val(r) for k, r in sh.items()}}
        s["mio_shared_with_ldmatrix"] = {"from": [r["quantity"] for r in sh.values()], "method": "B30.4: the same with 8 ldmatrix.x4"}
        s["cp_async_issue_cycles_per_chunk"] = _nm(sh.get("cp_async_alone"), "mio_share_issue_cycles[cp_async_alone]", "B30.4: clock64 around 8 cp.async 16 B chunks issued by one warp, per chunk (B15.3's 127 included the per-chunk address and guard code)")
        s["ldmatrix_issue_cycles"] = _nm(sh.get("ldmatrix_alone"), "mio_share_issue_cycles[ldmatrix_alone]", "B30.4: clock64 around 8 ldmatrix.x4 issued by one warp, per instruction")
    return f, s


FITTERS = [("b30_mio_queue", fit_b30), ("b27_dependency_ilp", fit_b27), ("b28_lsu_port_contention", fit_b28), ("b13_l2_structure", fit_b13), ("b14_l1_carveout", fit_b14), ("b15_copy_queue", fit_b15), ("b16_block_scheduler", fit_b16),
           ("b17_latency_hiding", None), ("b18_dram", fit_b18), ("b19_registers_local", fit_b19), ("b20_tensor_detail", fit_b20),
           ("b21_compute_block", fit_b21), ("b22_clock_governor", fit_b22), ("b23_unified_mechanics", fit_b23), ("b24_atomics_reductions", fit_b24),
           ("b25_multi_kernel", fit_b25), ("b26_functional_probes", fit_b26)]

FUNCTIONAL_FIELDS = ["cp_async_chunk_order", "cp_async_window_cycles", "cp_async_torn_chunks_observed", "wait_group_overcomplete_observed",
                     "threadfence_visibility_cycles", "tma_early_bytes_observed", "fma_contraction_default", "ftz_default",
                     "tensor_accumulate_ulp_bound", "smem_bank_map", "independent_thread_scheduling"]


def fit_micro(recs, contract):
    """Apply every fitter; functional fields go under contract['functional']. Returns (fields, sources, benchmarks_present)."""
    fields, sources, present = {}, {}, []
    # a counter measurement supersedes a timing inference of the same quantity: the sector size (B13.1 vs ncu L2 miss bytes per request)
    def reconcile(fields, sources):
        cc = fields.get("counter_crosscheck") or {}
        fill = cc.get("l2_fill_bytes_per_request_at_128B_plus")
        if fill and fields.get("l2_sector_bytes") and fill != fields["l2_sector_bytes"]:
            sources["l2_sector_bytes"] = dict(sources.get("l2_sector_bytes", {}), superseded_timing_inference=fields["l2_sector_bytes"],
                                              method="ncu: L2-miss bytes per request for a strided read at >= 128 B strides (counter); B13.1's timing inference gave " + str(fields["l2_sector_bytes"]) + " B and is kept in inferred_from")
            fields["l2_sector_bytes"] = int(fill)
        cap = cc.get("l2_capacity_mib_from_counters")
        if cap:
            fields["l2_capacity_bytes"] = int(cap) << 20
            sources["l2_capacity_bytes"] = {"from": sources.get("counter_crosscheck.l2_hit_rate", {}).get("from"), "method": "ncu: L2 capacity from the per-block cyclic-stream hit-rate curve (counter); B13's single-warp chase (l2_chase_capacity_bytes) varies between 16 and 24 MiB run to run and is secondary"}
        caps = cc.get("l1_capacity_bytes_from_counters_by_smem")
        if caps:
            fields["l1_bytes_by_carveout_counters"] = caps
            sources["l1_bytes_by_carveout_counters"] = sources.get("counter_crosscheck.l1_capacity", {})
    for name, fn in FITTERS:
        f, s = (fit_b17(recs, contract) if name == "b17_latency_hiding" else fn(recs))
        if f and not (name == "b19_registers_local" and "spill_penalty_curve" not in f):   # B19's granularity row is derived from B9
            present.append(name)
        fields.update(f); sources.update(s)
    cf, cs = fit_counters(recs)
    fields.update(cf); sources.update(cs)
    reconcile(fields, sources)
    functional = {k: fields.pop(k) for k in FUNCTIONAL_FIELDS if k in fields}
    if functional:
        fields["functional"] = functional
    return fields, sources, present
