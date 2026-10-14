#!/usr/bin/env python3
"""The fitter: raw records -> one measured hardware contract (notes/hardware_calibration_design.md §2.3).

  fit.py --raw calibration/<host>/raw --out calibration/<host>/contract.yaml [--name spark-6951-gb10]

Every value in the contract names the raw record it came from. The contract validates against
agentic/schema/chip.json.
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load(raw: Path) -> dict:
    recs = {}
    for f in raw.glob("*.json"):
        if f.name.startswith("_"):
            continue
        r = json.loads(f.read_text())
        recs[r["quantity"]] = r
    return recs


def keyed(recs, prefix):
    """quantity 'name[key]' -> {key: record}, keys parsed as int when possible."""
    out = {}
    for q, r in recs.items():
        m = re.fullmatch(re.escape(prefix) + r"\[(.+)\]", q)
        if m:
            k = m.group(1)
            if "=" in k:
                k = k.split("=", 1)[1]
            out[int(k) if k.isdigit() else k] = r
    return out


def plateau(series: dict, *, large=True):
    """Median of the records on the large-working-set plateau (>= 1 GiB) or the small one (<= 8 MiB)."""
    sel = [r for k, r in series.items() if (k >= (1 << 30) if large else k <= (8 << 20))]
    if not sel:
        return None, []
    vals = sorted(r["value"] for r in sel)
    return vals[len(vals) // 2], [r["quantity"] for r in sel]


def knee(series: dict):
    """Effective capacity: the largest working set whose bandwidth is still above 60 % of the small plateau."""
    small, _ = plateau(series, large=False)
    large, _ = plateau(series, large=True)
    if small is None or large is None:
        return None
    thr = large + 0.6 * (small - large)
    sizes = sorted(series)
    cap = None
    for k in sizes:
        if series[k]["value"] >= thr:
            cap = k
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None); ap.add_argument("--datasheet-bw", type=float, default=None)
    a = ap.parse_args()
    raw = Path(a.raw); recs = load(raw)
    summary = json.loads((raw / "_summary.json").read_text()) if (raw / "_summary.json").exists() else {}
    props = recs["device_properties"]["detail"]
    host = summary.get("host", raw.parent.name)
    name = a.name or f"{host}-{props['name'].split()[-1].lower()}"

    src = {}
    def take(field, quantity, transform=lambda v: v):
        r = recs[quantity]
        src[field] = {"from": quantity, "ci95": r.get("ci95"), "n": r.get("n"), "method": r.get("method")}
        return transform(r["value"])

    bw_read = keyed(recs, "bandwidth_read"); bw_write = keyed(recs, "bandwidth_write"); bw_copy = keyed(recs, "bandwidth_copy")
    lat = keyed(recs, "latency"); smem = keyed(recs, "smem_bytes_per_s_per_sm"); bar = keyed(recs, "barrier_s")
    dram_read, dram_src = plateau(bw_read, large=True)
    dram_write, _ = plateau(bw_write, large=True)
    dram_copy, _ = plateau(bw_copy, large=True)
    l2_bw, l2_src = plateau(bw_read, large=False)
    l2_eff = knee(bw_read)
    lat_sizes = sorted(lat)
    lat_l1 = lat[lat_sizes[0]]["value"] if lat_sizes else None
    lat_l2 = min((lat[k]["value"] for k in lat_sizes if (1 << 20) <= k <= (8 << 20)), default=None)
    lat_dram = max((lat[k]["value"] for k in lat_sizes if k >= (256 << 20)), default=None)

    contract = {
        "name": name, "kind": "measured", "gpu": props["name"], "compute_capability": props["cc"],
        "sms": props["sms"], "warp_size": props["warp_size"],
        "smem_bytes_per_block": props["smem_bytes_per_block_optin"], "smem_bytes_per_block_default": props["smem_bytes_per_block"],
        "smem_bytes_per_sm": props["smem_bytes_per_sm"], "regs_per_sm": props["regs_per_sm"], "l2_bytes": props["l2_bytes"],
        "max_threads_per_block": props["max_threads_per_block"], "max_blocks_per_sm": props["max_blocks_per_sm"],
        "bytes_per_element": 4, "threads_per_block": 256, "max_accumulators_per_thread": 128,
        "fp32_flops_per_s_per_sm": take("fp32_flops_per_s_per_sm", "fp32_flops_per_s_per_sm"),
        "dram_bytes_per_s": dram_read, "dram_bytes_per_s_write": dram_write, "dram_bytes_per_s_copy": dram_copy,
        "l2_bytes_effective": l2_eff, "l2_bytes_per_s": l2_bw,
        "latency_ns": {"l1": lat_l1, "l2": lat_l2, "dram": lat_dram},
        "smem_bytes_per_s_per_sm": smem[1]["value"] if 1 in smem else None,
        "bank_conflict_penalty": {str(k): round(smem[1]["value"] / r["value"], 3) for k, r in sorted(smem.items())} if 1 in smem else {},
        "barrier_s": {str(k): r["value"] for k, r in sorted(bar.items())},
        "datasheet": {"dram_bytes_per_s": a.datasheet_bw} if a.datasheet_bw else {},
        "fixed": ["sms", "smem_bytes_per_block", "smem_bytes_per_sm", "regs_per_sm", "l2_bytes", "dram_bytes_per_s",
                  "fp32_flops_per_s_per_sm", "warp_size", "max_threads_per_block"],
    }
    src.update({
        "dram_bytes_per_s": {"from": dram_src, "note": "median of the read plateau at >= 1 GiB working sets",
                             "ci95": max((bw_read[k]["ci95"] for k in bw_read if k >= (1 << 30)), default=None)},
        "l2_bytes_per_s": {"from": l2_src, "note": "median of the read plateau at <= 8 MiB"},
        "l2_bytes_effective": {"from": "bandwidth_read[*]", "note": "largest working set still above 60 % of the way from the DRAM plateau to the L2 plateau"},
        "latency_ns": {"from": [lat[k]["quantity"] for k in lat_sizes], "note": "l1 = smallest set; l2 = min over 1-8 MiB; dram = max over >= 256 MiB"},
        "smem_bytes_per_s_per_sm": {"from": "smem_bytes_per_s_per_sm[stride=1]", "ci95": smem[1].get("ci95") if 1 in smem else None},
        "barrier_s": {"from": [r["quantity"] for r in bar.values()], "ci95": {str(k): r.get("ci95") for k, r in bar.items()}},
    })
    # ---- F2 fields (present only after a --full run); absent quantities leave the field out
    tensor_sm = keyed(recs, "tensor_flops_per_s_per_sm"); tensor_chip = keyed(recs, "tensor_flops_per_s_chip")
    if tensor_sm:
        contract["tensor_flops_per_s_per_sm"] = {k: r["value"] for k, r in tensor_sm.items()}
        contract["tensor_shapes"] = {k: r["detail"].get("shape") for k, r in tensor_sm.items()}
        src["tensor_flops_per_s_per_sm"] = {"from": [r["quantity"] for r in tensor_sm.values()], "note": "WMMA mma_sync microkernel, one block per SM; chip-level cuBLAS numbers in tensor_flops_per_s_chip"}
    if tensor_chip:
        contract["tensor_flops_per_s_chip"] = {k: r["value"] for k, r in tensor_chip.items()}
    if "tensor_fp8" in recs:
        contract.setdefault("tensor_flops_per_s_per_sm", {})["fp8"] = None
        src["tensor_fp8"] = recs["tensor_fp8"]["detail"]
    acl = keyed(recs, "async_copy_latency_s"); acb = keyed(recs, "async_copy_bytes_per_s")
    if acl:
        contract["async_copy_latency_s"] = {str(k): r["value"] for k, r in sorted(acl.items())}
        contract["async_copy_bytes_per_s"] = {str(k): r["value"] for k, r in sorted(acb.items())}
        src["async_copy_latency_s"] = {"from": [r["quantity"] for r in acl.values()], "note": "cp.async 16 B/thread, one tile in flight, median SM, s per tile at the attribute clock"}
    if "tma_supported" in recs:
        contract["tma_supported"] = bool(recs["tma_supported"]["value"]) and bool(recs["tma_supported"]["detail"].get("correct", False))
        src["tma_supported"] = recs["tma_supported"]["detail"]
        if "tma_bulk_latency_s[16384]" in recs:
            contract["tma_bulk_latency_s"] = {"16384": recs["tma_bulk_latency_s[16384]"]["value"]}
    occ = keyed(recs, "max_blocks_per_sm")
    if occ:
        contract["max_blocks_per_sm_measured"] = {str(k): int(r["value"]) for k, r in occ.items()}
        regs = {k: r for k, r in occ.items() if isinstance(k, int) and r["detail"].get("regs_per_thread")}
        src["max_blocks_per_sm_measured"] = {"from": [r["quantity"] for r in occ.values()], "note": "smem=... rows measured by resident-block count; regs=... rows from the occupancy API with __launch_bounds__ variants"}
    if "launch_s" in recs:
        contract["launch_s"] = take("launch_s", "launch_s")
        contract["launch_host_s"] = recs["launch_host_s"]["value"] if "launch_host_s" in recs else None
    if "sustained_clock_mhz" in recs:
        sc = recs["sustained_clock_mhz"]["samples"]
        contract["clock_hz_sustained"] = sorted(sc)[len(sc) // 2] * 1e6 if sc else None
        contract["clock_variance_sustained"] = (max(sc) - min(sc)) / max(sc) if sc and max(sc) else None
        contract["sustained_fp32_flops_per_s_per_sm"] = recs["sustained_fp32_flops_per_s_per_sm"]["value"]
        contract["power_w_sustained"] = recs["sustained_power_w"]["value"] if "sustained_power_w" in recs else None
        src["clock_hz_sustained"] = {"from": "sustained_clock_mhz", "n": len(sc), "note": "nvidia-smi sampled once per second during a 60 s FMA load"}
    um = keyed(recs, "unified_read_bytes_per_s")
    if um:
        dev = um.get("device", {}).get("value")
        contract["unified_memory"] = {k: r["value"] for k, r in um.items()}
        if dev:
            contract["unified_memory_penalty"] = {k: round(dev / r["value"], 3) for k, r in um.items() if k != "device" and r["value"]}
            contract["coherent_access_bytes_per_s"] = um.get("pinned_host", {}).get("value")
        src["unified_memory"] = {"from": [r["quantity"] for r in um.values()], "note": "penalty = device read bandwidth / this allocation's read bandwidth, 1 GiB working set"}
    contract["suite_level"] = "full" if tensor_sm and um else "minimum"

    clocks = [r["clocks"] for r in recs.values() if r.get("clocks")]
    sm_mhz = [c["after"]["sm_mhz"] for c in clocks if c.get("after", {}).get("sm_mhz")]
    contract["provenance"] = {
        "suite_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip() or None,
        "host": host, "driver": summary.get("driver"), "toolkit": summary.get("nvcc"), "arch": summary.get("arch"),
        "ncu_available": bool(summary.get("ncu")),
        # under load: B11's per-second samples when present, else the driver's before/after samples
        # (which include idle readings between benchmarks and overstate the variance)
        "clocks": {"locked": False,
                   "sustained_hz": contract.get("clock_hz_sustained"),
                   "variance": contract.get("clock_variance_sustained") if contract.get("clock_variance_sustained") is not None
                               else ((max(sm_mhz) - min(sm_mhz)) / max(sm_mhz) if sm_mhz and max(sm_mhz) else None),
                   "variance_source": "B11 sustained load" if contract.get("clock_variance_sustained") is not None else "driver before/after samples",
                   "sampled_sm_mhz": {"min": min(sm_mhz) if sm_mhz else None, "max": max(sm_mhz) if sm_mhz else None},
                   "max_sm_mhz": clocks[0]["after"].get("max_sm_mhz") if clocks else None},
        "date": summary.get("finished") or summary.get("started"), "raw": str(raw.relative_to(ROOT)) if raw.is_relative_to(ROOT) else str(raw),
        "ci_target": summary.get("ci_target"), "max_reps": summary.get("max_reps"), "sources": src,
    }
    from agentic.schema import validate_chip
    validate_chip(contract)
    Path(a.out).write_text("# Measured hardware contract, written by calibration/suite/fit.py. Do not edit by hand.\n"
                           + yaml.safe_dump(contract, sort_keys=False, allow_unicode=True))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
