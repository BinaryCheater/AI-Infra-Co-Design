"""Instruction counts per phase read from the compiled kernel (cuobjdump -sass), not estimated.

The generated kernel's phase loop contains: the copy loops (one per operand, bodies with LDGSTS), the wait and the
barriers, and the k-loop (the FFMA body, unrolled by the compiler). Per warp and phase the dynamic count is

    outside + k_body x k_trips + copyA_body x warp_iters_A / warps + copyB_body x warp_iters_B / warps

where outside = the phase body minus the loop bodies it contains, k_trips = TK / unroll (the unroll is read from the
FFMA count of the k body), and a copy loop's warp-iterations are ceil(chunks / 32) spread over the block's warps (each
iteration moves one 16 B chunk per lane). Loops are found by backward branches; the k body is the loop with the most
FFMAs; copy loops are the ones with LDGSTS (or LDG when the kernel copies synchronously).

The compile runs on the GPU host (ssh alias; no address here) and the counts are cached by source hash in
calibration/<host>/sass_counts.json so a re-simulation needs no host.
"""
from __future__ import annotations

import hashlib, json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REMOTE_DIR = "~/work/research/AI-Infra-Co-Design-calib/sass"


def _loops(rows):
    loops = []
    for addr, op, rest in rows:
        m = re.search(r"0x([0-9a-f]+)", rest)
        if op.startswith("BRA") and m and int(m.group(1), 16) < addr:
            loops.append((int(m.group(1), 16), addr))
    return loops


def parse_sass(text: str) -> dict:
    seg = text.split("Function : smem_tiling_gemm")[1] if "Function : smem_tiling_gemm" in text else text
    seg = seg.split("Function :")[0]
    ins = re.findall(r"/\*([0-9a-f]+)\*/\s+(?:(@!?U?P\d+)\s+)?([A-Z][A-Z0-9_.]*)([^;]*);", seg)
    rows = [(int(a, 16), op, rest) for a, _, op, rest in ins]
    loops = _loops(rows)
    if not loops:
        return {"error": "no loops found", "total": len(rows)}
    def body(lo, hi):
        return [op for a, op, _ in rows if lo <= a <= hi]
    def hist(b):
        h = {}
        for o in b:
            k = o.split(".")[0]; h[k] = h.get(k, 0) + 1
        return h
    info = []
    for lo, hi in loops:
        b = body(lo, hi); h = hist(b)
        info.append({"lo": lo, "hi": hi, "n": len(b), "ffma": h.get("FFMA", 0), "lds": sum(v for k, v in h.items() if k == "LDS"), "ldgsts": h.get("LDGSTS", 0) + h.get("LDG", 0), "bar": h.get("BAR", 0)})
    k = max(info, key=lambda l: l["ffma"])
    phase = max((l for l in info if l["lo"] <= k["lo"] and l["hi"] >= k["hi"] and l is not k), key=lambda l: l["n"], default=None)
    copies = [l for l in info if l["ldgsts"] > 0 and l is not k and not (l["lo"] <= k["lo"] and k["hi"] <= l["hi"]) and (phase is None or (phase["lo"] <= l["lo"] and l["hi"] <= phase["hi"]))]
    # the copy loops may be nested in each other's ranges only if one contains the other: keep the innermost ones
    copies = [c for c in copies if not any(o is not c and c["lo"] <= o["lo"] and o["hi"] <= c["hi"] for o in copies)]
    # copy code that is not a loop (chunks <= threads): the region a forward branch skips around each LDGSTS, executed only
    # by the warps that own a chunk
    straight = []
    if phase:
        inloop = lambda a: any(c["lo"] <= a <= c["hi"] for c in copies)
        for i, (addr, op, rest) in enumerate(rows):
            if op.startswith("LDGSTS") and phase["lo"] <= addr <= phase["hi"] and not inloop(addr):
                region = None
                for a2, op2, rest2 in reversed(rows[:i]):
                    m = re.search(r"0x([0-9a-f]+)", rest2)
                    if a2 < phase["lo"]:
                        break
                    if op2.startswith("BRA") and m and int(m.group(1), 16) > addr:
                        region = (a2, int(m.group(1), 16)); break
                if region:
                    straight.append(sum(1 for a3, _, _ in rows if region[0] < a3 < region[1]))
    return {"phase_body": phase["n"] if phase else None, "k_body": k["n"], "k_body_ffma": k["ffma"], "k_body_lds": k["lds"],
            "copy_bodies": [c["n"] for c in copies], "copy_bodies_ldgsts": [c["ldgsts"] for c in copies], "copy_straight": straight, "total": len(rows)}


def phase_instructions(sass: dict, tm: int, tn: int, tk: int, threads: int = 256) -> dict:
    """Per-warp dynamic instructions per phase from the parsed SASS and the design."""
    rm, rn = tm // 16, tn // 16
    ffma_per_step = rm * rn
    unroll = max(1, round(sass["k_body_ffma"] / max(1, ffma_per_step)))
    k_trips = max(1, -(-tk // unroll))
    warps = threads // 32
    chunks = (tm * tk // 4, tk * tn // 4)                     # 16 B chunks of A and B per phase
    copy = 0.0
    bodies = list(sass.get("copy_bodies") or []); per_body = list(sass.get("copy_bodies_ldgsts") or [])
    # a copy loop that survived as a loop belongs to the operand with more chunks than threads; its body may hold several
    # LDGSTS (unrolled), so the trips per thread are the chunks per thread divided by the LDGSTS per body
    big = sorted(chunks, reverse=True)
    for i, body in enumerate(bodies[:2]):
        c = big[i] if i < len(big) else big[-1]
        per_thread = -(-c // threads)
        trips = max(1, -(-per_thread // max(1, per_body[i] if i < len(per_body) else 1)))
        copy += body * trips
    # straight-line copy code (no loop): every warp issues the branch, only ceil(chunks / 32) of the warps run the region
    straight = list(sass.get("copy_straight") or [])
    small = sorted(chunks)                                   # the operands whose chunks fit in one pass, smallest first
    skipped = 0.0
    for i, region in enumerate(straight):
        c = small[i] if i < len(small) else small[-1]
        frac = min(1.0, (-(-c // 32)) / warps)
        skipped += region * (1.0 - frac)
    outside = (sass["phase_body"] or 0) - sass["k_body"] - sum(bodies) - skipped
    total = outside + sass["k_body"] * k_trips + copy
    return {"unroll": unroll, "k_trips": k_trips, "outside": outside, "k_loop": sass["k_body"] * k_trips, "copy": round(copy, 1),
            "copy_straight_skipped": round(skipped, 1), "per_warp_per_phase": round(total, 1), "k_body_lds_per_step": sass["k_body_lds"] / unroll}


def cache_path(host_dir: str = "spark-6951") -> Path:
    return ROOT / "calibration" / host_dir / "sass_counts.json"


def sass_counts(source: str, tm: int, tn: int, tk: int, host: str | None, host_dir: str = "spark-6951", arch: str = "sm_121") -> dict:
    """Parsed SASS and per-phase counts for a generated kernel; compiled on `host` unless cached by source hash."""
    key = hashlib.sha256(source.encode()).hexdigest()
    cp = cache_path(host_dir)
    cache = json.loads(cp.read_text()) if cp.exists() else {}
    if key in cache:
        return cache[key]
    if not host:
        raise RuntimeError("SASS count not cached and no host given")
    subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {REMOTE_DIR}"], check=True, capture_output=True)
    local = Path("/tmp") / f"sass_{key[:12]}.cu"; local.write_text(source)
    subprocess.run(["rsync", "-az", str(local), f"{host}:{REMOTE_DIR}/{key[:12]}.cu"], check=True, capture_output=True)
    cmd = f"export PATH=$PATH:/usr/local/cuda/bin; cd {REMOTE_DIR} && nvcc -O3 -arch={arch} -std=c++17 -cubin -o {key[:12]}.cubin {key[:12]}.cu && cuobjdump -sass {key[:12]}.cubin"
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True)
    parsed = parse_sass(r.stdout)
    out = {"sass": parsed, "counts": phase_instructions(parsed, tm, tn, tk) if "error" not in parsed else None, "arch": arch}
    cache[key] = out
    cp.write_text(json.dumps(cache, indent=1) + "\n")
    return out
