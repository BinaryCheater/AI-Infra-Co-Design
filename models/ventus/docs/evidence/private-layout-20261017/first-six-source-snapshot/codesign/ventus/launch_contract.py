"""Resource admission for compiled programs and the pinned R1 driver interface.

This is a static capacity/representation check. It does not predict achieved
occupancy, allocator fragmentation, physical implementation or numerical accuracy.
"""

import hashlib
import json
from math import prod

from .config import Hardware

VERSION = "ventus-compiled-launch-contract-v2"
RTL_COMMIT = "681172541a8a34ffb43c483a19c075acbc11a4eb"
RTL_MAX_GROUPS_PER_AXIS = 1024
RTL_MAX_PRIVATE_BYTES_PER_THREAD = 4096


def resources(launch, hardware):
    """Reject impossible blocks before native instruction expansion/scheduling."""
    hardware.validate()
    if len(launch) < 20 or any(type(v) is not int or not 0 <= v < 2**32 for v in launch):
        raise ValueError("compiled launch must contain unsigned RV32 words")
    blocks, warps, threads, vgpr, sgpr, lds = launch[3:9]
    if blocks == 0 or warps == 0 or threads != hardware.threads:
        raise ValueError("compiled launch block/warp count or hardware lane width differs")
    if vgpr % hardware.rf_banks or sgpr % hardware.rf_banks:
        raise ValueError("compiled launch registers must occupy complete RF-bank rows")
    count = launch[19]
    tail = launch[20 + 2 * count :]
    if len(launch) < 20 + 2 * count or len(tail) not in (0, 1, 3, 5):
        raise ValueError("compiled launch has invalid memory/stack fields")
    usage = {"blocks": 1, "warps": warps, "vgpr": vgpr * warps, "sgpr": sgpr * warps, "lds": lds}
    capacity = {
        "blocks": hardware.blocks_per_sm,
        "warps": hardware.warps_per_sm,
        "vgpr": hardware.vgpr_slots,
        "sgpr": hardware.sgpr_slots,
        "lds": hardware.lds_bytes,
    }
    for name, value in usage.items():
        if value > capacity[name]:
            raise ValueError(
                f"compiled workgroup {name} allocation {value} exceeds per-SM capacity {capacity[name]}"
            )
    if len(tail) >= 3 and tail[1] + warps * tail[0] > lds:
        raise ValueError("compiled static data and per-warp stacks exceed allocated LDS")
    if len(tail) == 5 and tail[-1]:
        if warps > 1 and tail[3]:
            raise ValueError(
                "upstream private warp-base increments cannot guarantee lane isolation; use the corrected driver profile"
            )
        end = tail[2] + blocks * warps * threads * tail[-1] + (warps - 1) * tail[3]
        if end > 2**32:
            raise ValueError("compiled private address footprint exceeds RV32")
    return {
        "version": VERSION,
        "per_workgroup": usage,
        "per_sm": capacity,
        "resident_blocks_upper_bound": min(capacity[k] // v for k, v in usage.items() if v),
        "scope": "Static capacity upper bound; excludes allocator timing/fragmentation and achieved occupancy.",
    }


def rtl_resources(launch):
    """Admit only the fixed R1 source/driver resource envelope, without truncation."""
    result = resources(launch, Hardware())
    global_size, local_size = launch[10:13], launch[13:16]
    if any(n == 0 for n in (*global_size, *local_size)):
        raise ValueError("RTL launch dimensions must be positive")
    groups = [(g + extent - 1) // extent for g, extent in zip(global_size, local_size, strict=True)]
    if prod(local_size) != launch[4] * launch[5] or prod(groups) != launch[3] or launch[16] != 0:
        raise ValueError("RTL launch must describe the complete grid and full warps")
    if any(n > RTL_MAX_GROUPS_PER_AXIS for n in groups):
        raise ValueError("RTL launch exceeds declared 1024 workgroups per axis")
    tail = launch[20 + 2 * launch[19] :]
    private = tail[-1] if tail else 0
    if private > RTL_MAX_PRIVATE_BYTES_PER_THREAD:
        raise ValueError("RTL private bytes/thread exceed declared per-wavefront interface limit")
    result.update(
        {
            "target": "pinned-r1-source-and-driver",
            "rtl_commit": RTL_COMMIT,
            "workgroups_per_axis": groups,
            "private_bytes_per_wavefront": private * launch[5],
            "max_private_bytes_per_wavefront": RTL_MAX_PRIVATE_BYTES_PER_THREAD * 32,
            "scope": "Static source/driver capacity and representation checks; not binary elaboration or execution certification.",
        }
    )
    return result


def audit_rtl_source(root):
    """Check the reviewed declarations against the pinned, hash-bound source files."""
    manifest = json.loads((root / "SOURCE.json").read_text())
    if manifest["commit"] != RTL_COMMIT:
        raise ValueError("compiled launch contract requires the pinned R1 source")
    anchors = {
        "ventus/src/top/parameters.scala": [
            "def num_sm = 2",
            "var num_warp = 8",
            "var num_thread = 32",
            "def num_block = 8",
            "def num_bank = 4",
            "def num_vgpr:Int = 128*num_warp",
            "def num_sgpr:Int = 256*num_warp",
            "def sharedmem_depth = 1024",
            "def dcache_BlockWords: Int = 32",
            "def sharedmem_BlockWords = dcache_BlockWords",
            "def sharemem_size = sharedmem_depth * sharedmem_BlockWords * 4",
            "def NUM_WG_DIM = 1024",
            "val NUM_PDS_MAX = 4096*num_thread",
        ],
        "ventus/src/cta/cta_scheduler.scala": [
            "val num_wf = UInt(log2Ceil(CONFIG.WG.NUM_WF_MAX+1).W)",
            "val num_pds_per_wf = UInt(log2Ceil(CONFIG.WG.NUM_PDS_MAX+1).W)",
        ],
        "sim-verilator/cta_sche_wrapper.cpp": [
            "kernel->get_num_sgpr_per_wf() * kernel->get_num_wf()",
            "kernel->get_num_vgpr_per_wf() * kernel->get_num_wf()",
            "kernel->get_num_pds_per_thread() * kernel->get_num_thread()",
        ],
        "ventus/src/pipeline/operandCollector.scala": [
            "io.vgpr_base(io.writeVecCtrl.bits.warp_id) >> log2Ceil(num_bank).U",
            "io.sgpr_base(io.writeScalarCtrl.bits.warp_id) >> log2Ceil(num_bank).U",
        ],
    }
    evidence = []
    for name, patterns in anchors.items():
        data = (root / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != manifest["files"][name]:
            raise ValueError("compiled launch source differs from pinned manifest: " + name)
        lines = data.decode().splitlines()
        rules = []
        for pattern in patterns:
            hits = [i for i, line in enumerate(lines, 1) if pattern in line]
            if not hits:
                raise ValueError("compiled launch source rule changed: " + pattern)
            rules.append({"anchor": pattern, "lines": hits})
        evidence.append({"path": name, "sha256": digest, "rules": rules})
    baseline = Hardware().validate()
    if (
        baseline.sms,
        baseline.warps_per_sm,
        baseline.blocks_per_sm,
        baseline.threads,
        baseline.rf_banks,
        baseline.vgpr_slots,
        baseline.sgpr_slots,
        baseline.lds_bytes,
    ) != (2, 8, 8, 32, 4, 1024, 2048, 131072):
        raise ValueError("baseline hardware differs from reviewed R1 launch declarations")
    return {
        "version": VERSION,
        "rtl_commit": RTL_COMMIT,
        "files": evidence,
        "scope": "Reviewed launch capacity/representation declarations; not a general Scala elaborator or complete hardware audit.",
    }
