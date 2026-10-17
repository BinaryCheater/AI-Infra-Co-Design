"""Bind an explicit tile assignment to compiled cooperative FP32 GEMM source.

This registered constructor does not search. The checker regenerates its source,
launch and independent FP32 FMA references before compilation.
"""

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import math
import struct
from pathlib import Path

from .compiled_check import validate_names
from .config import Hardware

VERSION = "ventus-cooperative-gemm-lowering-v1"

BODY = """kernel void cooperative_gemm_f32(__global const float *a,
    __global const float *b, __global float *c, uint m, uint n, uint k) {
  __local float sa[TM*TK];
  __local float sb[TK*TN];
  uint lid=get_local_id(0), group=get_group_id(0);
  uint groups_n=(n+TN-1)/TN, lr=lid/TN, lc=lid%TN;
  uint row=(group/groups_n)*TM+lr, col=(group%groups_n)*TN+lc;
  float sum=0;
  for(uint at=0;at<k;at+=TK) {
    for(uint i=lid;i<TM*TK;i+=TM*TN) {
      uint r=(group/groups_n)*TM+i/TK, t=at+i%TK;
      sa[i]=(r<m && t<k) ? a[r*k+t] : 0;
    }
    for(uint i=lid;i<TK*TN;i+=TM*TN) {
      uint t=at+i/TN, q=(group%groups_n)*TN+i%TN;
      sb[i]=(t<k && q<n) ? b[t*n+q] : 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint i=0;i<TK && at+i<k;++i) sum=__builtin_fmaf(sa[lr*TK+i],sb[i*TN+lc],sum);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(row<m && col<n) c[row*n+col]=sum;
}
"""


def render(assignment):
    if set(assignment) != {"tile_m", "tile_n", "tile_k"}:
        raise ValueError("unsupported GEMM constructor assignment")
    m, n, k = [assignment[key] for key in ("tile_m", "tile_n", "tile_k")]
    if any(type(x) is not int or x <= 0 for x in (m, n, k)):
        raise ValueError("GEMM tile sizes must be positive integers")
    h = Hardware()
    if m * n % h.threads or m * n > h.warps_per_sm * h.threads:
        raise ValueError(
            "GEMM workgroup must contain complete warps within the target warp capacity"
        )
    if 4 * k * (m + n) >= h.lds_bytes:
        raise ValueError("GEMM static staging leaves no LDS capacity for compiler stacks")
    return f"// {VERSION}\n#define TM {m}u\n#define TN {n}u\n#define TK {k}u\n" + BODY


def f32(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def verify_cases(assignment, cases):
    validate_names(cases)
    # Use the host's correctly rounded FP32 FMA independently of generated code.
    lib = ctypes.CDLL(ctypes.util.find_library("m"))
    fma = lib.fmaf
    fma.argtypes = [ctypes.c_float] * 3
    fma.restype = ctypes.c_float
    tm, tn = assignment["tile_m"], assignment["tile_n"]
    for case in cases:
        arguments = case["arguments"]
        if len(arguments) != 6 or any(type(v) is not int or not 0 <= v < 2**32 for v in arguments):
            raise ValueError("GEMM ABI requires three RV32 pointers and M/N/K")
        ap, bp, cp, m, n, k = arguments
        if min(m, n, k) <= 0:
            raise ValueError("GEMM dimensions must be positive")
        if len(case["inputs"]) != 2 or len(case["outputs"]) != 1:
            raise ValueError("GEMM constructor requires exact A/B input and C output buffers")
        a, b = case["inputs"]
        c = case["outputs"][0]
        if [a["address"], b["address"], c["address"]] != [ap, bp, cp] or [
            len(a["words"]),
            len(b["words"]),
            len(c["expected_words"]),
        ] != [m * k, k * n, m * n]:
            raise ValueError("GEMM shape/argument/buffer binding differs")
        if m * k + k * n + m * n > 262144:
            raise ValueError("GEMM reference exceeds the compiled checker's 1 MiB payload budget")
        if any(
            type(v) is not int or not 0 <= v < 2**32
            for v in a["words"] + b["words"] + c["expected_words"]
        ):
            raise ValueError("GEMM buffers require RV32 words")
        if case["local_size"] != [tm * tn, 1, 1] or case["global_size"] != [
            ((m + tm - 1) // tm) * ((n + tn - 1) // tn) * tm * tn,
            1,
            1,
        ]:
            raise ValueError("GEMM tile assignment and launch differ")
        av, bv = list(map(f32, a["words"])), list(map(f32, b["words"]))
        if any(not math.isfinite(v) for v in av + bv):
            raise ValueError("GEMM reference currently requires finite FP32 inputs")
        expected = []
        for r in range(m):
            for q in range(n):
                acc = 0.0
                for t in range(k):
                    acc = fma(av[r * k + t], bv[t * n + q], acc)
                expected.append(bits(acc))
        if c["expected_words"] != expected:
            raise ValueError("GEMM reference differs from independent FP32 FMA")


def verify_binding(spec, source):
    binding = spec.get("binding")
    if (
        not isinstance(binding, dict)
        or set(binding) != {"constructor", "assignment"}
        or binding["constructor"] != VERSION
    ):
        raise ValueError("unknown concrete lowering binding")
    expected = render(binding["assignment"])
    if (
        source != expected
        or spec["kernel"] != "cooperative_gemm_f32"
        or hashlib.sha256(source.encode()).hexdigest() != spec["source_sha256"]
    ):
        raise ValueError("source or entry differs from the selected GEMM assignment")
    verify_cases(binding["assignment"], spec["cases"])
    return {
        "constructor": VERSION,
        "assignment": binding["assignment"],
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "scope": "Registered source/launch/reference correspondence; not universal correctness, optimality or hardware timing proof",
    }


def materialize(assignment, cases, out):
    source = render(assignment)
    validate_names(cases)
    if any(len(c["arguments"]) != 6 for c in cases):
        raise ValueError("GEMM constructor requires the complete six-word ABI")
    tm, tn = assignment["tile_m"], assignment["tile_n"]
    cases = [
        {
            **c,
            "local_size": [tm * tn, 1, 1],
            "global_size": [
                ((c["arguments"][3] + tm - 1) // tm)
                * ((c["arguments"][4] + tn - 1) // tn)
                * tm
                * tn,
                1,
                1,
            ],
        }
        for c in cases
    ]
    verify_cases(assignment, cases)
    out.mkdir(parents=True, exist_ok=False)
    (out / "kernel.cl").write_text(source)
    spec = {
        "schema": "ventus-compiled-workflow-v1",
        "source": "kernel.cl",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "kernel": "cooperative_gemm_f32",
        "optimization": "O1",
        "workitem_profile": "rtl-r1",
        "cases": cases,
        "binding": {"constructor": VERSION, "assignment": assignment},
    }
    (out / "workflow.json").write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    for key in ("m", "n", "k"):
        parser.add_argument("--tile-" + key, type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())["cases"]
    cases = [
        {k: v for k, v in c.items() if k not in {"elf", "elf_sha256", "kernel"}} for c in cases
    ]
    materialize(
        {"tile_m": args.tile_m, "tile_n": args.tile_n, "tile_k": args.tile_k}, cases, args.out
    )


if __name__ == "__main__":
    main()
