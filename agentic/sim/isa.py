"""The instruction stream (notes/instruction_stream.md): one dataclass per instruction as the SASS reader emits it, and the pipe
table that maps a SASS opcode to the pipe it issues on and the scoreboard class whose latency its consumers wait for. Every
cost comes from the measured contract: FFMA-relative rates from B21, latencies from B27/B31/B32/B38/B39, the queue from B30.
An opcode the table does not know is an error (`UnknownOpcode`), never a guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# pipes
FMA, ALU, LSU, TENSOR, XU, COPY, BAR, WAIT, BRANCH, UNIFORM, NONE, CONST = "fma", "alu", "lsu", "tensor", "xu", "copy", "bar", "wait", "branch", "uniform", "none", "const"


class UnknownOpcode(ValueError):
    pass


@dataclass
class Instr:
    opcode: str                       # the SASS mnemonic with its modifiers, e.g. "LDS.128", "HMMA.16816.F32.BF16"
    pipe: str                         # one of the pipes above
    sb_class: str                     # scoreboard class: fma, alu, lds, ldsm, gload, mma, sfu, cvt, shfl, none
    dst: tuple = ()                   # destination registers (R.., P.., UR..), widened: LDS.128 -> four registers
    src: tuple = ()                   # source registers
    pred: str | None = None           # guarding predicate, e.g. "!P0"
    addr: int = 0                     # SASS address (for traces and loop structure)
    width_bytes: int = 0              # memory ops: bytes per thread
    pattern: str = ""                 # memory ops: "lane-consecutive" | "broadcast" | "unknown" | "template:A" | "template:B"
    wavefronts: float = 0.0           # memory ops: LSU wavefronts when known (from the template's addresses), else 0 -> width rule
    chunk_ids: tuple = ()             # cp.async / TMA: the 16 B chunk ids this instruction issues (per thread one chunk)
    stage: int = 0                    # cp.async / TMA / DEPBAR: the copy stage (group) this belongs to or waits for
    barrier: str = ""                 # BAR: the barrier id
    count: int = 0                    # BAR: arrivals expected; DEPBAR: groups allowed pending
    cost: float = 1.0                 # cycles the pipe is busy per warp instruction on one sub-partition
    hold: float = 0.0                 # cycles the issuing warp is held after issue (mma 16, MUFU 8, cp.async chunk 6.8)
    tidx: int = -1                    # k-loop shared loads: position among the template's A (or B) loads, for the address-derived wavefronts


# opcode prefix -> (pipe, scoreboard class, B21 rate key or None)
PIPE_TABLE = {
    "FFMA": (FMA, "fma", "ffma"), "FADD": (FMA, "fma", "fadd"), "FMUL": (FMA, "fma", "fmul"), "FMNMX": (FMA, "fma", "fadd"), "FSEL": (FMA, "fma", "fadd"),
    "FSETP": (FMA, "fma", "fadd"), "FSET": (FMA, "fma", "fadd"), "HFMA2": (FMA, "fma", "ffma"), "FCHK": (FMA, "fma", "fadd"),
    "IMAD": (FMA, "alu", "imad"), "IADD3": (ALU, "alu", "iadd3"), "IADD": (ALU, "alu", "iadd3"), "LEA": (ALU, "alu", "imad"), "LOP3": (ALU, "alu", "lop3"),
    "SHF": (ALU, "alu", "shf"), "ISETP": (ALU, "alu", "iadd3"), "SEL": (ALU, "alu", "iadd3"), "VIMNMX": (ALU, "alu", "iadd3"), "MOV": (ALU, "alu", "mov"),
    "P2R": (ALU, "alu", "iadd3"), "R2P": (ALU, "alu", "iadd3"), "PRMT": (ALU, "alu", "lop3"), "IABS": (ALU, "alu", "iadd3"), "I2F": (XU, "cvt", "i2f_f2i"),
    "F2I": (XU, "cvt", "i2f_f2i"), "I2FP": (ALU, "alu", "i2f_f2i"), "F2FP": (XU, "cvt", None), "F2F": (XU, "cvt", None), "MUFU": (XU, "sfu", None),
    "LDS": (LSU, "lds", None), "LDSM": (LSU, "ldsm", None), "LDG": (LSU, "gload", None), "STG": (LSU, "gstore", None), "STS": (LSU, "sts", None),
    "ATOMS": (LSU, "lds", None), "LDGSTS": (COPY, "none", None), "LDGDEPBAR": (NONE, "none", None), "DEPBAR": (WAIT, "none", None), "BAR": (BAR, "none", None),
    "HMMA": (TENSOR, "mma", None), "IMMA": (TENSOR, "mma", None), "QMMA": (TENSOR, "mma", None), "SHFL": (LSU, "shfl", None),
    "BRA": (BRANCH, "none", None), "BSSY": (BRANCH, "none", None), "BSYNC": (BRANCH, "none", None), "EXIT": (BRANCH, "none", None), "RET": (BRANCH, "none", None),
    "NOP": (NONE, "none", None), "CS2R": (ALU, "alu", "mov"), "S2R": (ALU, "alu", "mov"), "S2UR": (UNIFORM, "alu", "mov"), "LDC": (CONST, "const", None), "LDCU": (CONST, "const", None),
    "SYNCS": (LSU, "mbar", None), "UTMALDG": (COPY, "none", None), "UBLKCP": (COPY, "none", None), "R2UR": (ALU, "alu", "mov"), "ELECT": (ALU, "alu", "mov"),
    "VOTEU": (UNIFORM, "alu", "mov"), "UP2UR": (UNIFORM, "alu", "mov"), "LEPC": (ALU, "alu", "mov"), "CALL": (BRANCH, "none", None), "BREAK": (BRANCH, "none", None),
    "LDL": (LSU, "local", None), "STL": (LSU, "sts", None),
    "UMOV": (UNIFORM, "alu", "mov"), "ULEA": (UNIFORM, "alu", "mov"), "UIADD3": (UNIFORM, "alu", "mov"), "USHF": (UNIFORM, "alu", "mov"), "UISETP": (UNIFORM, "alu", "mov"),
    "ULOP3": (UNIFORM, "alu", "mov"), "CS2UR": (UNIFORM, "alu", "mov"), "UIMAD": (UNIFORM, "alu", "mov"), "UPRMT": (UNIFORM, "alu", "mov"),
    "UVIMNMX": (UNIFORM, "alu", "mov"), "UPLOP3": (UNIFORM, "alu", "mov"),
    "DFMA": (FMA, "fp64", None), "DMUL": (FMA, "fp64", None), "DADD": (FMA, "fp64", None), "DSETP": (FMA, "fp64", None),
    "PLOP3": (ALU, "alu", "lop3"), "VOTE": (ALU, "alu", "iadd3"), "WARPSYNC": (BRANCH, "none", None), "MEMBAR": (NONE, "none", None), "ERRBAR": (NONE, "none", None),
    "CCTL": (NONE, "none", None), "BPT": (NONE, "none", None), "YIELD": (NONE, "none", None),
}


def lookup(opcode: str) -> tuple[str, str, str | None]:
    base = opcode.split(".")[0]
    if base not in PIPE_TABLE:
        raise UnknownOpcode(f"SASS opcode {opcode!r} is not in the pipe table (agentic/sim/isa.py); add it with its pipe, class and measured rate")
    return PIPE_TABLE[base]


class PipeCosts:
    """Per-warp-instruction pipe costs (cycles on one sub-partition) and scoreboard latencies, from the contract."""
    def __init__(self, contract: dict, schedulers: int = 4):
        hz = float(contract.get("clock_hz_sustained") or 2.4e9)
        self.hz = hz
        # one FFMA per sub-partition per cycle at the B2 rate; other FMA/ALU opcodes by their B21 rate relative to FFMA
        ffma = float(contract["fp32_flops_per_s_per_sm"]) / 2.0        # FMA per second per SM
        self.fma_cost = 1.0 / min(1.0, ffma / 32.0 / hz / schedulers)
        rates = {k: float(v) for k, v in (contract.get("inst_throughput_per_sm") or {}).items()}
        self.rate_cost = {}
        for k, v in rates.items():
            self.rate_cost[k] = self.fma_cost * rates["ffma"] / v if rates.get("ffma") and v else self.fma_cost
        self.rate_cost.setdefault("mov", 1.0)                              # not measured: one issue slot (notes/instruction_stream.md)
        xu = contract.get("sfu_warp_instr_per_cycle_per_smsp")
        self.xu_cost = 1.0 / float(xu) if xu else 8.0
        sh = contract.get("shfl_warp_instr_per_cycle_per_smsp")
        self.shfl_cost = 1.0 / float(sh) if sh else 8.0
        mi = contract.get("mma_issue_cycles") or {}
        self.mma_cost = float(mi.get("bf16_m16n8k16") or next(iter(mi.values()), 16.0))
        self.copy_hold = float(contract.get("cp_async_issue_cycles_per_chunk") or 6.8)
        sb = contract.get("scoreboard_latency_cycles") or {}
        lat = contract.get("mma_latency_cycles") or {}
        dep = contract.get("ldmatrix_mma_dependent_cycles")
        alu = float(sb.get("fma") or 4.2)
        self.latency = {"fma": alu, "alu": alu, "lds": float(sb.get("lds") or 27.0), "sts": 0.0,
                        "ldsm": max(1.0, float(dep) - float(lat.get("bf16_m16n8k16", 30.1)) - alu) if dep else float(contract.get("ldmatrix_issue_cycles") or 12.0),
                        "gload": float(contract["latency_ns"]["l2"]) * 1e-9 * hz, "gstore": 0.0, "mma": float(lat.get("bf16_m16n8k16") or 30.1),
                        "sfu": max(1.0, float((contract.get("sfu_latency_cycles") or {}).get("ex2") or 18.0) - 1.0),
                        "cvt": max(1.0, float((contract.get("sfu_latency_cycles") or {}).get("f2fp_bf16x2") or 10.0) - 1.0),
                        "shfl": max(1.0, float((contract.get("shfl_latency_cycles") or {}).get("bfly") or 28.0) - 1.0), "fp64": 8 * alu, "none": 0.0,
                        # constant-cache loads: no benchmark; the on-SM L1 hit latency (B5) stands in (notes/instruction_stream.md, assumptions)
                        "const": float(contract["latency_ns"]["l1"]) * 1e-9 * hz,
                        "local": float(contract.get("local_mem_latency_ns") or contract["latency_ns"]["l1"]) * 1e-9 * hz,
                        # mbarrier arrive / try_wait: B37's round trip less its reference loop
                        "mbar": float((contract.get("mbarrier_cycles") or {}).get("arrivals=1") or 75.0)}

    def apply(self, ins: Instr) -> Instr:
        pipe, cls, rate = lookup(ins.opcode)
        ins.pipe, ins.sb_class = pipe, cls
        if ins.pred == "@!PT":                                          # never executes: an issue slot and nothing else
            ins.pipe, ins.sb_class, ins.cost, ins.dst = NONE, "none", 1.0, ()
            return ins
        if pipe == FMA:
            ins.cost = self.rate_cost.get(rate or "ffma", self.fma_cost) if cls != "fp64" else 8 * self.fma_cost
        elif pipe == ALU:
            ins.cost = self.rate_cost.get(rate or "iadd3", self.rate_cost.get("iadd3", 2.0))
        elif pipe == XU:
            ins.cost = self.xu_cost
        elif pipe == TENSOR:
            ins.cost = self.mma_cost; ins.hold = self.mma_cost
        elif pipe == LSU and cls == "shfl":
            ins.cost = self.shfl_cost
        elif pipe == COPY:
            ins.cost = 1.0; ins.hold = self.copy_hold
        else:
            ins.cost = 1.0
        if pipe == XU and cls == "sfu":
            ins.hold = self.xu_cost                                     # B38: one warp issues a MUFU every 8.3 cycles
        return ins


# ------------------------------------------------------------------------------------------- the opcode table as a document
SOURCES = {"fma": "rate B21 (relative to FFMA, B2); latency B27", "alu": "rate B21; latency B27", "mov": "not measured: one issue slot",
           "sfu": "rate B38 (pipe counter); latency B38", "cvt": "rate B38 (same pipe as MUFU); latency B38", "mma": "pipe B31; warp hold B34; latency B31",
           "lds": "queue B30; latency B27", "sts": "queue B30", "ldsm": "queue B30; latency B32 (chain)", "gload": "queue B30; latency B5 (L2)",
           "gstore": "queue B30", "shfl": "rate B39 (counter); one LSU wavefront B39; latency B39", "mbar": "latency B37", "local": "latency B19",
           "const": "not measured: L1 hit latency B5 stands in", "fp64": "not measured: 8x FFMA stands in", "none": "issue slot"}


def opcode_table(contract: dict) -> str:
    """calibration/<host>/opcodes.md: every opcode the SASS reader knows, its pipe, its cost per warp instruction on one
    sub-partition, the warp hold, the result latency, and the benchmark each number comes from. Generated, never edited by hand."""
    c = PipeCosts(contract)
    rows = []
    for op in sorted(PIPE_TABLE):
        probe = Instr(opcode=op, pipe="", sb_class="")
        c.apply(probe)
        pipe, cls, rate = PIPE_TABLE[op]
        lat = c.latency.get(cls, 0.0)
        src = SOURCES.get("mov" if rate == "mov" and pipe in (ALU, UNIFORM) else cls, "")
        if pipe in (BRANCH, NONE, UNIFORM, WAIT, BAR, COPY) and cls in ("none", "alu"):
            src = "issue slot" + (" (copy timing: the memory model, B8/B15/B35)" if pipe == COPY else "")
        rows.append(f"| `{op}` | {pipe} | {cls} | {probe.cost:.2f} | {probe.hold:.1f} | {lat:.1f} | {src} |")
    head = ("# Opcodes the SASS reader knows (spark-6951, GB10, sm_121)\n\n"
            "Generated by `python -m agentic.sim.isa` from the pipe table (`agentic/sim/isa.py`) and the measured contract. "
            "Cost: cycles the pipe stays busy for one warp instruction on one sub-partition. Hold: cycles the issuing warp waits after issue. "
            "Latency: cycles until a consumer can read the result. An opcode not in this table stops the reader with an error. "
            "IMAD on the FMA pipe and the integer ops on the ALU pipe follow NVIDIA's profiling guide; every number is ours.\n\n"
            "| opcode | pipe | latency class | cost | hold | latency | source |\n|---|---|---|---|---|---|---|\n")
    return head + "\n".join(rows) + "\n"


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from .config import load
    out = Path(__file__).resolve().parents[2] / "calibration" / "spark-6951" / "opcodes.md"
    out.write_text(opcode_table(load("spark-6951-gb10")["contract"]))
    print("wrote", out, file=sys.stderr)
