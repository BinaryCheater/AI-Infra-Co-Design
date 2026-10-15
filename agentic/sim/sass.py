"""The SASS reader (notes/instruction_stream.md): `cuobjdump -sass` text -> the instruction stream of a kernel's inner loop per warp.

Reads the dump (cached under calibration/<host>/sass/<key>.sass; `cuobjdump` runs on the GPU host through the ssh alias when a
dump is missing), finds the loops by backward branches, picks the inner loop (the one with the most FFMA, HMMA or MUFU
instructions, as asked), and emits the stream: one `isa.Instr` per SASS instruction with registers widened by the opcode's width,
predicates, memory widths, copy and barrier ids. For a tiled-GEMM phase the stream is the phase body in address order with each
loop expanded by its trip count (the k loop by TK / unroll, the copy loops by their chunks per thread). An opcode the pipe table
does not know raises `UnknownOpcode`.
"""
from __future__ import annotations

import copy
import re
import subprocess
from pathlib import Path

from .isa import Instr, PipeCosts, lookup, UnknownOpcode, COPY, WAIT, BAR, LSU

ROOT = Path(__file__).resolve().parents[2]
SASS_DIR = ROOT / "calibration" / "spark-6951" / "sass"
LINE = re.compile(r"/\*([0-9a-f]+)\*/\s+(?:(@!?U?P[0-9T]+)\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);")
REG = re.compile(r"(?<![A-Za-z_])(-?\|?)(U?R\d+|U?P\d+|RZ|URZ|PT|SB\d)(?:\.64|\.H[01]|\.reuse|\.U32|\.S32|\.F32|\.X)?")


def read_dump(key: str, host: str | None = None, binary: str | None = None) -> str:
    f = SASS_DIR / f"{key}.sass"
    if f.exists():
        return f.read_text()
    if not host or not binary:
        raise FileNotFoundError(f"no SASS dump {f} and no host/binary to produce it")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"export PATH=$PATH:/usr/local/cuda/bin; cuobjdump -sass {binary}"], capture_output=True, text=True, check=True)
    SASS_DIR.mkdir(parents=True, exist_ok=True); f.write_text(r.stdout)
    return r.stdout


def function_text(dump: str, name_substr: str) -> str:
    parts = dump.split("Function : ")
    for p in parts[1:]:
        if name_substr in p.split("\n", 1)[0]:
            return p
    raise KeyError(f"no function containing {name_substr!r} in the dump; functions: {[p.split(chr(10),1)[0].strip() for p in parts[1:]]}")


def _width(opcode: str, default: int = 4) -> int:
    for w in (".128", ".64", ".32", ".16", ".8"):
        if w in opcode:
            return {".128": 16, ".64": 8, ".32": 4, ".16": 2, ".8": 1}[w]
    return default


def _widen(reg: str, n: int) -> tuple:
    if n <= 1 or not re.fullmatch(r"R\d+", reg):
        return (reg,)
    base = int(reg[1:])
    return tuple(f"R{base + i}" for i in range(n))


def parse_instruction(addr: int, pred: str | None, opcode: str, operands: str) -> Instr:
    """Registers read and written by one SASS instruction, widened by the opcode's width; memory and barrier fields."""
    base = opcode.split(".")[0]
    pipe, cls, _ = lookup(opcode)
    ops = [o.strip() for o in operands.split(",")] if operands.strip() else []
    regs_of = lambda s: [m.group(2) for m in REG.finditer(s) if m.group(2) not in ("RZ", "URZ", "PT")]
    dst, src = [], []
    ins = Instr(opcode=opcode, pipe=pipe, sb_class=cls, pred=pred, addr=addr)
    if base in ("STG", "STS", "LDGSTS", "ATOMS"):
        # stores: every register is read (address and data); atomics write the first register
        src = regs_of(operands)
        if base == "ATOMS" and ops:
            dst = regs_of(ops[0]); src = regs_of(",".join(ops[1:]))
        ins.width_bytes = _width(opcode, 4) if base != "LDGSTS" else 16
        if base == "LDGSTS":
            ins.chunk_ids = (addr,)
    elif base in ("BAR",):
        ins.barrier = operands.strip() or "0"
    elif base == "DEPBAR":
        m = re.search(r"SB(\d).*?0x([0-9a-f]+)", operands)
        ins.count = int(m.group(2), 16) if m else 0
    elif base in ("BRA", "BSSY", "BSYNC", "EXIT", "RET", "NOP", "LDGDEPBAR", "WARPSYNC", "MEMBAR", "ERRBAR", "CCTL", "BPT", "YIELD"):
        src = regs_of(operands)
    elif base == "ISETP" or base == "FSETP" or base == "DSETP" or base == "UISETP" or base == "PLOP3":
        # P0, P1 (dst predicates) then sources
        pr = [o for o in ops if re.fullmatch(r"!?U?P[0-9T]", o)]
        n_dst = 2 if len(pr) >= 2 and ops and ops[:2] == pr[:2] else 1
        dst = [p.lstrip("!") for p in ops[:n_dst] if p.lstrip("!") != "PT"]; src = regs_of(",".join(ops[n_dst:]))
    elif base == "SHFL":
        # SHFL.BFLY PT, Rd, Rs, lane, mask
        dst = [o for o in ops[:2] if o != "PT"]; src = regs_of(",".join(ops[2:]))
    elif base in ("HMMA", "IMMA", "QMMA"):
        d = regs_of(ops[0])[:1]; dst = list(_widen(d[0], 4)) if d else []
        a = regs_of(ops[1])[:1]; b = regs_of(ops[2])[:1]; c = regs_of(ops[3])[:1] if len(ops) > 3 else []
        src = (list(_widen(a[0], 4)) if a else []) + (list(_widen(b[0], 2)) if b else []) + (list(_widen(c[0], 4)) if c else [])
    elif base == "LDSM":
        d = regs_of(ops[0])[:1]; n = 4 if ".X4" in opcode else (2 if ".X2" in opcode else 1)
        dst = list(_widen(d[0], n)) if d else []; src = regs_of(",".join(ops[1:])); ins.width_bytes = 16
    elif base in ("LDS", "LDG", "LDC", "LDCU"):
        d = regs_of(ops[0])[:1]; w = _width(opcode, 4); n = max(1, w // 4)
        dst = list(_widen(d[0], n)) if d else []; src = regs_of(",".join(ops[1:])); ins.width_bytes = w
    else:
        # dst first, then sources; IMAD.WIDE writes a pair
        if ops:
            d = regs_of(ops[0]); dst = list(_widen(d[0], 2)) if d and ".WIDE" in opcode else d[:1]
            src = regs_of(",".join(ops[1:]))
    if pred:
        src = src + [pred.lstrip("@!")]
    ins.dst, ins.src = tuple(dst), tuple(src)
    return ins


def parse_function(text: str) -> list[Instr]:
    out = []
    for m in LINE.finditer(text):
        addr, pred, opcode, operands = int(m.group(1), 16), m.group(2), m.group(3), m.group(4)
        out.append(parse_instruction(addr, pred, opcode, operands))
    return out


def _target(ins: Instr) -> int | None:
    m = re.search(r"0x([0-9a-f]+)\s*$", getattr(ins, "_operands", "").strip())
    return int(m.group(1), 16) if m else None


def _conditional(ins: Instr) -> bool:
    ops = getattr(ins, "_operands", "")
    return (ins.pred not in (None, "@PT")) or bool(re.match(r"\s*!?U?P\d", ops))


class Function:
    """One SASS function: its instructions, its loops (backward branches) and a walker that flattens a region into the dynamic
    stream. The branch rule (notes/instruction_stream.md): a backward branch closes a loop, and the loop is expanded by its trip
    count; an unconditional forward branch is followed; a conditional forward branch falls through unless the code it skips holds
    a CALL, EXIT or BPT (a fallback or error path), in which case it is taken."""
    def __init__(self, text: str):
        self.instrs = []
        for m in LINE.finditer(text):
            ins = parse_instruction(int(m.group(1), 16), m.group(2), m.group(3), m.group(4)); ins._operands = m.group(4); self.instrs.append(ins)
        self.index = {i.addr: n for n, i in enumerate(self.instrs)}
        cand = []
        for i in self.instrs:
            if i.opcode.startswith("BRA"):
                t = _target(i)
                if t is not None and t < i.addr:
                    cand.append((t, i.addr))
        # an out-of-line slow path (the mbarrier retry spin the compiler places after the kernel's exit) branches back into a loop
        # body; it shows up as a "loop" overlapping a real one without nesting. Of two such, the one whose branch comes later is not a loop.
        def overlap(a, b):
            return a[0] < b[0] <= a[1] < b[1]
        self.out_of_line = [b for b in cand if any(overlap(a, b) for a in cand if a != b)]
        self.loops = [r for r in cand if r not in self.out_of_line]

    def body(self, lo: int, hi: int) -> list[Instr]:
        return [i for i in self.instrs if lo <= i.addr <= hi]

    def count(self, r, ops) -> int:
        return sum(1 for i in self.body(*r) if i.opcode.split(".")[0] in ops)

    def inner_loop(self, key=("FFMA", "HMMA", "MUFU"), exclude=("BAR",)) -> tuple[int, int] | None:
        """The loop with the most instructions of the given kinds and none of `exclude`; ties go to the innermost."""
        cands = [r for r in self.loops if self.count(r, key) and not self.count(r, exclude)]
        if not cands:
            return None
        return max(cands, key=lambda r: (self.count(r, key), -(r[1] - r[0])))

    def enclosing(self, r) -> tuple[int, int] | None:
        outer = [o for o in self.loops if o != r and o[0] <= r[0] and r[1] <= o[1]]
        return min(outer, key=lambda o: o[1] - o[0]) if outer else None

    def walk(self, lo: int, hi: int, trips: dict, costs: PipeCosts, _skip_loop=None, take: dict | None = None) -> list[Instr]:
        out = []
        n = self.index.get(lo)
        if n is None:
            n = next(k for k, i in enumerate(self.instrs) if i.addr >= lo)
        while n < len(self.instrs) and self.instrs[n].addr <= hi:
            ins = self.instrs[n]
            nested = [r for r in self.loops if r[0] == ins.addr and r != _skip_loop and lo <= r[0] and r[1] <= hi and (r[0], r[1]) != (lo, hi)]
            if nested:
                r = max(nested, key=lambda r: r[1] - r[0])
                body = self.walk(r[0], r[1], trips, costs, _skip_loop=r, take=take)
                out += body * max(1, int(trips.get(r, 1)))
                n = self.index[r[1]] + 1
                continue
            if ins.opcode.startswith("BRA"):
                t = _target(ins)
                if t is not None and t > hi:
                    # a branch out of the walked region: to an out-of-line slow path (conditional: not taken on the common path) or out
                    # of the loop (unconditional: the common path stays in the body); either way the walk continues in order
                    out.append(costs.apply(copy.copy(ins))); n += 1; continue
                if t is not None and t > ins.addr:
                    if not _conditional(ins):
                        out.append(costs.apply(copy.copy(ins))); n = self.index.get(t, n + 1); continue
                    skipped = [i for i in self.instrs if ins.addr < i.addr < t]
                    if (take or {}).get(ins.addr) or any(i.opcode.split(".")[0] in ("CALL", "EXIT", "BPT") for i in skipped):
                        out.append(costs.apply(copy.copy(ins))); n = self.index.get(t, n + 1); continue
            out.append(costs.apply(copy.copy(ins)))
            n += 1
        return out

    def loop_stream(self, costs: PipeCosts, key=("FFMA", "HMMA", "MUFU")) -> list[Instr]:
        r = self.inner_loop(key)
        return self.walk(r[0], r[1], {}, costs, _skip_loop=r)


def gemm_phase_stream(key: str, tm: int, tn: int, tk: int, costs: PipeCosts, threads: int = 256) -> tuple[list[list[Instr]], dict]:
    """The generated tiled-GEMM kernel's phase as a per-warp stream: the smallest loop around the k loop, the k loop expanded
    TK / unroll times, each copy loop by its chunks per thread; the k loop's shared loads labeled A (LDS.128) and B (32-bit) in
    order, so the template's address-derived wavefronts can be attached at run time."""
    F = Function(function_text(read_dump(key), "smem_tiling_gemm"))
    k = F.inner_loop(("FFMA",), exclude=("BAR", "LDGSTS"))
    if k is not None:
        phase = F.enclosing(k)
    else:                                                       # the k loop was unrolled into the phase loop
        cands = [r for r in F.loops if F.count(r, ("FFMA",)) and F.count(r, ("BAR",))]
        phase = min(cands, key=lambda r: r[1] - r[0]); k = None
    rm, rn = tm // 16, tn // 16
    region = k or phase
    ffma = F.count(region, ("FFMA",))
    unroll = max(1, round(ffma / max(1, rm * rn))) if k else tk
    k_trips = max(1, -(-tk // unroll))
    copy_loops = sorted(r for r in F.loops if phase[0] <= r[0] and r[1] <= phase[1] and r != phase and r != k and F.count(r, ("LDGSTS", "LDG")))
    copy_loops = [r for r in copy_loops if not any(o != r and o[0] <= r[0] and r[1] <= o[1] for o in copy_loops)]
    chunks = {"A": tm * tk // 4, "B": tk * tn // 4}
    trips = {}
    if k:
        trips[k] = k_trips
    if len(copy_loops) == 2:
        owners = ["A", "B"]
    else:
        owners = [max(chunks, key=chunks.get)] * len(copy_loops)
    for r, own in zip(copy_loops, owners):
        per_thread = -(-chunks[own] // threads); per_body = max(1, F.count(r, ("LDGSTS", "LDG")))
        trips[r] = max(1, -(-per_thread // per_body))
    # label the k loop's shared loads in order (A = 128-bit, B = 32-bit), on the originals, so every expanded copy carries it
    ia = ib = 0
    for i in F.body(*region):
        if i.opcode.split(".")[0] == "LDS" and i.pred != "@!PT":
            if ".128" in i.opcode:
                i.pattern, i.tidx = "template:A", ia; ia += 1
            elif not any(w in i.opcode for w in (".64", ".U8", ".S8", ".U16", ".S16")):
                i.pattern, i.tidx = "template:B", ib; ib += 1
    # straight-line copy code (an operand with no more chunks than threads): a conditional forward branch skips a region holding
    # LDGSTS and no loop; only the first ceil(chunks / 32) warps run it (one chunk per lane), the others take the branch
    loop_ops = owners[:len(copy_loops)]
    straight_ops = [o for o in ("A", "B") if o not in loop_ops]
    guards = []
    for i in F.body(*phase):
        if i.opcode.startswith("BRA") and _conditional(i):
            t = _target(i)
            if t is None or t <= i.addr or any(r[0] <= i.addr <= r[1] for r in copy_loops) or (k and k[0] <= i.addr <= k[1]):
                continue
            skipped = [j for j in F.instrs if i.addr < j.addr < t]
            if any(j.opcode.startswith("LDGSTS") for j in skipped) and not any(j.opcode.startswith("BRA") and (_target(j) or 1 << 60) < j.addr for j in skipped) \
                    and not any(j.opcode.split(".")[0] in ("CALL", "EXIT", "BPT") for j in skipped):
                guards.append(i.addr)
    warps = threads // 32
    streams = []
    for w in range(warps):
        take = {g: w >= -(-chunks[op] // 32) for g, op in zip(guards, straight_ops)}
        streams.append(F.walk(phase[0], phase[1], trips, costs, take=take))
    stream = streams[0]
    return streams, {"phase": [hex(phase[0]), hex(phase[1])], "k_loop": [hex(k[0]), hex(k[1])] if k else None, "unroll": unroll, "k_trips": k_trips,
                    "copy_loops": {hex(r[0]): trips[r] for r in copy_loops}, "straight_copy_guards": [hex(g) for g in guards],
                    "instructions": len(stream), "instructions_mean_per_warp": sum(len(x) for x in streams) / len(streams)}


def summarize(stream: list[Instr]) -> dict:
    h = {}
    for i in stream:
        k = i.opcode.split(".")[0]; h[k] = h.get(k, 0) + 1
    return {"instructions": len(stream), "by_opcode": dict(sorted(h.items(), key=lambda kv: -kv[1]))}
