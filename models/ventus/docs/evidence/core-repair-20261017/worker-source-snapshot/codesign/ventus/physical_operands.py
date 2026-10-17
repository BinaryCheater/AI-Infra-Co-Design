"""Physical collector slots for the bounded concrete FP32 GEMM emitter.

This decoder accepts only encodings emitted by packed_gemm/complete_gemm. It is
not a general Ventus ISA decoder. Slot identity matters to bank RR arbitration,
and an architectural x0 operand still consumes an RF read port in this RTL.
"""

from dataclasses import replace


def with_physical_operands(workload, words, *, base_pc=0x80000000):
    operations = []
    for op in workload.operations:
        pc = int(op.name.rsplit("pc", 1)[1], 16)
        word = words[(pc - base_pc) // 4]
        opcode, mode = word & 0x7F, (word >> 12) & 7
        rs1, rs2, rd = (word >> 15) & 31, (word >> 20) & 31, (word >> 7) & 31
        if op.kind == "endprg" and word == 0x400B:
            physical = []
        elif opcode == 0x37:
            physical = []
        elif opcode in {0x13, 0x73}:
            physical = [(0, f"x{rs1}")]
        elif opcode == 0x7B and op.kind in {"load", "store"}:
            physical = [(0, f"v{rs1}")]
            if op.kind == "store":
                physical.append((2, f"v{rs2}"))
        elif opcode == 0x0B and op.kind == "tensor":
            physical = [(0, f"v{rs1}"), (1, f"v{rs2}"), (2, f"v{rd}")]
        elif opcode == 0x57 and (word >> 25) & 1:
            if (word >> 26) == 0x14 and mode == 2 and rs1 == 17:  # VID
                physical = []
            elif mode == 0:
                physical = [(0, f"v{rs1}"), (1, f"v{rs2}")]
            elif mode == 3:
                physical = [(1, f"v{rs2}")]
            elif mode == 4:
                physical = [(0, f"x{rs1}"), (1, f"v{rs2}")]
            else:
                raise ValueError(f"unsupported bounded GEMM operand encoding at {op.name}")
        else:
            raise ValueError(f"unsupported bounded GEMM operand encoding at {op.name}")
        if sorted(reg for _, reg in physical if reg != "x0") != sorted(
            reg for reg in op.sources if reg != "x0"
        ):
            raise ValueError(f"machine encoding and IR sources disagree at {op.name}")
        operations.append(
            replace(
                op,
                sources=tuple(reg for _, reg in physical),
                source_slots=tuple(slot for slot, _ in physical),
            )
        )
    return replace(workload, operations=tuple(operations))
