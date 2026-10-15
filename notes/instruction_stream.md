# The instruction stream

*How the simulator takes a kernel's real instructions as input. Code: `agentic/sim/isa.py` (the instruction and the pipe table), `agentic/sim/sass.py` (the SASS reader), `WarpModel.run_stream` in `agentic/sim/warp.py` (the front end). Switch: `input: counts | stream` in `agentic/sim/config/spark-6951-gb10.yaml`.*

## Why

The warp model used to build each loop from counts: so many FFMAs, so many 32-bit and 128-bit loads per k-step, in an order we wrote by hand. That order is a guess about the compiler. The softmax rows showed the cost: the compiler puts two rows in each loop pass, so the second row's load is in flight while the first row reduces. The hand-built order runs the rows one after another, and missed by 60 % at low occupancy. The stream input reads the order from the compiled kernel instead.

## One instruction

An `Instr` holds what the front end needs and nothing else:

- **opcode** with modifiers, as SASS prints it (`LDS.128`, `HMMA.16816.F32.BF16`).
- **pipe** and **scoreboard class**, from the pipe table.
- **dst** and **src** registers. Wide operations are widened: `LDS.128 R4` writes R4 to R7; `HMMA` reads four A, two B and four C registers. Predicates count as registers (`ISETP` writes P0, a branch reads it).
- **pred**, the guard. `@!PT` never runs, so it takes an issue slot and nothing else.
- For memory operations, **width** and **pattern**. In a tiled-GEMM k loop the shared loads are labeled A (128-bit) or B (32-bit) in order, and take the wavefront count the template computes from their real addresses. Other loads use the width rule: 4 B per lane is one wavefront, 8 B two, 16 B four.
- For `cp.async`, the **chunk** and its copy **group**; for `DEPBAR`, how many groups may stay pending; for `BAR`, the barrier.
- **cost** (cycles the pipe stays busy for one warp instruction) and **hold** (cycles the warp itself waits after issue).

## The pipe table

Each opcode maps to a pipe and a latency class. Every number is from the contract.

| pipe | opcodes | cost per warp instruction | latency |
|---|---|---|---|
| FMA | FFMA, FADD, FMUL, HFMA2, IMAD, FMNMX, FSETP, FSEL | 1.07 cycles for FFMA; others by their B21 rate (IMAD 1.88, FADD 1.13) | 4.2 [B27] |
| ALU | IADD3, LOP3, SHF, ISETP, LEA, SEL, MOV, S2R | B21 rate (IADD3 3.9, LOP3 2.8, SHF 2.3); MOV 1 | 4.2 |
| XU | MUFU, F2FP, F2F, I2F | 8 [B38 counter] | 17 (ex2), 9.5 (convert) [B38] |
| tensor | HMMA, IMMA, QMMA | 16.1, and the warp waits 16.1 [B31, B34] | 30.1 [B31] |
| LSU queue | LDS, STS, LDSM, LDG, STG, SHFL, SYNCS, LDL | one dispatch per 4.2 cycles per sub-partition, drain one wavefront per cycle per SM [B30]; SHFL also one per 8 [B39] | LDS 27, LDSM 47, LDG 336 (L2), SHFL 27, mbarrier 75 |
| copy | LDGSTS, UTMALDG | an issue slot; the copy engines time the data | — |
| none | branches, uniform-datapath ops, NOP, BAR, DEPBAR | an issue slot | 4.2 |

IMAD on the FMA pipe and the rest of the integer ops on the ALU pipe follow NVIDIA's profiling guide; the rates are ours. Three entries have no benchmark yet and are stated, not hidden: MOV costs one ALU cycle, constant-cache loads (LDC) take the L1 hit latency [B5], and FP compares (FMNMX, FSETP) run at the FADD rate. An opcode missing from the table stops the reader with an error.

## Reading SASS

`sass.py` reads `cuobjdump -sass` (cached under `calibration/spark-6951/sass/`), finds loops by backward branches, and flattens a region into the order a warp runs it:

- A loop is expanded by its trip count. In the GEMM kernel the k loop runs TK / unroll times, where unroll comes from the FFMA count of its body; each copy loop runs its chunks per thread.
- An unconditional forward branch is followed. A conditional forward branch falls through, unless the code it skips holds a `CALL`, `EXIT` or `BPT`; that is a fallback or error path, and the branch is taken. (C2's tensor-map code has such a fallback.) A branch out of the region being walked is not followed.
- The compiler puts the mbarrier retry spin after the kernel's exit and branches back into the loop. That makes a backward branch that overlaps a real loop without nesting in it; such a branch is not a loop.
- The trip count of a loop comes from what its body holds: the C1 no-copy kernel's step loop carries four k-steps per pass, so 16 steps are 4 passes.
- Copy code with no loop around it runs only in the warps that own a chunk: the first ceil(chunks / 32). The other warps take the branch around it. So streams differ per warp.

## The front end

Each warp issues in order. An instruction issues when the registers it reads and writes are ready, its pipe on that sub-partition is free, and the scheduler picks it: one instruction per sub-partition per cycle, oldest warp first. Loads go through the same queue and drain as before. Two modes: inside the event simulator ("phase"), barriers and copy waits are issue slots and the event level does the waiting; for a whole kernel ("kernel", used for C1 and C3) the front end keeps barriers and copy groups itself, with the stage latency and chip copy queue of the existing memory model. The memory side is unchanged.

## Limits

The stream is one path through the code: the common path of each branch, the same for every row and every tile. The last k-step of a pass takes the same path as the others, so it issues one copy stage too many per pass (17 for 16 in C1). It does not model the compiler's register reuse beyond what the register names say, the instruction cache, or branch costs beyond one issue slot.
