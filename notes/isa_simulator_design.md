# The Instruction Front End: Running SASS on the Measured GB10 Model

*Design note, v0.1 · October 15, 2026 · Li Shang, drafted with Claude. For Claude Code. Builds on `mechanistic_simulator_design.md` (the chip model, tag `gb10-model-v1`) and `gb10_microarchitecture.md` (the chip). The instruction-set chapters of the book will be written from the opcode table this note asks for.*

---

## 1. What this adds, in one paragraph

Today the simulator's compute phase takes counts: so many FFMAs, so many shared loads, so many overhead instructions per loop step, charged in a fixed order. That is why it mispredicts the softmax kernel at low occupancy (the compiler interleaves rows; the model runs them one after another) and why B27's one-warp points miss (the compiler pipelines loads across iterations; the model charges each latency in turn). The fix is to feed the simulator the real instruction stream, one instruction at a time, with its dependencies, and let the measured scheduler, scoreboard, queues and pipes decide when each one issues. Nothing on the memory side changes. The same door accepts a stream from NVIDIA's compiler, from our own MIP scheduler, or from a hand-written schedule. The simulator models timing only; what the kernel computes stays with the verifier.

## 2. What the simulator does and does not do with an instruction

It does: decide the cycle the instruction issues, which warp scheduler issued it, what it waited for and how long, which pipe it held and for how many cycles, and for a memory instruction, which queue it entered and when its data came back.

It does not: compute the instruction's result. Registers hold no values in the timing model, only readiness times. Addresses for memory instructions come from the template (which knows the layout) or from the stream itself when the producer supplies them, never from executing address arithmetic. If a branch depends on data, the stream carries the taken path; the template knows its loop trip counts.

That split is the same one the project has used from the start: the verifier owns correctness, the evaluator owns time. The instruction front end is an evaluator feature.

## 3. The instruction stream

One file per kernel, `stream.json`, produced by the SASS reader (§4) or by any other producer. It holds, per warp role (a kernel usually has one role, the tensor kernels may have producer and consumer warps):

```json
{"kernel": "t5_128x128x32", "sm_arch": "sm_121", "clock_source": "contract",
 "roles": [{"role": "main", "warps_per_block": 8, "instructions": [ ... ]}],
 "loops": [{"label": "k_loop", "first": 41, "last": 118, "trips_per_phase": 8}]}
```

Each instruction:

| field | meaning | example |
|---|---|---|
| `i` | index in the stream | 57 |
| `op` | opcode as `cuobjdump` prints it, modifiers kept | `LDS.128`, `HMMA.16816.F32.BF16`, `LDGSTS.E.BYPASS.128` |
| `pipe` | from the pipe table (§5): `fma`, `alu`, `fmaheavy`, `xu`, `tensor`, `lsu`, `branch`, `uniform`, `none` | `lsu` |
| `dst` | registers written | `["R12","R13","R14","R15"]` |
| `src` | registers read | `["R2","R3"]` |
| `pred` | predicate register guarding it, or null | `"P0"` |
| `pred_dst` | predicate written, or null | |
| `mem` | for memory ops: `{"space": "shared"|"global"|"local", "width": 128, "addr": <pattern or per-thread list>, "wavefronts": 4}` | |
| `copy` | for `LDGSTS`/TMA: `{"group": 3, "bytes": 16, "slot": 1}` | |
| `bar` | for `BAR`/`DEPBAR`/mbarrier: `{"id": 0, "count": 256}` or `{"wait_group": 1}` | |
| `uniform` | true for the uniform datapath (`UIADD3`, `UMOV`, `ULDC`) | |

Fields the reader cannot fill are absent, never guessed. A stream with an opcode missing from the pipe table is rejected with the opcode named.

**Addresses.** The template is the authority: for every `LDS`/`STS`/`LDSM` in the loop, the template's `events.py` already computes the per-thread address pattern and its wavefront count (round G7 of the GEMM work, within 10 % of the counter on 5 of 6 kernels). The reader matches each memory instruction in the SASS to the template's access list by order within the loop body and width. When it cannot match (the compiler merged or dropped a load, as it did in the 32×128×32 tile), it marks the instruction `addr: unmatched` and the simulator charges the width-only wavefront count and logs it. The mismatch count is reported per kernel.

## 4. The SASS reader

`agentic/sim/sass.py`. Input: the compiled cubin of the template's kernel, which the evaluator already produces. Steps:

1. `cuobjdump -sass` and parse: one instruction per line, keep the address, predicate, opcode with modifiers, operands as printed. Skip the control-word lines (`/* 0x... */`); they encode the compiler's stall hints and barriers, which are not public, and the model derives dependencies itself (§6).
2. Find loops: back-edge `BRA` targets inside the kernel body. The inner loop is the one whose body contains the `HMMA`/`FFMA` block. Its trip count per phase comes from the template (TK / k-step), not from the SASS.
3. Classify each instruction by the pipe table.
4. Attach addresses from the template (§3).
5. Emit `stream.json`, and a one-line summary: instruction count per loop iteration, per phase outside the loop, and the opcode histogram. The counts must match what G7's count-based path already extracts, within 2 %; that is the reader's first test.

Register operand syntax to handle: `R12`, `R12.reuse`, `-R12`, `|R12|`, `R12.H0_H0`, register ranges written as the first register (`LDS.128 R12` writes R12–R15: the width decides the range), `UR4` (uniform), `P0`/`PT`, `c[0x0][0x160]` (constant bank, no dependency), `SR_TID.X` (special, no dependency), `[R2+0x100]` (address uses R2), `desc[UR4][R2.64]` (global descriptor form on Blackwell: uses UR4 and R2–R3).

## 5. The pipe table

`agentic/sim/isa.py` holds a table keyed by opcode root (`FFMA`, `HMMA`, `LDS`, …) and modifiers where they matter (`LDS.128` vs `LDS.32`, `HMMA.16816` vs `.1688`). Each entry:

| field | source |
|---|---|
| `pipe` | the functional unit it occupies |
| `issue_cycles` | cycles the sub-partition dispatch is held (B30 for `LDS`: 4.2; B31 for `HMMA`: 16; B2/B21 for FP32 and integer; B38 SFU 8; B39 `SHFL` 8) |
| `latency_cycles` | producer-to-consumer (FFMA 4, IMAD 4, `LDS` 27, L1-hit load 44, `HMMA` 30, `MUFU` 18, `SHFL` 27, `LDSM` 47) |
| `lsu` | for memory ops: wavefront rule (width and address pattern, B26.6/B30) |
| `scoreboard` | `short` (fixed latency: math) or `long` (variable: memory, texture) — the two classes `ncu` counts separately |
| `benchmark` | where the numbers came from |

Every entry carries its benchmark. An opcode with no benchmark is not in the table; the stream fails loudly. The first table covers the opcodes in T5, the 16×32×8 tile, the BF16 tensor tile and the softmax kernel (about thirty), and `calibration/spark-6951/opcodes.md` plus B41's `opcode_table` in the contract are its source.

Opcode groups the four kernels contain, from the SASS mixes already in the contract: FP32 math (`FFMA`, `FADD`, `FMUL`, `FSETP`, `FSEL`, `MUFU`, `F2I`, `I2F`, `I2FP`, `F2FP`), integer (`IMAD`, `IADD3`, `LEA`, `LOP3`, `SHF`, `ISETP`, `SEL`, `PRMT`, `VIMNMX`, `MOV`, `CS2R`, `S2R`), half (`HFMA2`, used by the compiler for packed moves), uniform (`UIADD3`, `UMOV`, `UISETP`, `ULDC`, `CS2UR`), memory (`LDS`, `STS`, `LDSM`, `LDGSTS`, `LDG`, `STG`, `LDC`, `LDCU`, `LDL`/`STL` when spilling), tensor (`HMMA`), sync (`BAR`, `DEPBAR`, `BSSY`/`BSYNC`, `WARPSYNC`), control (`BRA`, `EXIT`, `CALL`/`RET`, `NOP`), and for softmax `SHFL`, `REDUX` if present.

## 6. Dependencies, built from the text

No register renaming on a GPU: the register in the SASS is the physical one. So:

- **Register RAW.** Instruction B reads R5; the latest earlier instruction A in program order that writes R5 is its producer. B cannot issue before A's issue cycle + A's `latency_cycles`. A `.reuse` flag changes nothing in the model.
- **WAR and WAW.** In-order issue per warp makes these free unless a long-latency producer is still outstanding: a write to R5 while an earlier `LDS` into R5 is in flight waits for that `LDS`. Modeled: a register is busy from a long-scoreboard producer's issue until its latency elapses; a later writer waits.
- **Predicates.** `P0` written by `ISETP`, read by a guarded instruction: a RAW like any register, latency from the table. A guarded-off instruction still occupies issue (the hardware issues it as a no-op) unless the predicate is uniformly false, which the model assumes only for `PT`-style constants.
- **Memory order.** Shared-memory RAW between warps is enforced only by `BAR` and by `cp.async` waits; within a warp, `STS` then `LDS` to the same slot is a `long` dependency through the LSU, modeled as the `LDS` waiting for the `STS` to leave the LSU. This is the same slot model the verifier uses.
- **Copies.** `LDGSTS` chunks belong to a commit group; `DEPBAR.LE SB0, n` (`cp.async.wait_group n`) waits until at most n groups are pending; group completion time comes from the existing copy mechanism (M4, with its measured latency distribution). TMA: `UTMALDG` issues, the mbarrier `SYNCS` wait completes it, through the TMA engine of round 2.
- **Barriers.** `BAR.SYNC` releases when all warps of the block arrive, plus `barrier_s` (B7). `BSSY`/`BSYNC` are reconvergence and cost issue only.
- **Control.** `BRA` on the loop back-edge: the model unrolls by the trip count from the template; divergent branches inside the body are not expected in these kernels and are an error.

The compiler's control words encode the same dependencies as stall counts and scoreboard indices. The model does not read them. Where the compiler chose a longer stall than the dependency needs, the card is slower than the model; that shows as a small positive error and is logged, not fitted.

## 7. Issue: what changes in `sim/warp.py`

The warp model already has the four schedulers, the per-warp scoreboard, the LSU queue with its dispatch slot and drain, the FMA and tensor pipes, and the per-SM copy window. Today it is fed a per-step template of counts. With `input: stream`:

- Each warp holds a program counter into its role's stream and a scoreboard: per register, the cycle it becomes ready; per pipe, the cycle it is free.
- Each cycle, each scheduler looks at its warps in age order (oldest eligible first; B27's TLP curve and gap 27's `mma`-vs-FFMA arbitration are the checks of this rule) and issues the first whose next instruction has all sources ready, its pipe free, and its issue slot free. Memory instructions also need the LSU dispatch slot (4.2 cycles per sub-partition, B30).
- The issued instruction sets its destination registers' ready cycle to now + latency, holds its pipe for `issue_cycles`, and for memory ops enters the LSU with its wavefront count.
- Loop back-edges advance the trip counter; phases advance when the stream's phase marker (the `DEPBAR`/`BAR` pair the template identifies) passes.

The count-based path stays behind the switch so both can be run on the same design and compared. Cost: a few thousand instructions per phase per warp, integer arrays, no per-instruction objects; expect 10× the current compute-phase time, still seconds per design. The per-(design, residency) phase cost is cached as before.

## 8. The other producers

The door is the stream format, not the SASS reader. Two more producers, not built this round, are the reason the format exists:

- **The MIP scheduler** (paper §3.1 level 5; lecture Part 3). It takes the template's dataflow (the same per-step operations the count-based path knows), the pipe table as its cost coefficients (`latency_cycles`, `issue_cycles`, pipe capacity, register budget), and emits a `stream.json` with its own instruction order and register assignment. The simulator runs it like any other stream. Its obligations are the verifier's: every read after its write (lifetime), no two live values in one register, every operation exactly once.
- **A hypothetical ISA.** A new instruction is a new row in the pipe table with a benchmark-less latency marked `hypothetical`; the console labels every number resting on it as simulated. This is the only way to evaluate a chip whose instruction set NVIDIA's compiler cannot target.

## 9. Tests, in order

| # | Test | Pass |
|---|---|---|
| S1 | Reader counts vs the count-based extraction on the six B29 kernels | within 2 %, and the matched-address fraction ≥ 90 % on five of six |
| S2 | Dependency graph sanity: every `src` has a producer or is an input; no register read before any write in the loop body | zero violations; unmatched reads listed |
| S3 | Pure FFMA chain stream (B2's kernel) and pure `LDS` chain (B17's) | reproduce B2 and B17 within 2 % and 5 % |
| S4 | B27's dependency grid, now as a stream | the (a, w, L) grid within 10 %, including the one-warp points the count path misses |
| S5 | The six B29 kernels, stream vs card | median within 8 %, none worse than the count path by more than 3 points |
| S6 | Softmax composite C3 at 1, 2, 4, 6 blocks | within 15 % at every residency (was +67 % at one block) |
| S7 | The thirty GEMM designs, both paths | stream path median ≤ count path median; the per-design table reported |
| S8 | A hand-written stream with one load moved earlier | the stall disappears in the trace by the expected cycles; this is the lecture's demonstration |

A miss in S4–S7 is a gap-log entry with the counter that shows why. No parameter is changed to pass a test.

## 10. Build order

| Stage | Deliverable |
|---|---|
| I0 | `isa.py` pipe table from `opcodes.md` and B41; `notes/instruction_stream.md` (the format, one page, the table in §3) |
| I1 | `sass.py` reader; S1, S2 |
| I2 | `warp.py` stream input behind the switch; S3, S4 |
| I3 | S5–S7; gap-log entries; the console's inspector shows, for a selected cycle, the instruction each warp is on and what it waits for |
| I4 | S8 and the hand-written-stream path; a `stream.json` example checked into `notes/` for the lecture |

## 11. Instruction for Claude Code

> Read `notes/isa_simulator_design.md`. Build I0 through I3 in order; I4 after. The pipe table comes from `calibration/spark-6951/opcodes.md` and B41's `opcode_table`; an opcode without a benchmark is an error, not a guess. The count-based path stays behind `input: counts | stream`. Run S1 to S7 and report one table per test, plain language, with the per-design table for S7 showing measured, counts model, stream model and both errors. Nothing fitted. Stop and report if S4 or S6 misses; do not tune.
