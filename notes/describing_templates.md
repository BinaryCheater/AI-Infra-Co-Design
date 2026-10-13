# Describing Templates for Three Readers: Console Round 3

**Design document, v0.1 · October 14, 2026**
*Li Shang, drafted with Claude. Follows `architect_console_design.md` (round 2). Written for Claude Code to execute.*

---

## 0. The problem

Round 2 made the flow followable: a fixed map, one sentence per iteration, a pool of cards. But the cards and the inspector still describe each template in developer shorthand. A card reads *"drop beta from the precondition (P <= S-1)"* or *"grid = floor(M/Tm) x floor(N/Tn)"*. That text is the `refines` string the scripted formulator wrote for itself. It tells nobody outside the code what the template decides, what it assumes, or why it was rejected.

The console has three kinds of reader, and each asks a different first question:

| Reader | First question about a template | Wants to see |
|---|---|---|
| **AI infrastructure architect** | What hardware/software decision is this, and what did it cost? | tiles, buffers, barriers, memory, latency; what changed vs the last design |
| **Applied mathematician** | What is the optimization problem? | variables and domains, objective, constraints, problem size, solver status, bound and gap |
| **Formal verification expert** | What was proved, in what scope, and what failed? | obligations, proof status, scope of the family proof vs the instance check, the counterexample |

**The design idea:** every artifact (template, design, verdict, measurement) gets **three descriptions, generated from the same structured fields**, plus **one shared headline** all three readers see first. The reader picks a lens; the facts do not change, only the vocabulary and the emphasis.

This is possible without new data. The contract already carries, for every decision, constraint, precondition and obligation, a `meaning` field; the cost model names its terms; the certificate schema states the relation and the rank in words. The round-3 work is to render those fields, not to add them.

---

## 1. The headline

One line, shared by all three lenses, shown on the pool card and at the top of the inspector:

> **T5 · Tiling the ragged 1000³ GEMM through shared memory · accepted · best so far, 53.1 µs**

Composed as: `{short name} · {what it decides, in systems words} {workload, display name} · {status} · {outcome in one number}`. The "what it decides" phrase comes from a small table keyed by `contract.template` (one entry per template backend; §4), never from `refines`.

Below the headline, one line that says what is *different* about this template compared to its parent, because that is what an architect scanning the pool wants: "Same as T4, but the trailing barrier may be dropped when a slot is free." This is derived by diffing the contract against its parent (§4.3): precondition mode, structural choices, workload.

---

## 2. The three lenses, by example

Two templates from the demo run, rendered three ways. These are the acceptance targets.

### 2.1 T5: accepted

**Systems lens (architect)**

> This template tiles the 1000³ matrix multiply through on-chip shared memory. It decides the output tile (Tm × Tn rows by columns), the reduction tile Tk, how many buffer slots S each block rotates through, how far ahead to prefetch (P tiles), and whether to keep a trailing barrier (β).
>
> On the illustrative GPU (128 SMs, 96 KiB shared memory per block, 1.8 TB/s), the solver chose **128 × 128 × 32 tiles, 3 slots, prefetch 1, no trailing barrier**. Predicted 53.1 µs, 1.3 % faster than T4 because the barrier could be dropped. The design is compute-bound (compute 100 %, DRAM 89 %).

**Optimization lens (mathematician)**

> A mixed-integer linear program over 6 decisions: Tm, Tn ∈ {16, 32, 64, 128, 256}, Tk ∈ {8, 16, 32, 64}, S ∈ {1, 2, 3}, P ∈ {0, 1, 2}, β ∈ {0, 1}. Objective: minimize T with T ≥ T_comp, T ≥ T_mem, and T ≥ T_comp + T_mem when P = 0. Constraints: C1 shared-memory capacity, S·e·(Tm·Tk + Tk·Tn) ≤ 96 KiB; C2 accumulator budget, 256 ≤ Tm·Tn ≤ R_acc; C3 prefetch bound, P ≤ S − 2 + β.
>
> Encoded as 93 variables (90 binary) and 248 constraints. HiGHS: optimal, gap 0 %, dual bound 53.14, 48 ms. Of 722 feasible designs, the optimum lies on the boundary of C3.

**Verification lens (formal methods)**

> Family proof (once per template): C3 is literally the lifetime precondition, so K1 (C ⇒ Pre) holds by construction. The precondition P ≤ S − 2 + β was swept over the 24-point family (S, P, β) at tile 16×16×8 under a two-warp event model; the checker's safe set equals the precondition's admitted set, so the precondition is the **weakest** sound one on this family. This certificate was reused from T2 (same family key), 33 µs.
>
> Instance check (this design): 12 obligations proved in 15 ms. Index: coverage, exact-once, no-phantom over all extents. Schedule: readiness (no read before its load completes) and lifetime (no slot overwritten before its last read) over all interleavings; terminal agreement with the reference. Backward map: forget grid, slots and schedule ⇒ C = A·B over (M, N, K). Rank: k-tiles not yet consumed.

### 2.2 T6: rejected

**Systems lens**

> Same template as T5 on the same workload, but with a **looser prefetch rule**: prefetch may run up to S − 1 tiles ahead regardless of the trailing barrier. That would let the solver save shared memory.
>
> **Rejected before any design was chosen.** With the barrier dropped, a slow warp can still be reading slot 0 when the prefetch for tile k+2 overwrites it. The checker replayed the exact interleaving.

**Optimization lens**

> Identical program to T5 except C3 is relaxed to P ≤ S − 1. The feasible region grows (the solver would prefer S = 2, P = 1, β = 0, which T5's C3 forbids). The solver never ran: the template failed family certification, so by K1 the enlarged region admits designs that are not correct, and solving over it would optimize over wrong designs.

**Verification lens**

> Family proof **refuted** on the lifetime obligation at family point (S = 2, P = 1, β = 0): the precondition admits it, the checker finds an interleaving where warp 1's `LOAD k+2 → slot 0` completes before warp 0's `COMP k` reads slot 0. Counterexample replayed by the independent interpreter (0.25 s). The precondition is not sound; the diagnosis says to strengthen Pre (and C3 with it, to keep K1).

Both lenses say the same thing; the architect hears "slow warp still reading", the verifier hears "lifetime obligation, family point, replayed interleaving", the mathematician hears "enlarged feasible region over wrong designs".

---

## 3. Where the descriptions appear

| Place | Shows |
|---|---|
| **Pool card** | headline; "what's different from parent" line; status chip; ★ best so far; reuse link. No `refines` text anywhere. |
| **Inspector header** | headline; three lens tabs: **Systems · Optimization · Proof**. The last chosen lens is remembered per browser. |
| **Inspector body** | the lens text (§2) followed by the lens's structured block: Systems → design values table with units and the parent's values beside them; Optimization → variables, objective, constraints table with `expr` and `meaning` side by side, solver diagnostics; Proof → obligations table (name, kind, scope, status, time, reused?), the counterexample viewer. |
| **Map badges** | unchanged (round 2), but the evaluator badge adds time. |
| **Narration** | unchanged in structure; workload display names replace raw names (§4.2). |
| **Hover on any term** | a one-line gloss from the glossary (§4.4): hovering "prefetch distance" shows "how many tiles ahead the next load is issued". |

---

## 4. The describer (server side)

`agentic/ledger/describe.py`, new. Pure functions, like `story.py`.

```python
def describe(ledger, contract_id, run) -> dict:
    # { headline, delta_from_parent,
    #   systems:      {text, design_table, parent_values},
    #   optimization: {text, variables, objective, constraints, size, solver},
    #   proof:        {text, family, instance, obligations, counterexample} }
```

### 4.1 Sources, in priority order

1. **Structured fields of the contract**: `decisions[*].meaning`, `.domain`, `.kind`; `constraints[*].expr`, `.meaning`; `cost.model`, `cost.terms`; `preconditions[*].expr`, `.mode`, `.meaning`; `certificate_schema.relation`, `.rank`, `.obligations[*].meaning`, `.scope`; `backward_map`; `program`.
2. **The linked artifacts**: assignment (values, objective, bound, gap, solver_time, diagnostics), verdicts (obligations with status and time, `precondition_is_weakest`, `reused_from`, counterexample), measurement (utilizations, dominant resource), diagnosis.
3. **Template-backend tables** (§4.3): the systems-lens phrasing per template type.
4. **`rationale`** from the formulator: shown as a quote in the inspector under "the agent's own words", never as the description.

The `refines` string is retired from the UI. It stays in the ledger as provenance.

### 4.2 Display names

A small lexicon in `describe.py`, extended when new workloads or chips appear:

| Raw | Display |
|---|---|
| `square_4096` | the 4096³ GEMM |
| `skinny_64x4096x4096` | the skinny 64×4096×4096 GEMM |
| `ragged_1000x1000x1000` | the ragged 1000³ GEMM |
| `illustrative-fp32-gpu` | the illustrative FP32 GPU (128 SMs, 96 KiB smem/block, 1.8 TB/s) |
| `Tm` / `Tn` / `Tk` / `S` / `P` / `beta` | from `decisions[*].meaning`, with units |
| `lifetime` / `readiness` / `coverage` / … | from `certificate_schema.obligations[*].meaning` |

Unknown names fall back to the raw string, never to an error, so a new workload is never invisible.

### 4.3 Per-template phrasing

One entry per template backend in `agentic/templates/<name>/describe.yaml`:

```yaml
smem_tiling:
  what_it_decides: "tiles the {workload} through on-chip shared memory"
  decides_in_words: "the output tile ({Tm} × {Tn}), the reduction tile ({Tk}), how many buffer slots each block rotates through ({S}), how far ahead to prefetch ({P} tiles), and whether to keep a trailing barrier ({beta})"
  precondition_in_words:
    exact: "prefetch may run at most S − 2 tiles ahead, one more if the trailing barrier is kept"
    conservative: "the trailing barrier is always kept, and prefetch may run at most S − 1 tiles ahead"
    unsound: "prefetch may run up to S − 1 tiles ahead regardless of the trailing barrier"
  failure_in_words:
    lifetime: "a slow warp can still be reading a slot when the prefetch for a later tile overwrites it"
    readiness: "a warp reads a slot before the load into it has completed"
    coverage: "part of the output is never computed (the tail tiles are dropped)"
    exact_once: "some output elements are computed twice"
    no_phantom: "the program reads or writes past the end of a tensor"
  delta_fields: [preconditions[0].mode, scope.structure.grid_rounding, scope.structure.leading_barrier, scope.workload.name]
```

`delta_from_parent` is computed by comparing the `delta_fields` against the parent contract and rendering each change with the matching `*_in_words` entry: "Same as T4, but the prefetch rule is exact: prefetch may run at most S − 2 tiles ahead, one more if the trailing barrier is kept."

Adding a template backend means adding a `describe.yaml`; the three lenses then work for it with no UI change.

### 4.4 The glossary

`agentic/ui/static/glossary.json`, one line per term, used for hover glosses and for the "?" next to lens headings. Seeded from the paper's definitions (template, contract, K1–K3, precondition, backward map, obligation, family vs instance, regime, gate vs steer, weakest precondition, incumbent, counterexample, replay). Terms are matched by exact word in the lens text; no NLP.

---

## 5. Components

| Component | File | Change |
|---|---|---|
| Describer | `ledger/describe.py` (new) | §4 |
| API | `ui/server.py` | `/api/describe/<contract>` |
| Pool card | `pool.js` | headline + delta line; `refines` removed |
| Inspector | `inspector.js` | three lens tabs; structured blocks per lens; counterexample viewer unchanged |
| Glossary | `glossary.js` (new), `glossary.json` | hover glosses |
| Story | `story.py` | display names via `describe.display_name()` |
| Template phrasing | `templates/smem_tiling/describe.yaml` (new) | §4.3 |

---

## 6. Build order and acceptance tests

| Stage | Deliverable | Acceptance test |
|---|---|---|
| **D0** Describer | `describe.py`, `describe.yaml`, `tests/test_describe.py` | On `runs/demo.db`, `describe(T5)["headline"]` contains "ragged 1000³ GEMM", "accepted" and "53.1 µs"; `describe(T6)["proof"]["text"]` contains "refuted", "lifetime" and "(S = 2, P = 1, β = 0)"; `describe(T6)["systems"]["text"]` contains "slow warp" and does not contain "beta", "Pre" or "K1"; `describe(T7)["delta_from_parent"]` mentions "tail tiles". No lens text contains the raw string `ragged_1000x1000x1000`. |
| **D1** Pool cards | `pool.js` | Every card shows a headline and a delta line; the string `refines` does not occur in the rendered DOM. A reviewer who has not read the code can say, for each of T1–T7, what it decides and why it was accepted or rejected, from the card alone. |
| **D2** Inspector lenses | `inspector.js` | Switching lens changes the text and the structured block but not the headline; the Optimization lens for T5 shows 93 variables / 248 constraints / gap 0 %; the Proof lens for T6 shows the family point and opens the replayed interleaving. |
| **D3** Glossary | `glossary.js`, `glossary.json` | Hovering "prefetch distance" in any lens shows its gloss; every term in the glossary that appears in T5's three lens texts gets a gloss. |
| **D4** Narration names | `story.py` | Iteration 4's sentence reads "for the ragged 1000³ GEMM"; the round-2 story tests still pass with the new names. |

**Review gate for the round:** three people read the pool and the inspector for T5 and T6 without help: an architect (Li), a mathematician, and a verification expert. Each must be able to answer, from their lens alone, (a) what the template decides, (b) why T6 was rejected, (c) what makes T5 the best so far. If any reader cannot, the phrasing tables in §4.3 are revised, not the UI.

---

## 7. Out of scope

- Generating descriptions with a language model. The descriptions are deterministic renderings of structured fields; a model may later *propose* `describe.yaml` entries for a new template, but what is shown comes from the fields.
- Translating the counterexample interleaving into prose beyond the one `failure_in_words` line; the two-warp timeline stays the primary view.
- Describing EvE races; round 4, after `eve_integration.md`.

---

## 8. One question for Li

Should the **default lens be chosen per user** (remembered in the browser) or **per template status**: Systems for accepted templates, Proof for rejected ones, since that is the question each status raises first? My recommendation is the second, with the user's manual choice overriding it for the session.
