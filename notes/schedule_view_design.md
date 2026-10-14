# The Schedule View: Console Round 4

**Design document, v0.1 · October 14, 2026**
*Li Shang, drafted with Claude. Follows `describing_templates.md` (round 3). Written for Claude Code to execute.*

---

## 0. What is missing

The console now says *what* a template decides and *whether* it was proved. It does not show *how the design runs*. For T5 the architect reads "128 × 128 × 32 tiles, 3 slots, prefetch 1, no trailing barrier" and has to imagine the pipeline. The mathematician reads "T ≥ max(T_comp, T_mem)" and has to imagine where the overlap comes from. The verifier reads "lifetime proved over all interleavings" and has to imagine which window that protects.

All three want the same picture: **a schedule of the computation, the memory traffic and the synchronization that the solver chose and the verifier certified**, drawn so the overlap, the slot lifetimes and the barriers are visible.

Everything needed to draw it is already determined by the contract and the assignment. The template's `program` fixes the order of loads, barriers and computes; the design fixes the tile sizes, the slot count S, the prefetch distance P and the trailing barrier β; the cost model fixes the modeled durations; the certificate schema fixes which windows the obligations protect; and a refuted verdict carries the exact interleaving that breaks one. Round 4 renders these. No new agent, no new ledger event.

---

## 1. What the view shows

One tab in the inspector, **Schedule**, with three stacked panels and one strip. All panels share the same horizontal axis.

### 1.1 The block pipeline (the main panel)

For one thread block, the k-phases of the GEMM left to right. Lanes top to bottom:

| Lane | What is drawn | Source |
|---|---|---|
| **Memory: DRAM → shared** | one bar per tile load: `LOAD k-tile u → slot u mod S`, from issue to completion; colored by slot | `program`, design (S, P), cost model (per-tile bytes / bandwidth share) |
| **Slot 0 … Slot S−1** | one occupancy bar per slot per tile it holds: from the load's completion to the last read of that tile (its **lifetime**) | derived from the program and P |
| **Sync** | a vertical tick per barrier: leading barrier every phase; trailing barrier when β = 1 | `program`, design (β) |
| **Compute: MMA** | one bar per phase: `COMPUTE k-tile p from slot p mod S` | `program`, cost model (block FLOPs / KT / per-SM rate) |

What the picture makes obvious without reading: the **prefetch distance** is the horizontal gap between a tile's load and its compute; the **overlap** is the part of a load bar that sits above a compute bar; a **trailing barrier** is the extra tick after each compute that delays the next load; and a **free slot** is the gap in a slot lane. T4 (β = 1) and T5 (β = 0) side by side show exactly where 1.3 % came from.

The pipeline has KT = ⌈K / Tk⌉ phases (32 for T5). The panel shows the first P + S + 2 phases in full, an ellipsis, and the last phase, with a toggle "all phases" that scrolls.

### 1.2 The grid and waves (the small panel above)

The output grid of ⌈M/Tm⌉ × ⌈N/Tn⌉ blocks drawn as a small raster, grouped into **waves** of `sms` blocks (128 on the illustrative chip). Each wave is one row of the block pipeline in time: the full kernel is `waves` copies of panel 1.1 back to back. For T5 on the ragged GEMM: 8 × 8 = 64 blocks, 1 wave; for T1 on the 4096³ GEMM: 32 × 32 = 1024 blocks, 8 waves. Hovering a block highlights its tile extent; partial tiles on the ragged edge are hatched, and for T7 (floor grid) the uncovered tail is drawn red, which is the coverage counterexample made visible.

### 1.3 The resource strip (the bottom strip)

Three horizontal gauges over the same time axis, from the cost model's terms:

- **DRAM bandwidth:** `memory_s / time_s` as a fill level, with the traffic in bytes beside it.
- **Compute:** `compute_s / time_s`.
- **Shared memory:** `smem_bytes / smem_bytes_per_block`.

The dominant resource is labelled; this is the same number the Evaluator badge shows, so the architect sees *why* the design is compute-bound: the compute lane is solid and the memory lane has gaps.

### 1.4 What the verifier certified (the overlay)

A toggle **"show what was proved"** draws the obligations onto the pipeline as windows:

- **readiness:** for each compute bar, a bracket from the completion of its tile's load to the start of the compute: "the load finished before the read". Green when proved.
- **lifetime:** for each slot occupancy bar, a bracket from its last read to the next load into that slot: "nothing overwrote it in between". Green when proved.
- **coverage / exact-once / no-phantom:** on the grid panel: every output cell covered exactly once, no cell past the edge.

For a **refuted** template, the overlay switches to the counterexample: the two-warp interleaving from the verdict (already shown as a timeline in round 2) is drawn **in the same lanes**, with the violating pair in red. For T6 that is warp 1's `LOAD k+2 → slot 0` completing inside warp 0's `COMPUTE k` read window of slot 0: the lifetime bracket is broken, and the picture shows where. This is the moment the view pays for itself: the architect sees the bug as a schedule, the verifier sees it as an obligation, the mathematician sees it as the constraint C3 that was relaxed.

### 1.5 Compare with parent

A second toggle **"compare with parent"** draws the parent template's pipeline in a lighter color behind the current one, aligned on the phase axis. T5 over T4: the trailing ticks disappear and the compute bars close up. T3 over T2: same pipeline, different grid (1 wave instead of 8).

---

## 2. Honesty about time

The cost model is aggregate: it gives `compute_s`, `memory_s` and `max(comp, mem)` for the whole block, not per-phase durations. The pipeline therefore draws **structural** truth (ordering, overlap, slot lifetimes, barriers, which the verifier certified) with **modeled** durations:

- compute bar = `block_flops / KT / fp32_flops_per_s_per_sm`
- load bar = `tile_bytes / (dram_bytes_per_s / blocks_per_wave)`
- barrier tick = `barrier_s`
- phases are laid out by the program order with these durations, so the total block time equals the cost model's `compute_s` or `memory_s` term by construction (whichever dominates). A caption states the three formulas and says "modeled, not measured".

When the Ventus adapter lands (S4 follow-up), a measured trace replaces the durations and the caption changes to "measured on Ventus"; the lanes and the overlay do not change. The view is designed so that measured and modeled schedules are the same picture, which is also how a model–measurement gap becomes visible: the two pipelines drawn together.

For the verifier's event model (two warps, five phases, Tn = 2), the overlay uses the event model's own phases, not the cost model's, and the caption says so.

---

## 3. Data: the schedule object

`agentic/ledger/schedule.py`, new. A pure function over the ledger, like `describe.py`:

```python
def schedule(ledger, contract_id, run) -> dict:
    # {
    #   axis:  {unit: "s", phases: KT, shown: [...phase indices], modeled: True, caption: str},
    #   block: {
    #     loads:     [{tile, slot, issue_phase, t0, t1, bytes}],
    #     slots:     [{slot, tile, t_ready, t_last_read}],        # lifetime windows
    #     barriers:  [{phase, kind: "leading"|"trailing", t}],
    #     computes:  [{tile, slot, t0, t1, flops}],
    #   },
    #   grid:  {gm, gn, tm, tn, m, n, waves, sms, partial: [{i, j, rows, cols}], uncovered: [...]},
    #   resources: {dram: {util, bytes, s}, compute: {util, s}, smem: {bytes, cap}, dominant},
    #   proved: {readiness: [...windows], lifetime: [...windows], index: {coverage, exact_once, no_phantom}},
    #   counterexample: null | {warps: [[events]], violation: {obligation, slot, tile, pair}},
    #   parent: contract_id | null
    # }
```

Sources: the contract's `program`, `decisions`, `scope.workload`, `scope.chip`; the assignment's `values`; the verdict's `obligations` and `counterexample`; the cost model (`cost_model(d, w, hw)` from the template backend, so the numbers match the solver's objective exactly). The per-template part (how to turn `program` text into load/barrier/compute events) is one function per template backend, `agentic/templates/<name>/schedule.py`; the renderer is generic.

`/api/schedule/<contract>` returns it. Nothing is written to the ledger.

---

## 4. Components

| Component | File | Responsibility |
|---|---|---|
| Schedule builder | `ledger/schedule.py` (new) | §3; calls the template's `schedule.py` |
| Template events | `templates/smem_tiling/schedule.py` (new) | program → events for this template |
| API | `ui/server.py` | `/api/schedule/<id>` |
| Schedule panel | `ui/static/schedule.js` (new) | SVG: grid panel, pipeline lanes, resource strip, proved overlay, counterexample overlay, parent ghost; hover on any bar shows its tile, slot, times and bytes |
| Inspector | `inspector.js` | fourth tab **Schedule**, after the three lenses; the tab is the default for an accepted template with a solved design when the user came from the Solver or Gate station badge |
| Glossary | `glossary.json` | add: wave, phase, slot lifetime, prefetch overlap, leading/trailing barrier |

Rendering rules: one color per slot (three slots → three colors, consistent across all panels and the parent ghost); compute in navy; barriers as thin red ticks; proved windows in teal; counterexample pair in red with the rest dimmed. Minimum bar height 14 px; the pipeline never scrolls horizontally in the default (abridged) view.

---

## 5. Build order and acceptance tests

| Stage | Deliverable | Acceptance test |
|---|---|---|
| **E0** Schedule builder | `schedule.py` (ledger + template), `/api/schedule`, `tests/test_schedule.py` | On `runs/demo.db`: `schedule(T5)` has KT = 32 phases, 3 slots, no trailing barriers, 32 leading barriers; every compute's `t0` ≥ its tile's load `t1` (readiness by construction); the sum of compute bars equals `compute_s` of the cost model within 1e-9; `schedule(T4)` has 32 trailing barriers and its block time exceeds T5's by the barrier cost; `schedule(T7).grid.uncovered` is non-empty; `schedule(T6).counterexample` is non-null with `violation.obligation == "lifetime"`. |
| **E1** Pipeline + grid + strip | `schedule.js` | T5 renders with three slot lanes in three colors; hovering the phase-3 compute bar shows "k-tile 3 from slot 0"; the resource strip labels compute as dominant and its fill matches the Evaluator badge's utilization. T1 shows 8 waves. |
| **E2** Proved overlay | overlay in `schedule.js` | On T5, "show what was proved" draws one teal readiness bracket per compute and one teal lifetime bracket per slot occupancy, all green; the obligation names match the Proof lens list. |
| **E3** Counterexample overlay | overlay in `schedule.js` | On T6 the overlay shows two warp rows; the red pair is `LOAD k+2 → slot 0` (warp 1) against `COMPUTE k` (warp 0) and the broken lifetime bracket; the text under it is the round-3 `failure_in_words` for lifetime. On T7 the grid shows the red uncovered tail. |
| **E4** Compare with parent | ghost in `schedule.js` | T5 with "compare with parent" shows T4's trailing ticks ghosted and the compute bars shifted; the caption says "T4: 53.9 µs → T5: 53.1 µs (−1.3 %)". |

**Review gate for the round:** the architect can point at the T5 pipeline and say where the prefetch overlap is and why there is no trailing barrier; the verifier can point at the T6 overlay and say which write broke which read; the mathematician can say, from the strip, why the design is compute-bound. If any of the three cannot, the lanes are revised, not the data.

---

## 6. Beyond the GEMM template

This template has one communication path: DRAM ↔ shared memory. The lanes generalize: a template that introduces a NoC (the system-architecture level, Prof. Peh's CGRA testbed) adds a **Communication** lane group, one lane per link or per router hop, drawn with the same bar-and-bracket vocabulary; a multi-block or multi-core template adds one pipeline per core stacked vertically with the communication lanes between them. The renderer takes lanes from the schedule object, so adding a template backend adds lanes without changing `schedule.js`. This is the view that will carry the second template backend and the first descent when they arrive.

---

## 7. Out of scope

- Per-phase timing from measurement (needs the Ventus adapter).
- Editing the schedule by hand to propose a design (the architect's edit action stays at the contract level).
- Animating the schedule in time; the pipeline is a static Gantt, which is easier to read than an animation for this content.

---

## 8. Questions for Li

1. Should the abridged view show the first `P + S + 2` phases (enough to see one full slot rotation) or a fixed 8?
2. For the grid panel on the 4096³ GEMM (1024 blocks), draw the raster at block granularity or collapse to one cell per wave?
3. Should the resource strip also show the **baseline** (the coarse level's streaming design, `baseline_cost`) as a faint reference, so the gain from tiling is visible?
