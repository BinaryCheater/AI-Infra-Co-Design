# The Microarchitecture View: Seeing a Kernel Run on the Chip

*Design note, v0.2 · October 15, 2026 · Li Shang, drafted with Claude. For Claude Code. Companion to `schedule_view_design.md` (the time view) and `gb10_microarchitecture.md` (the chip).*

---

## 1. What this is for

An architect reads a schedule and wants to see where on the chip each thing happened. Today the console shows time on one axis and resources on the other. It does not show the chip. This note adds the second view: a drawing of the GB10, driven by the simulator's event trace, that you can scrub, step, and stop.

The two views are one trace drawn two ways. The time view answers "when". The chip view answers "where, and what was in the way". One cursor joins them. A time on the schedule is a state of the chip. A click on a box on the chip is a row on the schedule.

The audience is a person learning or checking the hardware model: Li, a student, a reviewer. Not the agent. So the test for every feature is: can a computer architect who has never seen this tool look at it for ten seconds and say what the kernel is waiting on?

## 2. Three rules

1. **The drawing is the book's figure.** One box per chapter of `gb10_microarchitecture.md`, one arrow per queue between them. If a box has no chapter, remove the box. If a chapter has no box, add one.
2. **One SM in detail, never more.** Every warp, every queue, every shared-memory slot of one SM, for one block, one phase at a time. The other 47 SMs are a strip of 48 small squares at the top, colored by what each is waiting on. The picture must fit in the eye.
3. **Every pixel traces to an event.** Click anything and you get the event behind it: `warp 3, LDS.128, 8-way conflict, 4 wavefronts, waited 11 cycles behind warp 1`. Nothing is drawn that the trace did not say.

## 3. The drawing is generated from the chip config

The simulator already takes the chip as data: `sim/config/<chip>.yaml`, validated by `schema/chip.json`. Every count, capacity, latency and rate is there, and the kernel is scheduled against it. The drawing must come from the same file. Change the config, and the drawing, the trace, the schedule and the chip view all change together. A hand-drawn picture would be a second copy of the chip that drifts.

**Four kinds of component.** Every component in the config gets a `kind`, and each kind has one way of being drawn and one way of showing state:

| kind | what it is | drawn as | state shown |
|---|---|---|---|
| `queue` | something that holds requests until a stage accepts them: the MIO queue, the copy engine's in-flight chunks, the L2 request queue, the DRAM queue, a barrier's arrival count | a bar of cells, filled from the left; capacity from the config, or unbounded with a soft mark at the measured depth | depth |
| `pipe` | something that is busy or not: a scheduler, the FMA pipe, the LSU, a tensor core | a cell that lights | busy, and for how many more cycles |
| `storage` | something with slots that hold data: shared-memory slots, L1, L2, DRAM, the register file as a capacity | slots, each `empty / in flight / ready / consumed`; for a cache, a hit-rate bar instead of slots | fill and a rate |
| `link` | something data travels along: scheduler to LSU, LSU to shared memory, SM to L2, L2 to DRAM | an arrow; a dot travels along it for an event's duration | traffic |

Warps are the one thing that is not a component. They are the actors. A warp is a dot that sits in its scheduler's box and changes color by what it waits on.

**Layout rule.** The config also says how components nest and in what order they appear on the data path. The layout is: the chip is a row of SM squares grouped as the config groups them (6 × 8 for the GB10, from the dispatch order in `block_dispatch_order`); below it, one SM drawn in detail, its components top to bottom in data-path order (schedulers, pipes, the queue into the LSU, the LSU, storage, copy engines, barrier); below that, each level of the memory hierarchy as one box with its queue, down to DRAM. Links are drawn between consecutive levels. That rule is the book's figure, written once as code in `agentic/ui/static/floorplan.js`, and it draws any chip whose config follows the schema.

For the GB10 it produces:

```
┌──────────────────────────── chip strip: 48 SMs ───────────────────────────┐
│ ■■■■■■■■ ■■■■■■■■ ■■■■■■■■ ■■■■■■■■ ■■■■■■■■ ■■■■■■■■   (6 groups of 8)  │
└────────────────────────────────────────────────────────────────────────────┘
┌────────────────────────── SM 17 (the one in detail) ───────────────────────┐
│  sched0   sched1   sched2   sched3        ← pipes                          │
│  [w0 w4]  [w1 w5]  [w2 w6]  [w3 w7]       ← warps, by scheduler            │
│  FMA pipe (4×)         MIO queue ▮▮▮▮░░░░  →  LSU ▮        ← pipe, queue, pipe│
│  shared memory 100 KB: [slot0][slot1][slot2]     L1 64 KB  ← storage        │
│  copy engine ▮▮▮▮▮▮▮▮░░░   TMA              barrier 5/8    ← queues         │
└───────────────────────────────┬────────────────────────────────────────────┘
│  L2  24 MiB   hits ▮▮▮▮▮▮▮░  queue ▮▮░░   1.0 TB/s           ← storage + queue
│  LPDDR5X  235 GB/s   queue ▮▮▮░░   shared with CPU           ← storage + queue
```

For a Ventus config (2 SMs, 1 scheduler each, a scratchpad, no L1, no L2) the same code draws two squares, one scheduler, one pipe, one scratchpad box, DRAM. Nothing in `floorplan.js` knows the word GB10.

**Names.** Element ids are the resource names the simulator uses in the trace: `chip.sm[17]`, `sm.sched[2]`, `sm.warp[5]`, `sm.fma`, `sm.mio`, `sm.lsu`, `sm.smem.slot[1]`, `sm.copy`, `sm.tma`, `sm.barrier`, `l2`, `l2.queue`, `dram`, `dram.queue`, and `link.<from>.<to>`. The config, the simulator and the drawing share one namespace. That is the only binding.

**Labels.** Each box carries its config numbers as static text: `4.2 cyc/dispatch`, `1 wavefront/cyc`, `24 MiB`, `136 ns`, `235 GB/s`. They come from the config, so a hypothetical chip shows its own numbers, marked `simulated` where the config marks them as not measured.

**Schema change.** `schema/chip.json` gains, per component: `kind` (one of the four), `capacity` (cells for a queue, slots for storage; absent means unbounded), `parent` (which component contains it), and `order` (position on the data path). `fit.py` writes them for the GB10 from the structure the benchmarks found. A config without them fails validation, so no chip can be simulated that cannot be drawn.

## 4. The trace

The simulator writes one JSON file per run, `trace.json`, already produced by `sim/run.py` for the schedule view. Each event:

```json
{"t": 184233, "res": "sm.mio", "sm": 17, "warp": 5, "kind": "lds",
 "width": 128, "wavefronts": 4, "conflict": 1, "queued": 11,
 "phase": 2, "step": 6, "dur": 4}
```

`t` is in SM cycles. `res` is the element id. `kind` is one of: `issue`, `fma`, `lds`, `sts`, `ldmatrix`, `copy_issue`, `copy_done`, `wait`, `barrier_arrive`, `barrier_release`, `l2_hit`, `l2_miss`, `dram_read`, `dram_write`, `block_start`, `block_end`. `dur` is how long the resource is held. Queue events carry `depth` after the event. Warp events carry `state` after the event: `issuing`, `scoreboard`, `mio`, `barrier`, `copy`, `done`.

Nothing new is invented for the view. If the simulator does not emit an event, the view does not draw it. If the view needs something the simulator does not emit, that is a simulator change, logged as such.

## 5. What moves

**State, not motion.** At cursor time `t`, every element has a state computed from the events up to `t`. Scrub the cursor, recompute, recolor. This is cheap: keep a running state and apply events forward or backward.

| Element | State shown | How |
|---|---|---|
| chip strip square | what that SM's slowest warp is waiting on | fill color: FMA green, MIO orange, copy blue, barrier grey, idle white |
| scheduler | issuing or not | border: bright when it issued this cycle |
| warp | `issuing / scoreboard / mio / barrier / copy / done` | dot color, same palette as the strip; the warp's dot sits in its scheduler's box |
| MIO queue | depth | a bar of up to 16 cells, filled from the left |
| LSU | busy for how many more cycles | a single cell that is lit while a wavefront drains, with the count |
| FMA pipe | busy | lit per sub-partition |
| shared-memory slot | `empty / in flight / ready / consumed` | fill: white, hatched, solid, dimmed; label A or B |
| cp.async engine | chunks in flight | bar |
| barrier | arrived / total | `5/8` and a ring that closes |
| L2 | hit rate over the last 1000 cycles; queue depth | two bars |
| DRAM | bytes/s over the last 1000 cycles; queue depth | two bars |
| arrows | data moving | a dot travels from source to destination for the event's `dur`; many dots is traffic, no dots is silence |

**The regime word.** At every cursor time the view names the regime for the detailed SM in large type at the top of its box: `load-bound`, `scheduler-bound`, `copy-bound`, `barrier-bound`, `DRAM-bound`. The rule is the simulator's own critical-path rule for the current phase, not a new one. Everything not on the critical path dims to 40 %. This is the ten-second test.

## 6. Moving through time

Controls, left to right under the drawing:

- **Scrubber**: the same cursor as the schedule view, in cycles and µs. Dragging it moves both views.
- **Step**: next event, previous event. Next event on the selected element (click a box, then step: you follow that queue).
- **Play** at 1, 10, 100, 1000 events per second. Not cycles per second; events, so quiet stretches go fast and busy ones slow down.
- **Phase**: jump to the start of phase *n* of the detailed block; the phase number is on the schedule view too.
- **SM**: choose which SM is shown in detail. Default: the SM whose block is on the kernel's critical path. Clicking a square on the chip strip selects it.

**Breakpoints** are predicates over events, written in a small box, one per line:

```
sm.mio.depth > 8
kind == barrier_release and phase == 2
warp 3 state == mio for > 50 cycles
res == dram.queue and depth == max
```

Play stops at the first event that satisfies one. The matching event is highlighted on both views and its payload is printed. The grammar is: `field op value`, joined by `and`, with `for > N cycles` for a state that persists. Nothing more.

## 7. The two views together

| Action | Schedule view | Chip view |
|---|---|---|
| drag the cursor | the vertical line moves | states recompute |
| click a resource row | the row highlights | the element highlights, and its history for ±500 cycles appears as a small sparkline under the drawing |
| click an element | the row highlights, the schedule scrolls to it | the element highlights |
| click an event (a bar on the schedule, a dot on the chip) | the inspector shows the event | the inspector shows the event, and the elements it touched flash |
| a breakpoint fires | the line jumps | states recompute, the event's elements flash |

The inspector is the existing one in the console (round 2). It shows the event's fields and, for a wait, what it waited behind: `queued 11 cycles behind warp 1's LDS.128 (4 wavefronts) and warp 4's LDS.32 (1)`. That "behind" list is computed from the queue's events between the wait's start and end; it is the sentence an architect wants.

## 8. Measured beside simulated

For a design that was run on the Spark, the profiler's counters are drawn on the same picture in a second, thinner bar next to each simulated one: `mio_throttle` per issue cycle next to the MIO depth, `issue_active` next to the schedulers, L2 hit rate next to the simulated hit rate, wavefronts per LDS next to the simulated count. Same scale, same place. A wrong queue is then seen, not computed. Counters exist per kernel, not per cycle, so the measured bar is flat across the run; that is fine, it is a reference line.

## 9. Teaching mode

A toggle that replaces the controls with a six-step tour of one phase of one block of the 128×128×32 GEMM: block starts, copies issue, copies land (slot goes solid), barrier releases, the k-loop runs (MIO fills, FMA lights), the next tile's copies overlap. Each step is a breakpoint with a one-sentence caption from the book. Six captions, no more. The same tour on the 16×32×8 tile shows the MIO bar pinned and the word `load-bound`; that contrast is the lesson.

## 10. How it is built

Two files under `agentic/ui/static`, no framework, served by the existing console next to `/schedule`, `/calibration` and `/validation`:

- `floorplan.js`: reads the chip config and emits the SVG. One function per component kind (`drawQueue`, `drawPipe`, `drawStorage`, `drawLink`) and one layout function that walks the config in data-path order. Under 400 lines.
- `microarch.html`: loads the config and the trace for the selected run, calls `floorplan.js`, then drives state. State is a plain object keyed by element id; rendering sets `fill`, `stroke`, a `data-state` attribute and the text of a few `<text>` nodes. Dots on links are SVG `<animateMotion>` along the link's path, started on the event and removed at `t + dur`. Breakpoints are a 60-line parser into a predicate function. Under 1100 lines.

The schedule view's cursor is shared through one URL hash parameter, `#t=184233`, so the two pages stay in step in two tabs.

A trace of a 4096³ GEMM is too big to load whole. The view loads one SM's events for a window of phases (default: the detailed SM, all phases of one block), and the chip strip is driven by a per-SM summary the simulator writes beside the trace (state per SM per 1000 cycles). Two files, not one.

**Changing the chip.** Edit the YAML or let the solver emit one, rerun the simulator, reload the page. The drawing, the labels, the trace, the schedule and the regime word all follow. That is the loop the co-design needs: the solver changes the chip every iteration, and the architect sees each iteration on its own floorplan.

## 11. Build order

| Stage | Deliverable | Done when |
|---|---|---|
| V0 | `kind`, `capacity`, `parent`, `order` in `schema/chip.json` and the GB10 config; `floorplan.js` draws the GB10 from it with every id in §3 and its labels; `microarch.html` loads it and the trace and draws the state at one fixed `t` | Li opens it and names every box without a legend; a config with 2 SMs and no L2 draws without code changes |
| V1 | scrubber, step, play; warp dots, MIO and LSU bars, shared-memory slot states, arrow dots | the six-step tour of T5 can be followed by hand |
| V2 | the regime word and dimming; the chip strip; SM selection | the 16×32×8 design shows `load-bound` with the MIO bar pinned; T5 shows `scheduler-bound` |
| V3 | the link to the schedule view (shared cursor, click-through both ways); the inspector's "behind" sentence | click a wait on either page, read what it waited behind |
| V4 | breakpoints; measured-beside-simulated bars; teaching mode | `sm.mio.depth > 8` stops in phase 1 of the 16×32×8 design; the T5 tour runs end to end |

Review after V2: Li scrubs through one phase of T5 and one of 16×32×8 and says, in one sentence each, what the SM is waiting on. If he needs the legend, V0 is not done.

## 12. What this is not

Not a replacement for the schedule view; the two are one tool. Not a general GPU visualizer; it draws this chip from this book and the trace from this simulator. Not for the agent; the agent reads the trace's numbers directly. When the chip changes (the 5090, Ventus, a chip the solver proposed), its config is the only thing that changes; the drawing follows from it.
