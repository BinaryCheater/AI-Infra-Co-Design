# The Microarchitecture View: Seeing a Kernel Run on the Chip

*Design note, v0.1 · October 15, 2026 · Li Shang, drafted with Claude. For Claude Code. Companion to `schedule_view_design.md` (the time view) and `gb10_microarchitecture.md` (the chip).*

---

## 1. What this is for

An architect reads a schedule and wants to see where on the chip each thing happened. Today the console shows time on one axis and resources on the other. It does not show the chip. This note adds the second view: a drawing of the GB10, driven by the simulator's event trace, that you can scrub, step, and stop.

The two views are one trace drawn two ways. The time view answers "when". The chip view answers "where, and what was in the way". One cursor joins them. A time on the schedule is a state of the chip. A click on a box on the chip is a row on the schedule.

The audience is a person learning or checking the hardware model: Li, a student, a reviewer. Not the agent. So the test for every feature is: can a computer architect who has never seen this tool look at it for ten seconds and say what the kernel is waiting on?

## 2. Three rules

1. **The drawing is the book's figure.** One box per chapter of `gb10_microarchitecture.md`, one arrow per queue between them. If a box has no chapter, remove the box. If a chapter has no box, add one.
2. **One SM in detail, never more.** Every warp, every queue, every shared-memory slot of one SM, for one block, one phase at a time. The other 47 SMs are a strip of 48 small squares at the top, colored by what each is waiting on. The picture must fit in the eye.
3. **Every pixel traces to an event.** Click anything and you get the event behind it: `warp 3, LDS.128, 8-way conflict, 4 wavefronts, waited 11 cycles behind warp 1`. Nothing is drawn that the trace did not say.

## 3. The drawing

A hand-drawn SVG, `agentic/ui/static/gb10.svg`, with named elements. No layout engine. Top to bottom, following the data path:

```
┌──────────────────────────── chip strip: 48 SMs ───────────────────────────┐
│ ■■■■■■■■ ■■■■■■■■ ■■■■■■■■ ■■■■■■■■ ■■■■■■■■ ■■■■■■■■   (6 groups of 8)  │
└────────────────────────────────────────────────────────────────────────────┘
┌────────────────────────── SM 17 (the one in detail) ───────────────────────┐
│  sched0   sched1   sched2   sched3        ← 4 sub-partitions               │
│  [w0 w4]  [w1 w5]  [w2 w6]  [w3 w7]       ← warps of the block, by scheduler│
│     │        │        │        │                                            │
│  FMA pipe (4×)         MIO queue ▮▮▮▮░░░░  →  LSU: wavefronts/cycle ▮       │
│                                   │                                         │
│  shared memory 100 KB: [slot0: A B][slot1: A B][slot2: ...]   L1 64 KB      │
│  cp.async engine: chunks in flight ▮▮▮▮▮▮▮▮░░░   TMA                        │
│  barrier: 5/8 arrived                                                       │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │ 32 B sectors
┌───────────────────────────────┴────────────────────────────────────────────┐
│  L2  24 MiB   hits ▮▮▮▮▮▮▮░  queue ▮▮░░   1.0 TB/s                         │
└───────────────────────────────┬────────────────────────────────────────────┘
┌───────────────────────────────┴────────────────────────────────────────────┐
│  LPDDR5X  235 GB/s read   queue ▮▮▮░░   shared with CPU                     │
└────────────────────────────────────────────────────────────────────────────┘
```

Element ids are the resource names the simulator already uses, so the trace binds to the drawing by name and nothing else: `chip.sm[17]`, `sm.sched[2]`, `sm.warp[5]`, `sm.fma`, `sm.mio`, `sm.lsu`, `sm.smem.slot[1]`, `sm.copy`, `sm.tma`, `sm.barrier`, `l2`, `l2.queue`, `dram`, `dram.queue`. Arrows have ids too: `arrow.dram.l2`, `arrow.l2.sm`, `arrow.lsu.smem`, `arrow.smem.fma`.

Static labels on the drawing are the book's numbers: `4.2 cyc/dispatch`, `1 wavefront/cyc`, `136 ns`, `235 GB/s`. They never change. What changes is color and fill.

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

One file, `agentic/ui/static/microarch.html`, no framework, served by the existing console next to `/schedule`, `/calibration`, `/validation`. It loads `gb10.svg` inline and `trace.json` for the selected run from the existing run endpoint. State is a plain object keyed by element id; rendering sets `fill`, `stroke`, a `data-state` attribute and the text of a few `<text>` nodes. Dots on arrows are SVG `<animateMotion>` along the arrow's path, started on the event and removed at `t + dur`. Breakpoints are a 60-line parser into a predicate function. Total under 1500 lines. The schedule view's cursor is shared through one URL hash parameter, `#t=184233`, so either page can be opened at a time and the two pages stay in step in two tabs.

A trace of a 4096³ GEMM is too big to load whole. The simulator already caps simulation by block-phases; the view loads one SM's events for a window of phases (default: the detailed SM, all phases of one block), and the chip strip is driven by a per-SM summary the simulator writes alongside the trace (state per SM per 1000 cycles). Two files, not one.

## 11. Build order

| Stage | Deliverable | Done when |
|---|---|---|
| V0 | `gb10.svg` with every id in §3, static labels from the book; `microarch.html` loads it and the trace, and draws the state at one fixed `t` | Li opens it and names every box without a legend |
| V1 | scrubber, step, play; warp dots, MIO and LSU bars, shared-memory slot states, arrow dots | the six-step tour of T5 can be followed by hand |
| V2 | the regime word and dimming; the chip strip; SM selection | the 16×32×8 design shows `load-bound` with the MIO bar pinned; T5 shows `scheduler-bound` |
| V3 | the link to the schedule view (shared cursor, click-through both ways); the inspector's "behind" sentence | click a wait on either page, read what it waited behind |
| V4 | breakpoints; measured-beside-simulated bars; teaching mode | `sm.mio.depth > 8` stops in phase 1 of the 16×32×8 design; the T5 tour runs end to end |

Review after V2: Li scrubs through one phase of T5 and one of 16×32×8 and says, in one sentence each, what the SM is waiting on. If he needs the legend, V0 is not done.

## 12. What this is not

Not a replacement for the schedule view; the two are one tool. Not a general GPU visualizer; it draws this chip from this book and the trace from this simulator. Not for the agent; the agent reads the trace's numbers directly. When the chip changes (the 5090, Ventus), the SVG is redrawn from that chip's book, and nothing else changes.
