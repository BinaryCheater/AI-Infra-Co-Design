# Chip view: build log, V0–V2

*October 15, 2026. What was built against `microarchitecture_view_design.md`, and where it differs from the note.*

## What exists

- `agentic/ui/static/gb10.svg`: the drawing, with every element id from §3. Each box carries its chapter number and the book's numbers as static labels. Chapter 4 has no box of its own; it is the regime word (the two regimes of Chapter 4). Chapter 9 has no box.
- `agentic/ui/static/microarch.html`: the page, served at `/microarch`. Scrubber over the kernel, a second one over the current phase, step (all events, or the selected element's), play at 1/10/100/1000 events per second, phase jump and "mid", SM select (dropdown or a click on the strip), the regime word with dimming, a click on any element prints the event behind it. The cursor is in the URL hash: `#design=…&sm=…&t=…`.
- `agentic/sim/microtrace.py`: writes the trace. `python -m agentic.sim.microtrace --workload ragged_1000x1000x1000 --tile 128x128x32 --stages 3 --prefetch 1 --beta 0` (T5, 2 s) and `--workload skinny_64x4096x4096 --tile 16x32x8 --stages 2 --prefetch 0 --beta 1` (4 min). Output in `runs/microtrace/<design>/` (ignored by git).
- Server: `/microarch`, `/microtrace/<design>/<file>`, `/api/microtrace`.
- `tests/test_microtrace.py`.

## Simulator change (§4: "that is a simulator change, logged as such")

The simulator did not emit per-instruction events. The note's premise that `sim/run.py` already writes `trace.json` was not true; the schedule view reads the ledger. One change was made:

- `sim/warp.py`: `WarpModel.trace_phase(rec, *args)` re-runs `phase()` uncached with a recorder that receives every issue, every shared load (dispatch, LSU start and end, result ready) and each warp's state per cycle. Recording adds `if rec:` lines inside the loop and changes no result (`test_trace_phase_matches_phase`).

Everything else is recorded from outside, without changing `run.py` or `memory.py`: a list stand-in for `Simulator.trace`, a `Hierarchy` subclass that logs lookups and queue intervals, and a proxy around the warp model that notes which phase used which arguments. The recording run must give the same kernel time as a plain run; `build()` asserts it.

## Where the build differs from the note

1. **Three files, not one.** The simulator computes each distinct k-loop phase once and reuses it, so the trace stores it once (`templates.json`) and each block points at it. Per SM, one block in detail (`smNN.json`), plus `summary.json` for the strip and the L2/DRAM bars.
2. **Which block per SM.** The one that ends last on that SM. The default SM is the one holding the kernel's last-ending block.
3. **A seventh warp state, `ready`.** Eligible but another warp was picked (`not_selected`). Without it, "scheduler-bound" has nothing to show on the warps.
4. **The regime rule.** The phase's critical reason from `run.py` (`phase_wait`, slowest warp per reason). When it is compute, the warp model's histogram decides, grouped by the profiler families `validate.WARP_TO_NCU` already uses: `issued + not_selected` → scheduler-bound; `scoreboard` or `lsu` → load-bound. A copy-critical phase whose tile missed L2 is DRAM-bound, otherwise copy-bound.
5. **Chip strip colour.** What holds the most warp-time in that SM's 1000-cycle window, not "the slowest warp". Same palette.
6. **Arrow dots** are placed by how far along each transfer is at the cursor, not with `<animateMotion>`. That keeps them right under scrubbing and stepping.
7. **Copy bar** counts this block's chunks only. L2 and DRAM bars are chip-wide.
8. **Other blocks' warps.** On an SM with several blocks resident, the warp model runs all their warps together. They are drawn as small dots; the large dots are the detailed block.

## Open

- T5's kernel has no cached SASS counts, so its trace leaves out the per-phase overhead instructions (address, guard, copy issue); the page says so. 16×32×8 has them (91.6 per warp per phase). Running `sim.validate` with a host fills the cache.
- In 16×32×8 the simulated warps stall mostly on `scoreboard` (load results not back), with the MIO bar pinned at depth 70–80 and a 1856-cycle k-loop at six blocks per SM. The profiler's dominant stall for this design is `barrier` (validation row 13); the simulator is 16 % slow on it. The view shows what the model thinks happens, not what the chip does.
- The warp model picks the oldest warp first, so the detailed block (model block 0) finishes its k-loop early and its large dots go to `done` while the small dots of the other blocks are still waiting. That is the model's scheduler choice (book Ch 9, item 2), drawn as it is.
