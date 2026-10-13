# agentic — multi-agent infrastructure with the architect in the loop

Implementation of [`notes/multi_agent_design.md`](../notes/multi_agent_design.md). A large model (or,
today, a script) proposes **templates**, a **solver** optimizes inside them, a **verifier** certifies
them, an **evaluator** measures the result, and every step lands in an append-only **ledger** that a
computer architect watches and steers from a web console.

```
agentic/
  schema/        JSON schemas: contract, assignment, verdict, measurement, diagnosis, event
  ledger/        SQLite append-only ledger; the three stores, the timeline and the template view as views
  agents/
    orchestrator.py   state machine + Controller (observe / step / drive, edit, inject, override, rewind)
    formulator/       catalog.yaml (the structure catalog as data), scripted.py (S3 formulator)
    solver/           solve(T): the 93-variable HiGHS MILP of paper §6.2
    verifier/         certify_family(T), check_instance(T, x): Z3 + independent replay
    evaluator/        measure(T, x) (analytic roofline), probe(T), OptimismRecord
  templates/smem_tiling/   paper Listing 1, vendored from the RQ1 prototype (provenance in __init__.py)
  examples/gemm_tiling/    Listing 1 as a contract (builder + contract.json, the S0 fixture)
  cli/           agentic run | inspect | replay | serve
  ui/            server.py (HTTP + WebSocket over the ledger); static/: the architect console
                 (index.html, console.css, session/transport/scrubber/map/narration/pool/inspector.js)
                 and the v1 ledger debugger (debug.html, app.js)
```

## Run it

Python ≥ 3.12 with `jsonschema numpy scipy z3-solver websockets pyyaml` (`pip install -e .[dev]`).

```bash
agentic run    --ledger runs/demo.db --run gemm-001                  # scripted loop, observe mode, ~2 s
agentic replay --ledger runs/demo.db --run gemm-001 --as gemm-001-r  # byte-for-byte replay check
agentic serve  --ledger runs/demo.db --port 8787                     # architect console
python -m pytest tests -q                                            # the stage tests of design §7
```

## The architect's console (round 2, `notes/architect_console_design.md`)

`agentic serve` opens the console at `/`: a fixed **map** of the five stations with the template token
as the only moving element, a one-paragraph **narration** per iteration written by
`agentic/ledger/story.py`, a **scrubber** with one tick per iteration, a transport (⏮ ◀ ▶ ⏭, Run,
Pause, Step iteration, Step station, Continue; speed 1× / 4× / ∞), **breakpoints** ("Stop at": bug
found and bottleneck moved on by default; proposed, solved, accepted off), the **pool** of template
cards (★ best so far, "proof reused from T2", lineage), and an **inspector** that answers, in order:
why, what was decided, was it safe (a race shown as a two-warp timeline, an index defect as the
missing terms), what was measured, what the orchestrator concluded, raw JSON.

Recorded and live runs are the same picture at different speeds. **new run** starts a scripted run;
the loop pauses at the toggled breakpoints and the station involved pulses. While paused at a
proposal the inspector offers **edit the proposal** and **insert from pool**; at a bottleneck-moved
pause it offers **stay / go deeper**. Verdicts cannot be changed (the rule is shown as a note); a
`verdict` override sent by hand over the WebSocket is refused and recorded as `human_refused`.
The v1 ledger debugger is still served at `/debug.html`.

Console acceptance tests (design §6) are in `tests/test_story.py` (C0, C4, C5 headless); C1–C3 and
the live C4/C5 were checked in the browser on `runs/demo.db`.

## What the architect can do (design §5)

| Action | Effect | Recorded as |
|---|---|---|
| edit | replace the pending proposal with an edited contract; the verifier still has to certify it | `human_edit` (both ids, who, why) |
| inject | replace the pending proposal with a certified template from the population | `human_inject` |
| override `regime` | decide keep / descend at a regime flip (a performance obligation: steers) | `human_override` |
| override `verdict` | **refused**: correctness obligations gate, for humans too | `human_refused` |
| rewind | restore the cursor from any checkpoint and replay; the formulator's recorded proposal is reused unless *fresh* | `rewind`, `replay` |

Commands arrive while the loop waits and are applied in place, so the architect sees their effect
before pressing *continue*. Every checkpoint is a complete state (cursor, pending artifact, ledger
offset); all three modes are one code path with different wait conditions.

## What the scripted run shows

Seven iterations on the shared-memory tiling template (paper §6.2), no model calls:

| # | proposal | what happens |
|---|---|---|
| 1 | square GEMM, trailing barrier kept | admitted; regime assumption wrong (flip) but the actual regime is **proved** as a bound over the 722-design menu |
| 2 | exact precondition `P ≤ S−2+β` | admitted; precondition verified weakest on the 24-point family |
| 3 | skinny GEMM | admitted; family certificate **reused** (0.24 s → 30 µs) |
| 4 | ragged GEMM, conservative | admitted; flip handled by bound → split → bounded upside (`skip_regime`) |
| 5 | ragged GEMM, exact | admitted; incumbent improves 1.34 % (the paper's number) |
| 6 | precondition `P ≤ S−1` (unsound) | **refuted at family certification** with a replayed race (interleaving with the stale read) |
| 7 | floor grid | family proved, solved, **refuted at the instance gate**: coverage counterexample replayed as missing terms |

## Stage status (design §7)

| Stage | Status | Test |
|---|---|---|
| S0 schema + ledger | done | fixture validates; stores rebuild from the ledger alone |
| S1 verifier service | done (wrapped) | 24-point family exactness; unsound refuted with replay; floor grid refuted |
| S2 solver service | done (wrapped) | 93 variables / 90 binary / 248 constraints; optimum reproduced in < 0.2 s |
| S3 orchestrator + scripted formulator | done | every path covered; cross-process replay identical |
| S4 evaluator | analytic backend only | regime detection, probe over the menu, optimism record. **Ventus adapter not built**; "measured" equals "predicted" by construction and is labelled so |
| S5 large-model formulator | not started | decided after the EvE question (`notes/eve_integration.md`) |
| S6 UI | done, v1 | timeline + template view + stores + live ledger |
| S7 drive mode | done | edit / inject / override / rewind / replay, headless tests |
| Console round 2 (C0–C5) | done | story module; map + narration; scrubber + transport; pool + inspector; live breakpoints; architect actions |

Known limits of v1: one template backend (the tiling template), so a regime flip records the sibling
request but cannot formulate it; the solver dispatches on `contract.template` rather than reading a
solver-agnostic search face; Z3's choice of race witness is deterministic across processes, not within
one, so an in-process rewind of a refuted iteration may record a different (equally valid) witness.
