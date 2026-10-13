# The Architect's Console: Design Specification, Round 2

**Design document, v0.1 · October 13, 2026**
*Li Shang, drafted with Claude. Companion to `multi_agent_design.md` (§5–§6) and the v1 implementation in `agentic/ui/`. Written for Claude Code to execute.*

---

## 0. Why a second round

The v1 console (`agentic/ui/`) was built from the ledger outward: a table of events and a template view. It is a debugger for the system. It is not a window for an architect. Li's verdict after using it: *"really hard to understand."*

**The governing principle for round 2:** the human architect's reading speed is far slower than the loop. The loop finishes an iteration in two seconds; an architect needs a minute to understand one. The interface's job is therefore to **pace and compress**, not to display. Everything below follows from that.

Three consequences:

1. **A fixed picture.** The agents are drawn once, as stations on a map that never moves. The only thing that moves is the template. People can follow one object moving on a still map; they cannot follow a scrolling log.
2. **One sentence per iteration.** Each iteration is summarized in plain language, composed from fields the ledger already holds. The architect reads the sentence first and opens details only when the sentence raises a question.
3. **The architect sets the pace.** Live and recorded runs are the same picture at different speeds. A scrubber moves through iterations; breakpoints stop the loop where the architect wants to look.

Nothing changes in the data model. The ledger stays the single source of truth; the console is a view over it, as in v1. The small additions to the server and Controller are listed in §7.

---

## 1. Vocabulary: what the screen says

The console uses plain words. Ledger terms appear only in the inspector and in this glossary.

| Ledger / paper term | On screen | Meaning for the architect |
|---|---|---|
| contract, template | **template** | one proposal for one refinement step |
| proposed | **proposed** | the agent has written a template |
| family_check proved | **proved safe for all designs** | the template is certified once, for everything it can produce |
| family_check refuted | **rejected: the checker found a bug** | a concrete failing case was found and replayed |
| solved | **best design found** | the solver has picked the design, with time and gap |
| instance_check proved | **this design checked** | the per-design gate passed |
| instance_check refuted | **rejected: this design has a bug** | the gate failed, with the failing case |
| instance_check open | **could not decide** | the checker timed out; not trusted |
| measured | **measured** | the evaluator ran it (today: the cost model; later: simulation) |
| regime_flip | **the bottleneck moved** | the template assumed memory-bound; the design came out compute-bound, or vice versa |
| regime_bound holds | **assumption proven** | the regime holds on every feasible design |
| skip_regime | **other regime can't win** | the other regime's best possible result is not worth searching |
| probe_result | **sampled the other regime** | n probes, w wins, confidence bound |
| admitted | **accepted into the pool** | the template joins the library of proven templates |
| incumbent | **best so far** | the best design for this workload on this chip |
| strength observed / proved | **assumed / proven** | whether the regime assumption is a measurement or a bound |
| gate / steer | **blocks / guides** | a bug blocks the design; a performance assumption only guides the search |
| counterexample | **the bug, step by step** | the replayed interleaving or the missing index terms |
| reused (family_check) | **proof reused** | certified earlier; checked in microseconds |
| diagnosis | **what the orchestrator concluded** | symptom, cause, response |
| checkpoint | **pause point** | where the loop can stop |
| human_edit / inject / override / refused | **you changed / you inserted / you decided / not allowed** | the architect's own actions |

---

## 2. The screen

One page, three regions. No scrolling for the main view; the inspector slides in from the right.

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ ARCHITECT CONSOLE   run: gemm-001 ▾   ● live · paused at "proposed"      speed  1×  4×  ∞ │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ ⏮  ◀  ▶  ⏭   [Run] [Pause] [Step iteration] [Step station] [Continue]                   │
│ Stop at:  [✓ bug found] [✓ bottleneck moved] [ proposed ] [ solved ] [ accepted ]        │
│ ├─1──2──3──4──5──6──7──┤  scrubber: ● accepted  ✕ rejected  ⟲ bottleneck moved          │
├────────────────────────────────────────────────────────┬─────────────────────────────────┤
│                 THE MAP (fixed)                        │   THE POOL (shelf of cards)     │
│  levels                                                │                                 │
│  1 workload       ┌───────────┐     ┌───────────┐      │  ★ T5  ragged · exact           │
│  2 graph          │ FORMULATOR│ ──▶ │ VERIFIER  │      │     best so far 53.14 µs        │
│  3 operators      │ propose   │     │ prove safe│      │  ✓ T4  ragged · conservative    │
│ ▶4 tensor map     └───────────┘     └─────┬─────┘      │  ✓ T3  skinny  (proof reused)   │
│  5 instructions         ▲                 ▼            │  ✓ T2  square · exact           │
│  6 architecture   ┌───────────┐     ┌───────────┐      │  ✓ T1  square · conservative    │
│  7 microarch.     │ORCHESTRATR│ ◀── │  SOLVER   │      │  ✕ T6  ragged · P≤S−1  bug      │
│                   │ decide    │     │ optimize  │      │  ✕ T7  floor grid       bug     │
│                   └─────▲─────┘     └─────┬─────┘      │                                 │
│                         │           ┌─────▼─────┐      │  NEW cards glow for 1 iteration │
│                         └────────── │ EVALUATOR │      │  reuse links drawn as arrows    │
│          [template token moves]     │ measure   │      │                                 │
│                                     └───────────┘      │                                 │
├────────────────────────────────────────────────────────┴─────────────────────────────────┤
│ Iteration 4 · ragged GEMM, trailing barrier kept                                         │
│ The agent proposed a tiling template for the ragged GEMM and assumed it is memory-bound. │
│ The verifier reused an earlier proof (30 µs). The solver found the best design in 46 ms. │
│ The gate passed. Measured: compute-bound, so the bottleneck moved. The other regime     │
│ can't beat the best design by more than 5%, so we stayed. Accepted into the pool.       │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 The map

- Five stations in a loop: **Formulator** → **Verifier** (prove safe) → **Solver** (optimize) → **Gate** is drawn as the Verifier's second door on the way out of the Solver → **Evaluator** (measure) → **Orchestrator** (decide) → back to the Formulator. Stations never move or resize.
- A **level ladder** on the left shows the seven refinement levels with the current level highlighted. When the bottleneck moves and the orchestrator decides to descend, the template token drops one rung on the ladder. In v1 the sibling template is not available, so the token drops and a note says "next level not yet available" (single backend). When a second template backend exists, the drop is real.
- The **template token** is a small card with the template's short name (T4) and workload. It is the only moving element. It travels station to station along the drawn arrows, pausing at each station long enough to read the station's result.
- Each station shows a **result badge** after the token passes: a verdict in plain words and the time it took. Verifier: "proved safe · 0.24 s" or "proof reused · 30 µs" or "rejected: bug found". Solver: "best design · 46 ms · gap 0%". Gate: "checked · 16 ms". Evaluator: "memory-bound · 1827 µs" or "compute-bound". Orchestrator: "accepted" / "bottleneck moved → stayed" / "rejected".
- A rejection turns the station red, stops the token there, and the scrubber marks the iteration ✕. Nothing continues past a red station; that is the point.

### 2.2 The narration

One paragraph per iteration, under the map, in large type. Written server-side by `agentic/ledger/story.py` (new) from the fields the ledger already holds: the proposal's `rationale` and `refines`, the structure block's `regime`, the verdicts, the timings, the measurement's dominant resource, the regime decision, and the diagnosis. Templates for the sentences are fixed; only the values change, so the paragraph is predictable and skimmable.

Sentence order is always the same: **what was proposed → what the verifier said → what the solver found → what the gate said → what was measured → what the orchestrator decided.** Missing stages are omitted; a rejection ends the paragraph with "Rejected: …" and the cause in one clause.

### 2.3 The scrubber and the transport

- The scrubber is a horizontal track with one tick per iteration, coloured by outcome: green accepted, red rejected, amber when the bottleneck moved. Dragging it sets the current iteration; the map shows that iteration's final state; the narration shows that iteration's paragraph.
- **Step station** advances the token one station. **Step iteration** advances to the next iteration's end. **Run** plays at the chosen speed (1× ≈ one station per 1.5 s; 4×; ∞ = no animation, jump to the end or the next breakpoint).
- In a **recorded run**, the transport moves through the ledger. In a **live run**, the same transport also drives the Controller: Run = resume in observe mode; Step iteration = resume with mode step; Pause = pause. If the architect scrubs backward during a live run, the view detaches from the live edge and a "⇥ back to live" button appears.

### 2.4 Breakpoints

A row of toggle buttons, each a pause point. Defaults on: **bug found** (any refutation) and **bottleneck moved**. Defaults off: **proposed**, **solved**, **accepted**. The architect toggles them by clicking. When the loop reaches a toggled condition it pauses, the station involved pulses, and the narration shows the paragraph so far with "paused here" appended.

Breakpoints map onto the existing Controller: `proposed` and `admitted` are its STEP_CHECKPOINTS; `solved` and `refuted` are existing checkpoint names. "bug found" = `refuted`; "accepted" = `admitted`; "bottleneck moved" is new and needs a `regime_flip` checkpoint (§7).

### 2.5 The pool

A vertical shelf of cards to the right of the map, newest on top. One card per template that has been proposed in this run:

- **Header:** short name (T1…), workload, what it refines (from `refines`).
- **Status chip:** proposed / proved safe / rejected / accepted. Accepted cards are solid; rejected cards are outlined red and dimmed; the current proposal pulses.
- **Best so far:** a ★ on the card whose design is the incumbent for its workload, with the cost.
- **Proof reused:** a small link arrow to the card whose certificate was reused, with "proof reused from T2".
- **Lineage:** a thin line to the parent card.
- **New:** a card that arrived in the current iteration glows for one iteration, then settles.
- Clicking a card opens the inspector (§2.6) for that template.

### 2.6 The inspector

Slides in from the right over the pool. It is the v1 template view, re-ordered to answer the architect's questions in order:

1. **Why** (the structure block, in plain words): "Assumed memory-bound. Move: optimize bandwidth, freeze port count. Obligation: memory stays the bottleneck. Strength: assumed." With the catalog entry's explanation one click away.
2. **What was decided** (the assignment): the design values, cost, bound, gap, solver time.
3. **Was it safe** (the verdicts): the family result and the instance result, each with time. A rejected verdict shows **the bug, step by step**: for a race, the interleaving as a two-row timeline (warp 0 / warp 1) like the paper's Figure 4; for an index defect, the missing or duplicated terms listed.
4. **What was measured:** predicted vs measured, utilizations as two bars, the dominant resource named.
5. **What the orchestrator concluded:** the diagnosis, if any, in its three parts (symptom, cause, response).
6. **Raw:** the contract JSON, collapsed.

The inspector also holds the architect's actions when the loop is paused at a proposal: **edit** (opens the contract JSON), **insert from pool**, and, at a bottleneck-moved pause, **stay / go deeper**. The "change the verdict" action is not offered; the rule from the design doc (correctness obligations gate, for humans too) is shown as a one-line note instead of a disabled button.

---

## 3. Data: what feeds each view

Everything comes from the ledger through the existing API (`/api/state`, `/api/events`, `/api/template/<id>`, `/api/artifact/<id>`, `/api/checkpoints`, `/api/catalog`) plus one new route, `/api/story`, and the existing WebSocket for live events and commands.

| View element | Ledger events (kind) | Artifacts read | Notes |
|---|---|---|---|
| Token position | `proposed`, `family_check`, `solved`, `instance_check`, `measured`, `admitted`, `rejected`, `diagnosis` | — | the sequence of kinds within an iteration *is* the token's path |
| Station badges | same, plus payload fields `verdict`, `checker_time`, `solver_time`, `gap`, `dominant_resource` | `verdict` (for `reused`) | evaluator time: see §7 |
| Level ladder | `request.level`, `split`, `regime_flip` | — | descend = level + 1 when a `split` has a non-null second child |
| Narration | `proposed.refines`, `proposed.structure.regime`, `family_check`, `solved`, `instance_check`, `measured`, `regime_flip`, `regime_bound`, `skip_regime`, `probe_result`, `admitted`, `diagnosis`, `rejected` | `contract.rationale` | composed in `story.py`; cached per iteration |
| Scrubber ticks | one tick per `iteration`; colour from `admitted` / `rejected` / `family_check`+`instance_check` refuted / `regime_flip` | — | from `timeline()` |
| Pool cards | `proposed`, `admitted`, `family_check` (`reused_from`), `human_edit`, `human_inject` | `contract` (scope, refines, parent) | incumbent from `rebuild().incumbents` |
| Inspector | all events whose `artifact` is the contract or points at it | `contract`, `assignment`, `verdict`, `measurement`, `diagnosis` | this is `template_view()` unchanged |
| Breakpoints state | `checkpoint` events; Controller snapshot `waiting` | — | over the WebSocket `controller` message |
| Your actions | `human_edit`, `human_inject`, `human_override`, `human_refused`, `rewind`, `replay` | — | shown on the scrubber as a small person icon on that iteration |

Timings shown, all already in the ledger: `family_check.checker_time`, `instance_check.checker_time`, `solved.solver_time`. Two are missing and are added in §7: the evaluator's time and the formulator's time.

---

## 4. The story module (server side)

`agentic/ledger/story.py`, new. Pure function over the ledger, like `timeline.py`.

```python
def story(ledger, run) -> list[dict]:
    # one entry per iteration:
    # { iteration, title, sentence, outcome: accepted|rejected|in_progress,
    #   flip: bool, level, descended: bool,
    #   visits: [ {station, verdict_text, time_s, ok: bool, artifact} ... ],   # in order
    #   template: {id, short, workload, refines},
    #   pool_delta: {added: [...], reused_from: id|None, incumbent: bool} }
```

Sentence templates (values in braces come from the ledger; clauses are dropped when their event is absent):

- *Proposed:* "The agent proposed {refines} for the {workload} and assumed it is {regime}."
- *Verifier:* "The verifier proved it safe for all designs in {t}." / "The verifier reused an earlier proof ({t})." / "**Rejected:** the verifier found a bug in {obligation}; the failing case is replayed in the inspector."
- *Solver:* "The solver found the best design in {t} (gap {gap})."
- *Gate:* "The gate checked this design in {t}." / "**Rejected:** this design has a bug in {obligation}."
- *Evaluator:* "Measured {cost} µs, {regime}." + (if flip) " The bottleneck moved from {assumed} to {actual}."
- *Orchestrator:* "The assumption is now proven on every feasible design." / "The other regime cannot beat the best design by more than {eps}%, so we stayed." / "Sampled the other regime {n} times, {w} wins; chance of a missed win below {bound}." / "Accepted into the pool." + (if incumbent) " New best so far for the {workload}."
- *Diagnosis:* "The orchestrator concluded: {symptom}. Cause: {source}. Response: {response}."

Short names: templates are numbered T1, T2, … in order of first `proposed` within the run; the map and the pool never show content hashes.

`/api/story?run=…` returns the list; the WebSocket sends an updated story entry whenever an event for the current iteration arrives, so the live narration grows clause by clause.

---

## 5. Components (front end)

Vanilla JS, no build step, as v1. One file each.

| Component | File | Responsibility |
|---|---|---|
| `Transport` | `transport.js` | buttons, speed, breakpoint toggles; sends Controller commands; owns the play clock |
| `Scrubber` | `scrubber.js` | ticks, drag, current iteration; emits `seek(iteration, visitIndex)` |
| `Map` | `map.js` | SVG stations, ladder, token animation, badges; consumes `story[i].visits[j]` |
| `Narration` | `narration.js` | renders `story[i].sentence` with clauses revealed up to visit j |
| `Pool` | `pool.js` | cards from `story[*].pool_delta` plus `rebuild()` summary; incumbent, reuse arrows, glow |
| `Inspector` | `inspector.js` | the six sections of §2.6 over `template_view()`; actions when paused |
| `Session` | `session.js` | WebSocket, `/api` fetches, live/recorded mode, detach/back-to-live |

The **play clock** is the single source of "where are we": `(iteration, visitIndex)`. Scrubber, Map and Narration all render from it. In recorded mode the clock advances on a timer; in live mode it advances as events arrive and the Controller releases.

Styling: the deck's palette (navy `183039`, text `2D4047`, red `C54A38`, teal `5A7D78`, light `F1EFEB`), 16 px minimum body text, 20 px narration. Dark mode optional and not required for acceptance.

---

## 6. Build order and acceptance tests

Each stage has a test that must pass before the next starts. Stages C0–C3 need only the recorded `runs/demo.db`; C4–C5 need a live loop.

| Stage | Deliverable | Acceptance test |
|---|---|---|
| **C0** Story module | `story.py`; `/api/story`; `tests/test_story.py` | On `runs/demo.db` run `gemm-001`, `story()` returns 7 entries; iteration 6's sentence contains "Rejected" and "bug"; iteration 4's contains "bottleneck moved" and "stayed"; iteration 3's contains "reused"; every sentence is under 80 words. |
| **C1** Map + narration, recorded | `map.js`, `narration.js`, `session.js` in recorded mode | Open the console on `gemm-001`; at ∞ speed the map ends on iteration 7 with the Gate red and the narration ending in "Rejected: this design has a bug in coverage". No content hash is visible anywhere on the map. |
| **C2** Scrubber + transport, recorded | `scrubber.js`, `transport.js` (recorded part) | Drag to iteration 4: the Evaluator badge reads "compute-bound", the ladder stays on level 4, the narration says "stayed". Step station from iteration 1 start: the token visits Formulator, Verifier, Solver, Gate, Evaluator, Orchestrator in that order, six presses. |
| **C3** Pool + inspector | `pool.js`, `inspector.js` | Pool shows 7 cards, 5 solid, 2 outlined red; ★ on T5 (ragged, 53.14 µs); T3 shows "proof reused from T2". Click T6: the inspector shows the race as a two-warp timeline. |
| **C4** Live mode + breakpoints | Controller additions (§7); `transport.js` live part | Start a new run in the console with defaults. It pauses at iteration 6 (bug found) with the Verifier pulsing, and at iteration 4 (bottleneck moved) with the Evaluator pulsing. Toggle "proposed" on: the next run pauses at iteration 1 before the Verifier runs. Continue finishes the run. |
| **C5** Your actions | inspector actions wired to edit / inject / regime decision | Pause at iteration 1 proposed; edit the precondition mode to `exact`; continue. The pool shows T1 replaced, the scrubber shows the person icon on iteration 1, and the narration for iteration 1 begins "You changed the proposal". A "change the verdict" attempt is not offered in the UI; sending it over the WebSocket by hand yields `human_refused` and a toast quoting the rule. |

A single HTML page served by the existing `agentic serve`; a static export (`agentic export --ledger runs/demo.db --run gemm-001 --out console.html`) that embeds the story and template views as JSON so a recorded run can be opened without the server is **optional** for round 2 and recommended for sharing runs with Prof. Peh and Prof. Yang.

---

## 7. Changes outside the UI (small, listed in full)

1. **Controller breakpoints.** `Controller.breakpoints: set[str]`, default `{"refuted", "regime_flip"}`; `should_block(name)` returns True in any mode when `name in breakpoints`; new op `breakpoints` over the WebSocket to set them. Mode semantics unchanged.
2. **A `regime_flip` checkpoint.** In `Orchestrator._iterate`, immediately after the `regime_flip` event is appended and before `_handle_regime`, call `self._checkpoint("regime_flip", m)`. This is where the architect's stay/descend decision is taken; today the `regime` override is applied inside `_handle_regime`, which already pops it.
3. **Two timings.** Add `evaluator_time` to the `measured` event payload and `formulator_time` to the `proposed` event payload (both `perf_counter` deltas around the calls). Measurement and contract schemas do not change; the fields live in the event payloads.
4. **`story.py` and `/api/story`** as in §4; the server broadcasts `{"type": "story", "iteration": i, "entry": …}` on each event, after the existing `event` message.
5. **Short names.** `story.py` owns the T-numbering; the inspector header shows "T4 · tmpl_3ae2…" so the hash is still findable.

Nothing else in `agentic/agents/` changes. The replay test (S3) must still pass: the new checkpoint and the two timing fields are stripped by `strip_timing` / excluded from `replay_signature` like the existing ones.

---

## 8. Out of scope for this round

- Editing templates as forms rather than JSON.
- The multi-level descent animation beyond one rung (needs the second template backend).
- EvE race view (two formulators side by side); this is the natural round 3 once `eve_integration.md` is implemented.
- Authentication, multi-user, remote access.

---

## 9. Open questions for Li

1. The default pace at 1×: one station per 1.5 s (an iteration in about 9 s)? Or should the token wait at each station until the architect presses Step station, with Run as the only auto-advance?
2. Should the narration also be read aloud (browser speech synthesis) for a hands-free review? Cheap to add; off by default.
3. For the race replay in the inspector: the two-warp timeline as in the paper's Figure 4, or a plain numbered list of steps? The timeline is better but costs a day.
