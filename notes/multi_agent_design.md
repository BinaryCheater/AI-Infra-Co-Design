# Multi-Agent Infrastructure for Agentic Co-Design

**Design document, v0.1 · October 11, 2026**
*Li Shang, drafted with Claude. Starting point for Ding Yi.*

Companion documents: the unified paper (`unified_paper/paper.md`) defines the template contract and the refinement flow; `notes/search_structure.md` defines the structure catalog, the three buckets, and the gate/steer split. This document says how to turn them into running software with a computer architect in the loop.

---

## 0. One-page summary

We are building a system in which a large model proposes **templates** (formulations of one refinement step), a **solver** optimizes inside them, and a **verifier** certifies them, repeated down the stack from workload to microarchitecture. The paper proves why this is sound; the structure notes say what guides the proposals. What is missing is the machinery, and a way for the architect to see what the machinery is doing.

Six decisions shape the design:

1. **Roles become agents.** Orchestrator, Formulator, Solver, Verifier, Evaluator. Each has one job and one kind of output.
2. **Agents exchange artifacts, not prose.** The template contract $T=(O,X,C,\mathrm{Pre},P,\pi,\Sigma,\hat J,\Gamma)$ is the message. The solver returns an assignment and a bound; the verifier returns a verdict and, on failure, a counterexample. Free text appears only inside a `rationale` field that is stored, displayed and never acted on.
3. **The "why" is a first-class field.** Every template carries a `structure` record: which structure from the catalog it exploits, which bucket, which regime it assumes, and whether that assumption was *proved* or *observed*. The architect audits this field; the orchestrator uses it to decide whether an assumption may gate or only steer.
4. **Three shared stores, one append-only ledger.** The template population, the design archive and the proof library are shared memory. Every event the agents produce is appended to a ledger. The UI is a view over the ledger; nothing on screen comes from anywhere else.
5. **The architect is a participant, not a spectator.** Three modes: *observe* (watch), *step* (approve each iteration), *drive* (pause anywhere, edit a template, replay from a checkpoint). Every agent step is a checkpoint, so all three modes are the same code path with different wait conditions.
6. **Two views.** The *timeline* (one row per iteration) and the *template view* (one contract, three faces side by side, with the why on top).

Build order: ledger and contract schema first, then the verifier and solver as deterministic services, then the orchestrator with a scripted formulator, then the large-model formulator, then the UI. The system should run end to end on the GEMM tiling template, with a scripted formulator, before any large model is called.

---

## 1. Goals and non-goals

**Goals.**

- Run the refinement loop of the paper (§2.2) autonomously on a workload and a chip family, producing certified designs and a growing template population.
- Make every decision the agents take inspectable by a computer architect, at two granularities: per iteration and per template.
- Let the architect intervene: pause, override, inject a template, reject a verdict, replay.
- Reuse the existing assets: the bounded verifier and certificate templates from the verification work, the MILP formulations from the optimization work, and the Ventus co-design harness.

**Non-goals (for v1).**

- A general agent framework. We build the minimum needed for this loop.
- Autonomous proof construction from scratch. In v1 the Formulator *instantiates and adapts* certificate templates; the Verifier checks them. Agent-written certificates for new obligations are v2.
- Hardware-in-the-loop evaluation. Evaluation uses analytic models and simulation (Ventus timing simulation, RTL where available).

---

## 2. Architecture

```
                   ┌─────────────────────────────────────────────┐
                   │              Architect (human)              │
                   │   observe · step · drive  ←→  UI (views)    │
                   └──────────────┬──────────────▲───────────────┘
                                  │ commands     │ ledger stream
┌─────────────────────────────────▼──────────────┴───────────────────┐
│                           Orchestrator                             │
│  owns: refinement cursor (level, current template), run state      │
│  does: select next step · dispatch · diagnose · checkpoint         │
└───┬───────────────┬────────────────┬────────────────┬──────────────┘
    │ request        │ contract       │ contract+assign │ design
┌───▼────────┐  ┌────▼───────┐  ┌─────▼──────┐  ┌──────▼──────┐
│ Formulator │  │   Solver   │  │  Verifier  │  │  Evaluator  │
│ (LLM)      │  │ (MILP/CP)  │  │ (bounded   │  │ (models,    │
│ templates, │  │ assignment │  │  checker)  │  │  simulation)│
│ structure  │  │ + bound    │  │ verdict +  │  │ measured    │
│ rationale  │  │ + diag.    │  │ cex / cert │  │ cost + gap  │
└───┬────────┘  └────┬───────┘  └─────┬──────┘  └──────┬──────┘
    │                │                │                │
┌───▼────────────────▼────────────────▼────────────────▼───────────┐
│                      Shared stores (append-only ledger)          │
│  Template population · Design archive · Proof library · Ledger   │
└──────────────────────────────────────────────────────────────────┘
```

### 2.1 Agents

| Agent | Input | Output | Implementation | Deterministic? |
|---|---|---|---|---|
| **Orchestrator** | run spec, ledger | next action, checkpoints | code (state machine) | yes |
| **Formulator** | refinement request + context (population, evidence, structure catalog) | a template contract with `structure` and `rationale` | large model with tools | no |
| **Solver** | contract (search face) | assignment, bound, gap, diagnostics | MILP/CP solver behind a thin service | yes |
| **Verifier** | contract (correctness face) + assignment | verdict ∈ {proved, refuted, open}; counterexample or certificate | bounded checker (existing) | yes |
| **Evaluator** | decoded design | measured cost, discrepancy vs. predicted | analytic model, Ventus simulation, RTL | yes |

Two principles:

- **Only the Formulator is a large model.** Everything else is code. This keeps the loop's guarantees where the paper puts them: insight from the model, guarantees from solver and verifier.
- **The Orchestrator is a state machine, not a model.** Its decisions (what to refine next, when to split a branch, when to stop) follow explicit rules drawn from the paper's diagnosis table (§5.4) and the structure notes. If we later want a model to make these decisions, it proposes and the state machine disposes.

### 2.2 The Formulator in detail

The Formulator receives a *refinement request*:

```yaml
request:
  level: 4                       # tensor execution mapping
  coarse_template: tmpl_0007     # accepted template one level up
  workload: gemm_1000_1000_1000
  chip_scope: ventus_v2_family
  evidence:                      # from the evaluator and verifier, most recent first
    - kind: bottleneck
      resource: HBM
      utilization: 0.95
      compute_utilization: 0.40
    - kind: counterexample
      template: tmpl_0006
      obligation: lifetime
      trace: ...
  population_hits:               # nearest certified templates by signature
    - tmpl_0003 (shared-memory tiling, scope matches)
    - tmpl_0005 (double buffering, scope: S slots)
```

It must return a **contract** (schema in §3) in which the `structure` block is mandatory. The Formulator has three tools and no others:

- `population.search(signature)` — find candidate templates.
- `structure.catalog()` — the catalog from `search_structure.md` as data: each structure with its bucket, its formulation move, its obligation, and whether the obligation is typically a bound or a trend.
- `cost_model.probe(design)` — a cheap analytic evaluation to detect regimes (e.g. roofline utilization).

The Formulator is **prompted around the catalog**: *detect the structure, name it, state the regime, choose the move, state the obligation and its strength.* The three buckets become the three questions the model must answer before emitting a template:

1. What should be optimized here? (collapse the objective)
2. What is independent here? (collapse the variables)
3. What can be skipped here? (collapse the enumeration)

A template that answers none of these is rejected by the Orchestrator before it reaches the solver.

### 2.3 The Orchestrator loop

```
loop:
  step = select_next_refinement(cursor, ledger)          # physics-ordered: bottleneck first
  T    = Formulator.propose(step)                        # contract with structure+rationale
  checkpoint("proposed", T)                              # architect may edit/replace here

  if not well_formed(T): reject("malformed"); continue
  if T.structure.strength == "observed":
      mark T.assumptions as STEER                        # may reorder, may not prune
  else:
      mark T.assumptions as GATE                         # may prune

  cert = Verifier.certify_family(T)                      # K1–K3 once per template
  if cert.verdict != proved: diagnose(cert); continue    # refuted contract → repair

  (x, bound, diag) = Solver.solve(T)                     # inside C
  checkpoint("solved", x)

  v = Verifier.check_instance(T, x)                      # cheap per-instance gate
  if v.verdict == refuted: diagnose(v.counterexample); continue

  m = Evaluator.measure(decode(T, x))
  if regime_flipped(T.structure.regime, m):              # bottleneck moved
      ledger.append("regime_flip"); split_or_descend()   # this is the refinement signal
  if discrepancy(m, bound) > tol:
      diagnose("model")                                  # revise cost model, not template

  population.admit(T, cert, evidence=m)                  # only certified templates enter
  cursor.advance()
  checkpoint("admitted", T)
```

**Diagnosis** is the paper's table, implemented as a function that maps a symptom to exactly one source (constraints, cost model, search policy, contract) and one response. The response is appended to the ledger *before* it is acted on, so the architect can see what the Orchestrator concluded and why.

**Regime flips are not failures.** When the Evaluator reports that the assumed bottleneck has moved, the Orchestrator records it as the signal to refine one level lower, as described in the structure notes (§6).

---

## 3. The contract as the message

Everything between agents is a typed artifact. The canonical one is the contract. Schema (YAML for readability; stored as JSON):

```yaml
contract:
  id: tmpl_0008
  parent: tmpl_0007                 # the coarse template this refines
  level: 4
  object: O                         # the new object introduced at this level
  scope: Γ                          # workloads and chip parameters it applies to
  signature: ...                    # structural hash for population search

  # --- search face (read by the Solver) ---
  decisions: X                      # variables with domains
  constraints: C                    # e.g. C3: P <= S - 2 + beta
  cost: Ĵ                           # predictive cost model
  decomposition: ...                # optional: Benders split, clusters

  # --- correctness face (read by the Verifier) ---
  preconditions: Pre                # must be implied by C (K1)
  program: P                        # next-level program
  backward_map: π
  certificate_schema: Σ             # relation R, rank ρ, obligations

  # --- why (read by the Architect and the Orchestrator) ---
  structure:
    name: bottleneck_dominance
    bucket: objective               # objective | variables | enumeration
    regime: "HBM-bound: util(HBM) >= 0.9 and util(compute) <= 0.6"
    move: "surrogate objective = HBM utilization; froze {port_count, tensor_shape}"
    obligation: "HBM stays dominant across feasible region of this branch"
    strength: observed              # proved | observed  →  gate | steer
    evidence: [probe_0142]
  rationale: |                      # free text, stored and shown, never parsed
    Roofline probe shows HBM at 95% ... chose tiling template tmpl_0003 as base ...

  # --- status (written by the Orchestrator) ---
  status: proposed                  # proposed | certified | solved | admitted | refuted
  certificate: cert_0008            # after K1–K3
  provenance: {formulator_run: ..., prompt_hash: ..., model: ...}
```

Other artifacts:

- **Assignment**: `{contract, values of X, bound, gap, solver_time, diagnostics}`.
- **Verdict**: `{contract, assignment?, verdict, counterexample?, certificate?, checker_time}`.
- **Measurement**: `{design, predicted_cost, measured_cost, utilizations per resource, simulator, seed}`.
- **Diagnosis**: `{symptom, source, response, target_artifact}`.

Every artifact has an `id`, a `parent` chain and a `provenance` block. That is what makes the ledger replayable.

---

## 4. Shared stores

All three stores are materialized views over one **append-only ledger** of events. Agents never mutate; they append. This gives replay, audit and the "why" trail for free.

| Store | Holds | Key operations |
|---|---|---|
| **Template population** | certified contracts with scope, signature, evidence, lineage | `search(signature, scope)`, `admit(T, cert)`, `lineage(T)` |
| **Design archive** | assignments and decoded designs, incumbents per (workload, chip, objective) | `incumbent(key)`, `dominated(design, context)` (Proposition 4: dominance stored *with context*) |
| **Proof library** | certificates, lemmas, obligations discharged, dependencies | `reuse(obligation, context)` — the incremental reuse of paper §4.6 |

Storage: v1 uses a single SQLite file with an `events` table plus three view tables, and artifacts as JSON blobs on disk. This is enough for one run at a time and is trivially replayable. Move to Postgres only if runs become concurrent.

---

## 5. The architect in the loop

The architect is a domain expert who needs to audit the reasoning, not just read numbers. Three modes, one mechanism:

| Mode | Behavior | Who uses it |
|---|---|---|
| **Observe** | Loop runs freely; UI streams the ledger live. | Long runs, demos. |
| **Step** | Loop pauses at every `checkpoint("proposed")` and `checkpoint("admitted")`; architect approves, edits or rejects. | Early development; auditing a new workload. |
| **Drive** | Loop pauses at any checkpoint on demand; architect can edit a contract, inject one from the population, overrule a verdict (recorded as `human_override`), or rewind to any checkpoint and replay. | Debugging a wrong assumption. |

Design rules:

- **Checkpoints are complete states.** A checkpoint stores the cursor, the pending artifact, and the ledger offset. Replaying from it re-executes only deterministic agents; Formulator outputs are replayed from the ledger unless the architect asks for a fresh proposal.
- **Overrides are events.** When the architect edits a template, the ledger records both versions and who changed what. The population never contains an unaudited human edit presented as agent output.
- **The architect cannot bypass the gate.** A human may change a template; the Verifier still has to certify it. "Correctness obligations gate" applies to humans too.

Interface to the loop: a small command set — `pause`, `resume`, `step`, `edit <artifact>`, `inject <template>`, `override <verdict> reason=...`, `rewind <checkpoint>`, `replay`. The UI is a client of these commands; so is a CLI, which is what Ding Yi will use first.

---

## 6. The two views

### 6.1 Timeline (per iteration)

One row per iteration of the loop, newest at the bottom:

| # | Level | What was refined | Structure (bucket) | Regime | Solver: bound / gap / time | Gate | Measured vs predicted | Outcome |
|---|---|---|---|---|---|---|---|---|
| 12 | 4 | tiles → slots | bottleneck (objective) | HBM-bound | 0.231 / 0.3% / 0.16 s | proved | 0.236 (+2%) | admitted |
| 13 | 4 | prefetch distance | bottleneck (objective) | HBM-bound | 0.228 / 0.1% / 0.09 s | **refuted** | — | repair |
| 14 | 5 | slots → bank requests | near-decomp. (variables) | **flip → SMEM** | … | proved | … | descend |

Colors: gate verdicts in green/red; regime flips highlighted; `strength: observed` shown with a hollow marker, `proved` with a filled one. The row is a link into the template view.

### 6.2 Template view (per contract)

One screen, three faces side by side, with the **why** across the top:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ WHY  bottleneck_dominance · bucket: objective · regime: HBM-bound        │
│      move: surrogate = HBM util; froze {port_count, tensor_shape}        │
│      obligation: HBM dominant across branch · strength: OBSERVED → steer │
│      rationale: "Roofline probe shows HBM at 95% ..."     [probe_0142]   │
├──────────────────────┬──────────────────────┬────────────────────────────┤
│ FORMULATION          │ OPTIMIZATION         │ VERIFICATION               │
│ X, C, Ĵ              │ assignment x*        │ K1–K3: proved (cert_0008)  │
│ C3: P ≤ S−2+β        │ bound / gap          │ instance: proved, 25 ms    │
│ decomposition        │ solver diagnostics   │ or: counterexample trace   │
│ lineage: tmpl_0003 → │ incumbent comparison │ obligations reused: 3/5    │
└──────────────────────┴──────────────────────┴────────────────────────────┘
│ EVIDENCE  measured 0.236 vs predicted 0.231 · utilizations · simulator   │
│ ACTIONS   edit · inject · override · rewind to proposed                  │
└──────────────────────────────────────────────────────────────────────────┘
```

The counterexample, when present, is shown as the replayed interleaving (as in the paper's Figure 4), not as a solver log.

### 6.3 Implementation

v1: a web UI (React or Svelte) reading the ledger over a WebSocket, with the command set of §5 as buttons. No design-system work yet; correctness of what is shown matters more than polish. The timeline is a table; the template view is three columns. Ding Yi should be able to build this from the ledger alone, which is the test that the ledger is complete.

---

## 7. Build order

Each stage has a test that must pass before the next begins.

| Stage | Deliverable | Test |
|---|---|---|
| **S0** Schema and ledger | JSON schemas for all artifacts; SQLite ledger; CLI to append and query | Hand-write the GEMM tiling contract (paper Listing 1) as JSON; it validates; the three stores can be rebuilt from the ledger. |
| **S1** Verifier service | Wrap the existing bounded checker: `certify_family(T)`, `check_instance(T, x)` | Reproduces 24/24 on the tiling template; returns the replayed race for the unsound precondition. |
| **S2** Solver service | Wrap the MILP: `solve(T)` | Reproduces the 93-variable GEMM solve in ≤ 0.2 s with the same optimum. |
| **S3** Orchestrator + scripted Formulator | State machine; Formulator replaced by a script that replays the paper's five-step GEMM progression | Full loop runs end to end on GEMM, five levels, no model calls; ledger is complete; replay from any checkpoint reproduces the run byte-for-byte. |
| **S4** Evaluator | Analytic roofline model + Ventus simulation adapter | Regime detection on the Qwen workload reproduces the HBM-bound / compute-bound split from the co-design study. |
| **S5** Large-model Formulator | Prompted around the structure catalog, with the three tools of §2.2 | On the GEMM run, proposes templates that the Verifier certifies ≥ 80% of the time; every proposal carries a well-formed `structure` block. |
| **S6** UI | Timeline + template view + commands | The architect can find, from the timeline alone, the iteration where a regime assumption was wrong, open it, and see the counterexample. |
| **S7** Drive mode | Checkpoints, rewind, replay, override events | Edit a template mid-run, replay, and show both lineages in the population. |

S0–S3 are the foundation and are pure engineering; they can start now. S5 is where the research is.

---

## 8. Technology choices (recommendations, not decisions)

- **Language:** Python for the Orchestrator, services and adapters; the existing verifier and MILP code are already Python.
- **Solver:** keep what the optimization work uses (OR-Tools / Gurobi / HiGHS behind one interface). The contract's search face must be solver-agnostic.
- **Verifier:** the existing bounded checker, exposed as a library with two entry points. SMT backend unchanged.
- **Large model:** any; access through one `propose(request) -> contract` function so models can be swapped and compared. Log prompt hash and model id in `provenance`.
- **Ledger:** SQLite + JSON files, as above.
- **UI:** web; WebSocket for the live stream.
- **Repo layout** (proposal for `AI-Infra-Co-Design`):

```
agentic/
  schema/        # JSON schemas for contract, assignment, verdict, measurement, diagnosis
  ledger/        # append, query, rebuild views, checkpoints
  agents/
    orchestrator.py
    formulator/  # llm.py, scripted.py, prompts/, catalog.yaml
    solver/      # service wrapping the MILP
    verifier/    # service wrapping the bounded checker
    evaluator/   # roofline.py, ventus.py
  cli/           # run, step, inspect, replay
  ui/            # timeline, template view
  examples/
    gemm_tiling/ # the paper's Listing 1 as the S0 fixture
```

---

## 9. How the structure catalog enters the system

The catalog is **data**, not prompt prose. `catalog.yaml` lists each structure with:

```yaml
- name: bottleneck_dominance
  bucket: objective
  detect: roofline probe; resource with max utilization/ceiling ratio
  move: surrogate objective = that resource's utilization; freeze non-touching variables
  obligation: resource stays dominant across the branch
  default_strength: observed        # becomes proved only with a one-sided bound
  verification_cost: expensive
- name: monotonicity
  bucket: objective
  move: one-sided bounds for pruning
  default_strength: proved
  verification_cost: cheap
...
```

Three consumers read it:

1. The **Formulator**, as the vocabulary it must use in `structure`.
2. The **Orchestrator**, to map `strength` to gate/steer and to set the verification budget.
3. The **UI**, to render the why consistently.

Adding a structure means adding a catalog entry, not changing code. This is also how the list stays open, as the structure notes require.

---

## 10. What self-improvement means here

The population already compounds: each admitted template makes the next problem easier. The multi-agent version adds a second, grounded loop:

- The ledger records, for every template, which structure was claimed, whether its assumption was *observed* or *proved*, and whether it later held.
- Periodically, a **curator** job (code, not a model) computes which structure–regime pairs reliably hold for which workload families, and promotes recurring `observed` assumptions to candidates for proved bounds.
- The Formulator's context includes these statistics. It gets better at *recognizing* structure, which is the sharp version of self-improvement from the discussion.

Because nothing enters the population without passing the gate, this improvement rests on proofs and measurements, not on the model's own judgment.

---

## 11. Open questions for Ding Yi

1. **Granularity of checkpoints.** Per agent call (fine, expensive to store) or per iteration (coarse, cheaper)? Recommendation: per agent call, with artifacts stored by content hash so repeated states cost nothing.
2. **Formulator output format.** Should the model emit the contract JSON directly, or emit a structured plan that code expands into a contract? Recommendation: start with the latter (plan → code → contract); it is easier to validate and to keep K1 (C ⇒ Pre) true by construction.
3. **Regime detection threshold.** What utilization ratio counts as "bound"? Needs calibration on Ventus.
4. **Branch splitting.** When a branch straddles a regime boundary, split on which variable? Start with the one with the largest effect on the bottleneck resource in the cost model.
5. **Human override semantics.** When the architect overrules a *performance* verdict (steer), the loop continues. When they attempt to overrule a *correctness* verdict (gate), the system refuses. Confirm this is the behavior we want.
6. **Concurrency.** v1 is one run at a time. Does the population need to be shared across concurrent runs on different workloads? If so, S0 should use Postgres from the start.
