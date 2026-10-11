# Integrating EvE with the Agentic Co-Design Infrastructure

**Design document, v0.1 · October 12, 2026**
*Li Shang, drafted with Claude. Companion to `multi_agent_design.md`; that document is unchanged and remains the reference for what Ding Yi is building.*

Related: EvE paper (Yu & Yang, arXiv 2605.09018) and repository (`scaling-group/eve`, Apache-2.0), from Prof. Liu Yang's Scaling Group at NUS.

---

## 0. Summary

`multi_agent_design.md` specifies a loop in which a Formulator (the only large-model agent) proposes templates, a Solver optimizes inside them, a Verifier certifies them, and an Orchestrator walks the refinement levels. It also specifies a population of templates and says, in §10, that the population should improve over time. It does not say *how* the population is managed: how candidates are sampled, how competing proposals are compared, or how credit for an improvement is assigned to the proposal that caused it.

EvE answers exactly those questions, in general form and with working code. It keeps a fixed base coding agent and evolves only the guidance that steers it; it maintains two scored populations; it races agents on identical reference sets so that differences in outcome are attributable to the agent; and it assigns credit by Elo. None of this is domain-specific, and none of it should be rebuilt.

**The integration in one sentence:** EvE manages the Formulator and only the Formulator; everything downstream of the Formulator (Solver, Verifier, Evaluator, Orchestrator) stays fixed, deterministic, and outside EvE's reach, and is exposed to EvE as the evaluation that scores a proposal.

Three things are plugged into EvE:

1. **The Verifier as a mandatory evaluation step.** A proposal that fails K1–K3 or the instance gate receives EvE's failure score and never enters the population. This is the property EvE lacks on its own: its fitness is purely empirical.
2. **A two-part fitness.** Certified first, then measured cost. In EvE terms, a vector score with a gate component and a cost component.
3. **The structure catalog as the guidance tree.** EvE's evolvable guidance (problem context, search directions, editable boundary, literature notes, one evaluation-reading skill) becomes our catalog, our Formulator prompts and our evaluation-reading skill. What EvE evolves is therefore how the Formulator recognizes and exploits structure.

The race gives us something the existing design did not have: a controlled measurement of how much each structure contributes, because two Formulators that start from the same references and produce different templates differ only in formulation.

---

## 1. EvE in our vocabulary

| EvE term | Meaning in EvE | Our object |
|---|---|---|
| Base agent | fixed coding agent (Codex + GPT-5.4) | the Formulator's model; fixed |
| Optimizer / agent population | guidance trees with logs and Elo | Formulator guidance: `catalog.yaml`, prompts, read-eval skill; one entry per guidance version |
| Solver population | editable code files in a base repo, with score | template contracts (and the designs they produced); one entry per template |
| Base repository | the shared code both populations refer to | the `agentic/` package plus `examples/` fixtures (the paper's Listing 1 template) |
| Editable files / boundary | the five files an agent may change | the template directory: `contract.yaml`, `structure.yaml`, `rationale.md`; nothing else |
| Evaluation steps | ordered shell scripts and/or judge agents that write a score | step 1 Verifier (family), step 2 Solver + instance gate, step 3 Evaluator |
| Score (PyTree) | arbitrary structured score | `{certified, cost, gap, regime_strength, components...}` |
| Elo evaluator | `ScalarEloEvaluator` / `VectorEloEvaluator`, K = 32 | vector Elo on the certified-cost component |
| Phase 2 (race) | sample agents, run in isolated workspaces on the same references, evaluate, build win/loss matrix | the per-structure credit-assignment race |
| Phase 3 (integration) | revised agents join the population with logs | admitted templates and revised guidance join the population |
| Rank-softmax sampler | sample by Elo rank | retrieval from the template population, made regime-aware (§4) |
| Working logs | cumulative session notes, eval artifacts, failures | our ledger entries for the Formulator |

Terminology note for the merged system: we keep *template* for the individual and *guidance* for what steers the Formulator. EvE's internal name for the agent population ("optimizer") collides with our Solver; in our code and documents, "optimizer" always means the MILP.

---

## 2. The boundary

```
┌────────────────────────────── EvE manages ──────────────────────────────┐
│  Formulator guidance (catalog, prompts, read-eval skill) · sampling ·   │
│  race · Elo · population bookkeeping · lineage · resume                 │
│                                                                         │
│        Formulator ──proposes──▶ template directory (editable files)     │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ EvE evaluation steps (fixed, ours)
┌──────────────────────────────────▼──────────────────────────────────────┐
│  step 1  Verifier.certify_family(T)            → certified? cex?        │
│  step 2  Solver.solve(T); Verifier.check_instance(T, x) → x, bound, gap │
│  step 3  Evaluator.measure(decode(T, x))       → cost, utilizations     │
│  score   {certified, cost, gap, regime_strength, ...}                   │
└─────────────────────────────────────────────────────────────────────────┘
          Solver, Verifier, Evaluator: deterministic, never evolved
```

What EvE may change: the guidance tree, and through it the templates the Formulator writes.
What EvE may not touch: the Verifier, the Solver, the Evaluator, the cost model, the ledger schema. These are the trusted base of the paper (§4.5), and the whole point of the design is that insight comes from the model while guarantees come from code.

The Orchestrator from `multi_agent_design.md` §2.3 does not disappear. Its refinement cursor, regime handling (§2.4) and checkpoints remain. What changes is that "ask the Formulator for a template" becomes "run one EvE iteration on this refinement step", and the population operations (`search`, `admit`) are backed by EvE's populations rather than our own SQLite views.

---

## 3. The three plug-ins

### 3.1 Verifier as a mandatory evaluation step

EvE evaluates a candidate with an ordered list of steps, each a shell script that writes a score file or a judge agent. We define three steps, and the first is the gate:

```yaml
evaluation:
  steps:
    - configs/eve/evaluation/codesign/01_verify_family.sh     # K1–K3; writes certified: true/false, cex
    - configs/eve/evaluation/codesign/02_solve_and_gate.sh    # MILP inside C; instance check; bound, gap
    - configs/eve/evaluation/codesign/03_measure.sh           # analytic model or Ventus; cost, utilizations
  failure_score:
    score: {certified: false, cost: inf}
    summary: "verification failed"
```

If step 1 fails, the remaining steps do not run, the candidate receives the failure score, and it is never added to the population. A counterexample is written to the workspace log so the next Formulator can read it. This is the "admission requires certification" rule of the paper (§2.5) implemented as EvE configuration. EvE's existing post-generation checks (boundary, import, smoke) run before step 1 and reject malformed templates for free.

### 3.2 Two-part fitness

EvE's `VectorEloEvaluator` accepts a structured score and updates Elo per component. We use:

```yaml
optimizer:
  evaluation:
    _target_: scaling_evolve.algorithms.eve.populations.evaluators.elo.VectorEloEvaluator
    k_factors: {cost: 32.0}           # Elo is driven by cost among certified candidates only
    initial_score: {elo: 1500.0}
```

with a comparison rule: a certified candidate beats an uncertified one unconditionally; among certified candidates, lower measured cost wins; ties on cost are broken by smaller gap, then by `regime_strength` (`proved` over `observed`). Only the cost component drives Elo, but the gate is applied first, so a Formulator cannot gain rating by proposing fast designs that are not correct. This is "correctness obligations gate, performance obligations steer" (paper §4.7) expressed as a fitness.

### 3.3 The catalog as the guidance tree

EvE's initial guidance for the ICON task is four documents and one skill. Ours maps one to one:

| EvE guidance file | Ours |
|---|---|
| `docs/problem.md` | the refinement request: level, coarse template, workload, chip scope |
| `docs/directions.md` (a menu of search families) | `catalog.yaml`: the structures, their buckets, moves, obligations and strengths |
| `docs/mutation_surface.md` (editable files) | the template directory and the contract schema |
| `docs/literature_notes.md` | population notes: which structure–regime pairs have held, flipped, or been promoted |
| `skills/read-eval/SKILL.md` | our read-eval skill: how to read a verdict, a counterexample, a bound and a measurement, and how to turn them into guidance |

Everything in this tree is evolvable by EvE. That is the intended effect: over many races the catalog's *use* (which structure to try first in which regime) improves, and the notes accumulate the regime map that `multi_agent_design.md` §10 asked for. The catalog's *entries* may also be revised by the Formulator, but a new or changed structure is just guidance until a template that uses it is certified; the gate stays the arbiter.

---

## 4. What the race gives us

**Per-structure credit assignment.** In an EvE iteration, two Formulators receive the same reference set (the same 8 templates, 4 guidance versions and the base repo) and the same refinement request. Each proposes a template. Both go through the same three evaluation steps. Because everything except the formulation is held fixed, the difference in certified cost is attributable to the formulation, and through the `structure` block, to the structure each Formulator chose to exploit. Over many races this yields the measurement the paper lists as missing (§8): how much each structure shrinks the search, as solver time and nodes, and how often its regime assumption held.

**Phase mismatch is a regime effect.** EvE's central ablation is that a frozen best-evolved agent does not beat the seed agent in a fresh search, because its guidance was tuned for a late-stage population. In our terms, a template that was right when the design was HBM-bound is wrong once the bottleneck has moved. Two consequences:

- Retrieval must be regime-aware. The sampler that draws reference templates for a race should condition on the current regime (from `structure.regime` and the Evaluator's last utilizations), not on Elo alone. EvE's samplers are pluggable; we add a `RegimeAwareSampler` that filters by regime before rank-softmax.
- Never freeze. The population keeps evolving per run; there is no "best template" oracle. This matches the paper's scope rule Γ and is now empirically motivated.

**A second kind of evidence for the ledger.** EvE's working logs are the Formulator's own account of what it tried and why. They slot into our ledger as `formulator_log` events, and the UI's template view (`multi_agent_design.md` §6.2) shows them under the rationale.

---

## 5. Impact on the build stages

Stage numbers refer to `multi_agent_design.md` §7. The right column is what Ding Yi keeps, changes or drops.

| Stage | Before | With EvE |
|---|---|---|
| **S0** Schema and ledger | JSON schemas; SQLite ledger; three store views | **Keep** the schemas and ledger. The population views become adapters over EvE's `solver_population` and `optimizer_population`; keep the ledger as the source of truth for the UI. |
| **S1** Verifier service | `certify_family`, `check_instance` | **Keep unchanged.** Additionally wrap each as an EvE evaluation step (shell script writing a score file). |
| **S2** Solver service | `solve(T)` | **Keep unchanged.** Wrap as evaluation step 2 together with the instance check. |
| **S3** Orchestrator + scripted Formulator | state machine; scripted five-step GEMM | **Keep the state machine and the scripted test.** Replace the "propose" call by an EvE iteration whose worker is the scripted Formulator. The S3 pass test (byte-for-byte replay) must still hold; EvE's `enable_resume` and lineage snapshots are the mechanism. |
| **S4** Evaluator | roofline + Ventus adapter; optimism record | **Keep unchanged.** Wrap as evaluation step 3. The optimism record is read by the Orchestrator's regime handler, not by EvE. |
| **S5** Large-model Formulator | prompted around the catalog; three tools | **Replace the agent harness with EvE's worker.** The Formulator becomes an EvE agent (Codex or OpenCode backend) whose guidance tree is the catalog and prompts. The three tools (`population.search`, `structure.catalog`, `cost_model.probe`) become files in the immutable scaffold plus a `check_runner`. |
| **S6** UI | timeline + template view | **Keep.** Add Elo and race outcome to the timeline row; add the Formulator's working log to the template view. |
| **S7** Drive mode | checkpoints, rewind, replay, override | **Keep.** Human edits enter as a template candidate that still passes evaluation steps 1–3; an architect-injected template receives initial Elo like any other. |
| **New S8** Race harness | — | Configure a two-worker race on the GEMM fixture; verify that two scripted Formulators with different structure choices receive different Elo for the right reason. |

The practical consequence: S0–S4 and S6–S7 are unaffected or only wrapped. Only S5 changes substantively, and it changes toward *less* code, because EvE already has the agent harness, workspace isolation, boundary checks and sampling.

---

## 6. Repository layout (proposal)

```
agentic/                      # unchanged from multi_agent_design.md §8
  schema/ ledger/ agents/ cli/ ui/ examples/
eve_codesign/                 # new: the EvE application for co-design
  configs/eve/
    application/codesign.yaml         # base repo, editable template dir, boundary
    evaluation/codesign.yaml          # the three steps (§3.1)
    evaluation/codesign/01_verify_family.sh
    evaluation/codesign/02_solve_and_gate.sh
    evaluation/codesign/03_measure.sh
    optimizer/codesign.yaml           # VectorElo, worker scaffold (§3.2)
    optimizer/codesign/initial_guidance/
      docs/problem.md directions.md mutation_surface.md literature_notes.md
      skills/read-eval/SKILL.md
    optimizer/codesign/immutable/     # catalog.yaml, contract schema, check_runner
    optimizer/codesign/prompt/        # ENTRYPOINT.md, BOUNDARY_REPAIR.md, budget
    loop/codesign.yaml                # 2 workers, 8 reference templates, regime-aware sampler
    codesign.gemm.yaml                # experiment: the GEMM fixture
    codesign.ventus.yaml              # experiment: Qwen on Ventus
  samplers/regime_aware.py            # RegimeAwareSampler (§4)
```

EvE is consumed as a dependency (`uv add` the package or a git submodule), not vendored. Changes we need in EvE itself (the regime-aware sampler, possibly a vector-score comparison hook) should be proposed upstream to Prof. Yang's group rather than forked.

---

## 7. Questions for Prof. Yang's group

1. **Fixed downstream.** EvE evolves everything the agent can touch. We need the Solver, Verifier and Evaluator fixed and the agent's edits confined to the template directory. The boundary checker appears to give us this; is there any path by which guidance evolution can alter evaluation steps?
2. **Vector fitness with a hard gate.** Does `VectorEloEvaluator` support a lexicographic comparison (gate first, then cost), or should we implement the comparison in a custom evaluator?
3. **Regime-aware sampling.** The samplers are pluggable; is there a preferred extension point for conditioning sampling on a candidate attribute (our regime) rather than on score alone?
4. **Reference-set size and race width.** EvE uses 2 workers and 8 solver references per race. Our templates are smaller than ICON code files and our evaluation is faster (sub-second solve, ~25 ms gate, minutes for simulation). Would wider races (4 workers) be appropriate, and does the Elo update assume pairwise races?
5. **Logs as evidence.** EvE keeps working logs on the agent entry. We want them in our ledger as first-class events. Is there a hook on log write, or should we tail the workspace logs?
6. **Compute.** ICON runs used 35–40M equivalent tokens per run. Our per-iteration cost is dominated by simulation, not tokens. Any experience with evaluation steps that run minutes rather than seconds?

---

## 8. Risks

- **Two sources of truth.** EvE's populations and our ledger could drift. Rule: the ledger records every EvE population event; the UI reads only the ledger. Enforce with a single adapter module.
- **Evolving the catalog away from physics.** EvE can revise `directions.md`; a Formulator could learn to omit a structure. Mitigation: the catalog's entries are versioned in the immutable scaffold, and guidance may add emphasis but not delete entries; the gate makes a bad emphasis cost Elo.
- **Elo on noisy measurements.** Simulation noise could flip pairwise outcomes. Mitigation: ties within measurement tolerance count as draws (S = 0.5), which EvE supports.
- **Agent backend.** EvE defaults to Codex; our Formulator may use another model. OpenCode is supported; confirm the backend before S5.

---

## 9. Next steps

1. Ding Yi: continue S0–S4 as specified; nothing there changes.
2. Run EvE's `circle_packing.smoke` to see the race mechanics end to end (one afternoon).
3. Write the three evaluation-step scripts against the S1/S2/S4 services and the GEMM fixture; this is S8 and can be done as soon as S1–S2 exist.
4. Send §7 to Prof. Yang's group and agree on the extension points before S5.
