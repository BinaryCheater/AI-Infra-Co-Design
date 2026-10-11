# Exploitable Structure in Hardware–Software Co-Design Search

*Brainstorm notes, October 11, 2026. Li Shang with Ding Yi; drafted from the discussion with Claude.*

## The question

Given a workload and a library of hardware components, the joint hardware–software design space is combinatorially huge. Yet expert designers search it effectively. Why?

Because the problem is not an arbitrary combinatorial problem. It has structure, and most of that structure comes from physics.

**Our hypothesis.** The agent's job is to recognize that structure and write it into the problem formulation. Each structure it exploits removes a large part of the search space. The verifier then checks that the structure really holds in the region where the agent relies on it.

This note records three results from the discussion:

1. **A catalog of structures** that reduce search complexity.
2. **Three buckets** that organize the catalog by what each structure does to the search: it collapses the objective, the variables, or the enumeration.
3. **A split between two kinds of verification obligation:** *correctness obligations gate; performance obligations steer.*

A worked example, the HBM bottleneck, runs through all three.

---

## 1. Where the structure comes from

The basic fact of computer systems is physical:

- Memory can be large, but large means far away, and far means slow and costly to reach.
- Computation is fast, but it is useless without data.
- Moving data is physics, and the designer cannot change physics.

This is why the memory hierarchy exists, and why it is a fixed ladder of cost rather than a free design choice:

**HBM → L2 → shared memory → registers.**

HBM is roughly an order of magnitude worse than the level above it in bandwidth, latency and energy per byte.

Two consequences follow:

- **Data movement dominates cost.**
- **It dominates in a known order.**

Much of the structure below is a direct consequence of these two facts.

---

## 2. Catalog of structures

### 2.1 Bottleneck dominance

When one resource is the bottleneck, it is close to a *sufficient statistic* for performance.

**Example.** If a design is HBM-bound, compare two partial solutions (two branches) by HBM bandwidth utilization. The one with better utilization usually leads to the better design. HBM is so much more expensive than everything above it that improvements elsewhere barely register.

**Effect on search.** Within a regime, many objectives collapse to one tight objective. "Optimize this one bound, nothing else, for now." Branch-and-bound starts to behave like dynamic programming, with optimal substructure keyed on the bottleneck metric.

**Caveat.** This holds only while the resource stays the bottleneck. Once HBM is relieved, the bottleneck moves down the hierarchy, for example to shared memory or the register file. The ordering of branches can then flip. The same argument applies again one level down.

### 2.2 Near-decomposability (Herbert Simon)

Interactions across the stack are mostly local. The register file size couples strongly to tiling but barely to the HBM channel count. The interaction matrix is roughly block-banded, not dense.

**Concrete form: the hierarchy cuts the chain.** When shared memory exists, data does not normally flow from HBM straight into registers; it is staged through shared memory. A well-used machine is one whose shared memory delivers everything the compute units need. Shared memory therefore *separates* HBM from the registers:

- **Without the separator:** a three-way coupled search over A (HBM), B (shared memory) and C (registers).
- **With it:** two pairwise problems, A–B and B–C, coupled only through B.

**Effect on search.** Solve tightly coupled clusters jointly, and pay only for the weak coupling between them. This removes the cross-product explosion across distant layers.

### 2.3 Concavity and diminishing returns

Performance as a function of any single resource saturates. This is the roofline knee: performance rises roughly linearly with bandwidth until the design becomes compute-bound, then flattens. Beyond that point, extra bandwidth buys nothing.

**Effect on search.** For a concave, saturating function, the optimum lies on a constraint boundary or near a knee, and there are few knees. Search the knees, not the continuum.

### 2.4 Scenario decomposition (Benders structure)

Prefill and decode share one chip but want different software. The chip is the *complicating variable*. Once it is fixed, the software for each scenario is an independent and much easier subproblem.

**Effect on search.**

- Fix the hardware, then optimize each scenario separately, each again starting from its dominant bottleneck.
- Feed the results back to the hardware choice as cuts.

This structure is already present in the paper's co-design studies. It is worth naming explicitly as a complexity reducer.

### 2.5 Monotonicity

At fixed area, more of a resource never hurts performance. For example, a larger buffer never makes a fixed schedule slower.

**Effect on search.** Monotonicity gives *valid one-sided bounds*, which is what branch-and-bound needs to prune a branch while staying exact.

### 2.6 Low effective dimension

Once the bottleneck is fixed, only the few variables that touch it matter. For example, in an HBM-bound region the relevant knobs are tiling, what lives in shared memory and the transfer schedule. Register port count and tensor-unit shape are nearly free.

**Effect on search.** A problem over 20 coupled knobs becomes a problem over the 3 or 4 that move bytes. The remaining knobs can be frozen or coarsened.

### 2.7 Symmetry

Many configurations are permutation-equivalent: identical units, interchangeable banks, mirrored layouts.

**Effect on search.** Symmetry breaking removes large factorial blocks of the space at almost no cost.

### 2.8 Tiny discrete menus

Each decision has a small discrete menu: powers of two, a handful of tile sizes, a few precisions.

**Effect on search.** Domains are small and enumerable. Combined with symmetry and dominance, this keeps the leaves of the search tree few.

### 2.9 Dominance (thin Pareto front)

Most designs are dominated: some other design is at least as good on every metric. The Pareto front is thin.

**Effect on search.** Skip dominated designs without evaluating them.

### 2.10 Further candidates (not yet discussed)

We agreed the list is not closed. These candidates are worth examining next:

- **Repetition across layers.** Transformer layers are structurally identical. Solve once, then replicate, adjusting only at the boundaries.
- **Affine loop structure.** Tensor programs are affine loop nests. Reuse distances, footprints and transfer volumes have closed forms, as in polyhedral analysis, so the cost model is cheap and exact for these terms.
- **Regime boundaries as a small set.** The bottleneck changes only at a few points in the space. The space may therefore split into a small number of regimes, each simple inside.

---

## 3. The three buckets

The catalog is easier to use when it is organized by *what each structure does to the search*. The exact members will change; the buckets should not.

| Bucket | What it tells the agent | Structures |
|---|---|---|
| **Collapse the objective** | *What* to optimize | Bottleneck dominance; concavity (optima at knees); monotonicity (valid bounds) |
| **Collapse the variables** | *What* is independent | Near-decomposability (hierarchy cuts the chain); scenario decomposition (Benders); low effective dimension |
| **Collapse the enumeration** | *Which points* to skip | Symmetry; dominance (thin Pareto front); tiny discrete menus |

---

## 4. From structure to template

This is the link to the framework. The large-model agent maintains the pools of problem formulations, templates and solutions. The structures should drive how it generates templates.

**Each structure maps to two things at once:**

1. **A move in the formulation (optimizer side).** The agent writes the structure into the MILP as one of the following:
   - a surrogate objective;
   - a decomposition;
   - a bound;
   - a frozen variable;
   - a symmetry-breaking constraint.
2. **An obligation for the verifier.** A structure holds only in a *regime*, for example "this region is HBM-bound". The agent therefore owes evidence that the regime holds. That evidence becomes part of the template's precondition.

**The structure is the template.** A template captures a structure the agent has recognized, the formulation move that exploits it, and the regime in which that move is valid.

Not all obligations cost the same to discharge:

- **Cheap: symmetry, monotonicity, menus, integrality.** These are algebraic. They are proved once per template and reused.
- **Expensive: near-decomposability, bottleneck dominance.** These require showing that the coupling really is weak, or that the bound really is tight, in this regime. That may cost a sub-solve.

---

## 5. Correctness obligations gate; performance obligations steer

There are two kinds of obligation, and they behave completely differently.

**Correctness obligations** prove that every admitted design computes the workload. This is what the current framework verifies. A correctness counterexample is final: the design is rejected, with no negotiation.

**Performance obligations** prove that a structural assumption holds, for example "this region is HBM-bound". If this assumption fails, the design is *not wrong*. A compute-bound design still computes the right answer. The failed assumption costs **optimality, never correctness**: the surrogate ranked candidates on a wrong premise, but every admitted design is still certified correct.

Performance obligations come in two strengths:

| Strength | How it is established | What it may do | If it is wrong |
|---|---|---|---|
| **Proved bound** | One-sided bound, e.g. from monotonicity | **Prune** branches; branch-and-bound stays exact | Cannot be wrong within its scope |
| **Observed trend** | Empirical, e.g. HBM-bound on the samples checked | **Reorder** exploration only | Reopen the branch; never a wrong answer |

The rule follows:

> **Correctness obligations gate. Performance obligations steer.**
> Whether a structural assumption may *gate* (prune) or only *steer* (prioritize) depends on whether it was proved as a bound or merely observed as a trend.

This also answers the open question of hard versus soft feedback. Every structure in the three buckets either narrows *what is correct* or narrows *what is worth searching*, and the agent must treat the two differently.

### How a performance obligation is discharged (added after the follow-up discussion)

Three means, in decreasing strength; use the strongest available:

1. **Analytic bound over the branch.** The regime claim is an inequality between resource times with closed forms in the cost model. Show $\min_B t_{\text{HBM}} \ge \max_B t_r$ for every other resource $r$; usually interval arithmetic because traffic is monotone in tile size. Result: proved bound, may prune.
2. **Regime as a constraint.** Write the inequality into the template's search constraints; the assumption then holds by construction (like K1). Designs outside go to a sibling template with a different surrogate; the boundary between them is the regime boundary.
3. **Observation.** Probe sampled designs and record a trend. May only steer.

**When the regime fails on a corner.** It is not a Pandora's box: the next resource takes over, which is one more compact regime, and the number of flips is bounded by the rungs of the hierarchy. The witness design names the variable that moved the bottleneck; split there. Guarantee: *regime-local optimality*, exact within each proved regime and over the regimes visited.

**Is the corner worth searching?** The roofline ceiling of the next resource caps what the corner can deliver. Compare the cap with the incumbent; skip the corner if the upside is marginal. This is a bound only while the cost model is *optimistic* (never predicts faster than measured), which is checked continuously against the evaluator; the first violation demotes all ceiling-based skips to steer and reopens them.

**Probing with a confidence.** Otherwise sample the corner, near the boundary first, best predicted upside first, within a budget $B$. After $n$ losing probes the fraction of the corner that would beat the incumbent is below $3/n$ at 95% (rule of three). Stop when that bound is below a threshold or the budget is spent. Record the best gain seen and the bound on what remains; the population keeps both as the regime's evidence. No magic here, but the budget is written down.

### Summary table

| Structure | Bucket | Formulation move | Obligation | Typical strength | Verification cost |
|---|---|---|---|---|---|
| Bottleneck dominance | Objective | Surrogate objective (maximize bottleneck utilization) | Regime holds across the branch | Trend, or bound if proved | Expensive |
| Concavity | Objective | Restrict search to knees and boundaries | Saturation in this range | Bound when analytic | Moderate |
| Monotonicity | Objective | One-sided bounds for pruning | Monotone in the resource | Bound | Cheap |
| Near-decomposability | Variables | Split into clusters; couple through separators | Cross-cluster coupling is weak | Trend, or bound if proved | Expensive |
| Scenario decomposition | Variables | Benders: fix hardware, solve scenarios separately | Scenarios interact only through hardware | Exact by construction | Cheap |
| Low effective dimension | Variables | Freeze or coarsen variables that do not touch the bottleneck | Frozen variables do not affect the bound | Trend | Moderate |
| Symmetry | Enumeration | Symmetry-breaking constraints | Configurations are equivalent | Exact | Cheap |
| Tiny menus | Enumeration | Small discrete domains | Menu covers the useful values | By definition | Cheap |
| Dominance | Enumeration | Discard dominated designs | Dominance relation holds | Exact when compared directly | Cheap |

---

## 6. Worked example: the HBM bottleneck, end to end

**1. Detection (cheap, no search).** The agent runs the coarse cost model once and reads utilization against each roofline ceiling. Suppose HBM is pinned at about 95% while compute sits at 40%. The output is a claim: *this design region is HBM-bound.*

**2. Formulation move (where the complexity disappears).** The agent rewrites the MILP in two ways:

- It replaces the full performance objective with a surrogate: **maximize HBM bandwidth utilization**, a tighter and simpler function.
- It freezes or coarsens every variable that does not affect HBM traffic, such as register port count and tensor-unit shape.

A problem over about 20 coupled knobs becomes a problem over the 3 or 4 that move bytes: tiling, shared-memory residency and the transfer schedule.

**3. Verification (the price of trusting the regime).** The surrogate is valid only while HBM stays the bottleneck. The precondition is: *HBM remains the dominant constraint across the whole feasible region of this subproblem.* A tiling choice could relieve HBM and pass the bottleneck to shared memory, flipping the ordering. In practice the verifier proves a bound: even the most HBM-relieving design in the branch still leaves HBM above the next level.

- If the bound is proved, the obligation is **proved**, and the surrogate may *gate*: it can prune.
- If it is only observed, the obligation **steers**: it can reorder branches, and a failure reopens them.
- If the bound fails, the branch straddles a regime boundary, and the agent splits it.

**4. A regime flip is a refinement signal, not a failure.** When the verifier finds that the bottleneck has moved, it has moved down the hierarchy. The agent re-detects, finds shared memory pinned, and reformulates one level lower with a new surrogate and a new precondition. **The regime boundaries in the search space become the refinement steps.** This is the progressive coarse-to-fine refinement of the paper, seen through cost: the agent always formulates around the data movement that currently dominates.

---

## 7. Connection to the framework and to self-improvement

- **Progressive refinement.** The bottleneck ordering is the refinement order seen through cost. Physics tells the agent what to refine next.
- **Template population.** Each certified template records which structure held, where it held, and how expensive it was to establish. The population becomes a library of *structure–regime pairs*.
- **Grounded self-improvement.** Over time the agent improves at *recognizing* structure: which bottleneck patterns recur, which decompositions hold for which workload families, and which observed trends were later proved as bounds. Because every template must pass the gate, this improvement rests on proof and measurement, not on the agent's own judgment.

---

## 8. Open questions

1. **Completeness of the catalog.** What other structures exist, beyond §2.10? Can the three buckets be argued to be exhaustive?
2. **Cheap regime certificates.** How cheaply can bottleneck dominance and near-decomposability be proved as *bounds* rather than observed as trends? Which analytic models, such as roofline or affine footprints, give them almost for free?
3. **Regime splitting.** When a branch straddles a regime boundary, where should the agent split? Is the number of regimes small in practice?
4. **Measurement.** For each structure, how much does it shrink the search? This could be measured as nodes explored, solver time, or optimality gap, with and without the structure, on the GEMM, attention and Ventus studies.
5. **Agent prompting.** How should the agent be asked to detect structure? Should it propose a structure, its formulation move and its obligation as one unit, with the obligation's strength stated explicitly?
