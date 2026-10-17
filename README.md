# AI Infrastructure Co-Design

Research on progressive hardware–software co-design for AI infrastructure, with two connected areas:

- **[Design optimization](design-optimization/):** formulate and solve workload-driven design problems across computation graphs, tensor mappings, memory systems, schedules, and hardware resources.
- **[Design verification](design-verification/):** establish that progressive design refinements preserve the required computation and reuse correctness evidence as designs evolve.

Optimization proposes and evaluates design choices. Verification checks their preservation obligations and returns counterexamples or accepted evidence to guide further refinement.

## Infrastructure

[`agentic/`](agentic/) implements the multi-agent loop of [`notes/multi_agent_design.md`](notes/multi_agent_design.md):
a scripted or large-model Formulator proposes template contracts, a MILP Solver and a bounded Verifier
work on their two faces, an Evaluator measures the decoded design, and an append-only ledger feeds a
web console in which a computer architect can observe, step, edit, inject, override and rewind the loop.
It runs end to end on the paper's shared-memory tiling template; see [`agentic/README.md`](agentic/README.md)
for the stage status and how to run it.

## Hardware models

- **[Ventus](models/ventus/):** standalone Python/Rust performance model,
  compiled FP32 GEMM example, independent RTL checking tools and versioned model
  evidence. It runs independently of `agentic/`; this addition does not install
  a Ventus adapter into the agent loop.

## Current paper

[Agentic Progressive Verification for Hardware-Software Co-Design of AI Infrastructure](design-verification/papers/agentic_progressive_verification/paper.tex)

The manuscript is a research draft. It distinguishes the proposed agentic framework from measured bounded verification experiments. This initial publication contains the paper source; experiment artifacts are not yet included.
