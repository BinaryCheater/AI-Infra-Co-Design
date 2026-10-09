# Agentic Progressive Optimization for Hardware–Software Co-Design of AI Infrastructure

## Central argument

AI infrastructure design is a joint choice of algorithmic organization, executable software, instruction interfaces, and hardware structure. The useful design space is too large to expose or solve all at once. Agents expand the representation when valuable implementations are missing, then exploit structure to make the resulting MIP tractable. A solution is decoded into paired software and hardware, and implementation feedback changes the next formulation.

## Contributions and evidence

1. A design-space formulation that distinguishes representation expansion, equivalent reformulation, and search restriction, and makes the software–hardware boundary a decision.
2. A hierarchical construction and search method connecting algorithmic paradigms, tensor dependencies, implementation capabilities, MIP decisions, and paired lowering.
3. Mechanism-centered Ventus case studies: conditional value of software transformations, gains over a one-pass design process, and the consequences of cost-model error.

The method is presented as a framework. The measured studies exercise bounded software/hardware choices and MIP-guided selection; they do not measure an autonomous end-to-end formulation agent or a newly synthesized ISA. Those distinctions belong in one evaluation-scope paragraph and a focused discussion, not repeated disclaimers.

## Article structure

1. Introduction: a register/reuse example; why the interface is a design choice; the dual challenge of expressing and finding good designs; contributions.
2. Related Work: CoSA, Axon, and LLM4MIP as complementary foundations; compiler representations; architecture/mapping search; evaluation infrastructure.
3. Co-Design as Progressive Design-Space Construction: workload and design domain; realizable sets; model transformations; conditional dominance; movable boundaries.
4. Constructing Coupled MIP Formulations: representation hierarchy; online-attention worked example; resource and timing encoding; joint boundary choices; paired software/hardware generation.
5. Agent-Guided Reformulation and Search: typed actions; structural neighborhoods; decomposition; candidate retention; diagnosis and feedback; algorithm.
6. Experimental Evaluation: setup and evidence levels; fusion preference reversal and controlled comparisons; tiling/register/cache coupling; model fidelity and transfer; synthesis of findings.
7. Discussion and Limitations: coverage of modern algorithms; evaluation needed for frontier comparisons; hardware boundary experiments; amortization and portability.
8. Conclusion.

Appendices: notation and linearization, a complete finite boundary-choice example, empirical data definitions and provenance.

## Figures

1. Joint design and feedback: representation levels, formulation agent, candidate archive, solver, paired decoding, implementation evaluation.
2. Attention: materialized matrices versus streaming state; connection from algorithm to storage/ports/engines.
3. Movable boundary: software sequence, programmable instruction mechanism, and specialized data path.
4. Empirical mechanism: fusion preference reversal and six controlled optimization strategies in the same study.
5. Hierarchical expansion: the full-instruction-simulation trajectory and its software/hardware changes.

## Editorial constraints

Write original English prose. Reuse the verification article's article-class format and argumentative order only. Do not import proof certificates, refinement-map theorems, or proof-cache experiments. Keep project administration, assignment instructions, researcher identities, and drafting history outside the article. Cite public experiment reports through stable source identifiers and include a data/provenance file. No performance result is invented, extrapolated across studies, or labeled as a physical measurement.
