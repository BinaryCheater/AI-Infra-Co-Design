# Agentic Progressive Co-Design

**Agentic Progressive Optimization for Hardware–Software Co-Design of AI Infrastructure**

[Read the PDF](paper.pdf) · [LaTeX source](paper.tex) · [Detailed outline](OUTLINE.md)

The manuscript develops a joint design-space formulation spanning algorithmic organization, tensor programs, software–hardware interfaces, and hardware resources. Agents expand the representation and exploit optimization structure; a common design assignment drives paired software and hardware lowering. Three Ventus case studies examine cross-layer preference reversals, controlled design strategies, and performance-model error.

The paper is written in English and contains five original vector figures. Its organization follows the companion verification manuscript, while the technical content concerns design-space construction and optimization. The framework and analytical examples are distinguished from the measured case studies.

## Build

Install Tectonic and, for rendering, Poppler. On macOS:

```sh
brew install tectonic poppler
make
```

`make` checks citations, labels, input files, and empirical data before compiling `paper.pdf`. The first Tectonic run downloads its TeX dependencies; subsequent builds can reuse the cache. An existing TeX Live installation can also compile `paper.tex` with two PDFLaTeX passes. All figures are TikZ/PGFPlots sources; no proprietary fonts or external graphics programs are required.

```sh
make render
```

This creates page previews in the ignored `build/previews/` directory. Python validation uses only the standard library.

## Contents

- `paper.tex`: article setup, abstract, and section order.
- `sections/`: full text, formulation, method, evaluation, and appendices.
- `figures/`: editable vector diagrams and data-driven plots.
- `references.tex`: bibliography, including CoSA, Axon, and LLM4MIP.
- `evidence/`: frozen public report snapshots and plotted CSV data.
- `validate.py`: source and numerical consistency checks.

The evidence package reproduces the paper's figures and source transcriptions. It does not re-run the underlying accelerator experiments. Links to the experiment reports and their available reproduction material are retained in the bibliography and evidence directory.
