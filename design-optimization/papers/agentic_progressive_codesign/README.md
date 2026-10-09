# Agentic Progressive Co-Design

**Agentic Progressive Optimization for Hardware–Software Co-Design of AI Infrastructure**

[Read the PDF](paper.pdf) · [LaTeX source](paper.tex)

## Build

Run `make` with Tectonic installed. On macOS, install it with `brew install tectonic`. The first compilation downloads TeX dependencies. `make render` additionally requires Poppler (`brew install poppler`) and produces page previews in `build/previews/`.

## Manuscript files

- `paper.tex` and `sections/`: manuscript text and appendices.
- `references.tex`: research-paper bibliography.
- `figures/`: editable TikZ/PGFPlots figures and their CSV data in `figures/data/`.
- `paper.pdf`: compiled manuscript.
