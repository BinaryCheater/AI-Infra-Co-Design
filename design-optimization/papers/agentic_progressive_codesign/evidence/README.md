# Empirical source records

The manuscript analyzes three public Ventus studies retrieved on 2026-10-09. JSON files retain the report body, update timestamp, URL, and relevant comments in the source language. They are report snapshots; they are not raw simulator receipts or a new reproduction campaign. CSV files transcribe the values used by the LaTeX figures directly from those records.

| Data | Source | Scope |
|---|---|---|
| `controlled.csv` | Issue 4, six-method comparison | Qwen2.5-14B calibrated whole-network estimates; maximum normalized cycle ratio across each scenario set |
| `fusion.csv` | Issue 4, D128 gate/up example | Compiled representative-program timing, in million cycles |
| `trajectory.csv` | Issue 3 body; comments 6077807611 and 6077879211 | Successive accepted designs evaluated by complete instruction simulation; values retain reported rounding |

Source URLs:

- https://github.com/BinaryCheater/ventus-codesign/issues/4
- https://github.com/BinaryCheater/ventus-codesign/issues/3
- https://github.com/BinaryCheater/ventus-codesign/issues/6

The third study is discussed in prose to explain model mismatch and transfer behavior. Its results are not pooled with the first two studies. A full instruction-timing simulation is distinct from numerical execution of all inference tensors and from RTL or physical measurement.

`validate.py` checks source transcriptions, figure dependencies, citation/label closure, and derived percentage calculations. Figures use the CSV values through PGFPlots. The article cites reports by project and mechanism; original report titles remain intact in the frozen evidence.

## October 10 update

The main evaluation now emphasizes the programmable Phase 2 benchmark and the Qwen2.5-0.5B Ventus iterations. Existing snapshots and 14B CSV files remain unchanged as historical evidence, outside the primary plotted comparison.

- `phase2.csv`: five frozen baseline/candidate cycle pairs, weights and speedups. `update-20261010/phase2-verified.json` retains the private-seed verification receipt. Baselines, weights and workloads follow `docs/phase_two_release_cn.md` in the parent break-layer project, model `phase-two-compact-v2`. The weighted geometric mean is 7.0173192128.
- `ventus-controls-05b.csv`: second-iteration slice estimates from `update-20261010/iteration-2.md`. These controls must not be relabeled as complete simulations or combined with third-iteration candidates as a new controlled trial.
- `update-20261010/iteration-{1,2,3}.md`: reports from https://github.com/yinsumirage/ventus-codesign/tree/gwt/project-abc/docs/evidence . They document hardware reversions, software-dependent cache value, coupled MIP terms, and larger-tile exploration. The third report records that targeted exploration, rather than the existing surrogate, found the new tile/hardware family.
- `update-20261010/kv-storage-issue-5.json`: https://github.com/BinaryCheater/ventus-codesign/issues/5 . Storage comparisons cover a local attention subproblem and local error thresholds, not whole-model accuracy.

The 2.93×/3.58× headline speedups use full candidate runs, with the directly reproduced 128-token baseline determining both task ratios. The 512-token baseline remains a historical extrapolation. Timing and area are model results, not silicon measurements. “Short campaign” refers to three documented design iterations and does not assert an unrecorded end-to-end wall time.
