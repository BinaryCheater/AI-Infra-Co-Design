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
