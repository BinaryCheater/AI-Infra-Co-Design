# Model import into AI-Infra-Co-Design — 18 October 2026

Destination: `memxlife/AI-Infra-Co-Design`, branch `codex/ventus-model-v10`,
based on fetched `main` commit `7a932cfdbe24876bc80605849313d14708915727`.
The model source is the previously packaged commit
`c4d138e5cd800b1b081ce06693eff3e39a8aa5bb` of the standalone `ventus-codesign` model package.
That earlier publication used the wrong destination repository; this branch
places the independent model in the repository requested by the user.

## Import boundary

`codesign/` and `ventus_project/` are copied without runtime changes, together
with model tests, examples, dependency lockfile and model documentation/evidence.
`codesign/` contains the Ventus timing/cost model and its small numerical MILP
helper. `ventus_project/` is an optional local/SSH execution controller, with no
LLM/backend/UI dependency. The root repository's package/environment and existing
`agentic/`, GB10, calibration and paper files remain unchanged, except for a
README link to this model directory.

The original agent system, chat/session state, private configuration, new result
folders, caches, build products, full upstream Git history and approximately
1 GB of synthesis intermediates are not imported. Small independent instruction
reference/input fixtures are retained so the model regressions remain executable.

## Provenance and source excerpts

Frozen evidence is copied byte-for-byte, including failed numerical runs and
its own original source snapshots/manifests. Those snapshots describe their
recorded executions, not a new run in this repository. Model outputs remain
predictions; no optimization benefit or new accuracy calibration is claimed.

`vendor/ventus-gpgpu/SOURCE.json` explicitly describes a source excerpt: reviewed
RTL/driver declarations needed by static model audits, plus upstream licenses.
Its paths/hashes bind the shipped subset. The original full upstream manifest is
preserved separately as `SOURCE.upstream.json`. This excerpt is not a buildable
replacement for a full upstream RTL checkout; fresh numerical RTL execution uses
the separately configured checkout exactly as before.

`archive/export-manifest.json` binds only the shipped small regression fixtures.
The original full manifest is preserved as `archive/UPSTREAM-export-manifest.json`.
The original large historical replay dataset is not bundled; historical replay
commands need that dataset supplied separately. Archive-only integrity in this
package checks the shipped fixtures, not the original 1,834-artifact full archive.
No Git LFS or synthesized netlist download is required for model execution.

`IMPORT.json` binds the unchanged imported runtime files to their source commit.
Packaging documentation, ignore rules and subset manifests are local additions.
The target repository's existing agent loop is not wired to this model by this
import.
