# Ventus model import decisions — 18 October 2026

The user explicitly selected `memxlife/AI-Infra-Co-Design` as the destination.
The branch starts at its latest fetched `main` (`7a932cf`) and adds the independent
model under `models/ventus/`, preserving the existing root package and agent code.

The assistant chose a nested package with its own lockfile/environment so model
execution does not inherit agent dependencies or alter the existing project.
Runtime files retain the earlier packaged model's exact bytes. Model tools,
examples, regressions and evidence are included; private settings and temporary
runs/builds are excluded.

Large synthesis intermediates and a complete upstream RTL copy are omitted.
Small regression fixtures and reviewed source excerpts retain explicit subset
manifests; original full manifests remain reference records. Fresh RTL still
requires an external configured full checkout. See `../IMPORT.md` for scope.

No model mechanism, coefficient or default changed in this import. Compiled v10,
probe v4, historical modes and documented approximations remain as recorded.
No new accuracy, optimization benefit, PPA or user acceptance is claimed.
