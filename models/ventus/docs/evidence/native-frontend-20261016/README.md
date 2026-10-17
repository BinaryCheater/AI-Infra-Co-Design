# Native source frontend: parity and performance, 16 October 2026

This receipt covers `ventus-source-frontend-v3` implemented by
`ventus-frontend-rust-v1`, on the configured Linux x86_64 execution host with
Python 3.12.3. Source identities are recorded in every row. The input manifests
are the existing frozen development/validation programs; no new RTL run or new
independent validation sample is claimed.

## Observed scope

- `parity-96.json`: 24 development inputs and 24 validation inputs, each evaluated
  at RF4 and RF8. The historical directory name `holdout` contains development
  inputs; it does not confer independence. The validation RF8 subset reuses
  post-reference replay evidence.
- `parity-16.json`: the 16 separately frozen RF8 validation inputs.
- `summary.json`: combined timings, identities and exact matching to all 112
  existing RTL-comparison predictions. Accuracy and ordering therefore remain
  unchanged, including the previously reported failures.
- `default-cli.json`: direct execution without an explicit backend flag, showing
  that `rtl-accuracy predict --frontend` actually selects Rust. This local CLI
  smoke receipt includes cold build/startup and is not used in the host benchmark.

Each parity case compares all five per-operation event classes (collect,
operands-ready, issue, writeback, ENDPRG), endpoint spans and every named counter.
Absent counters are treated as zero. Every failure remains in the receipt;
failed parity would suppress its speedup and produce a nonzero command exit.

## Measured result

All **112 cases**, **163,560 instruction events**, all counters and endpoints match.

| Per-case median metric | Minimum | Median across cases | Maximum |
|---|---:|---:|---:|
| Python summary | 22.59 ms | 172.99 ms | 1111.14 ms |
| Rust full call | 5.71 ms | 16.99 ms | 99.59 ms |
| Rust prepared call | 4.34 ms | 7.89 ms | 28.53 ms |
| One-time preparation | 0.97 ms | 8.70 ms | 70.72 ms |
| Full-call speedup | 3.85 × | 9.69 × | 13.50 × |
| Prepared-call speedup | 4.99 × | 21.73 × | 40.24 × |

## Timing method

For each input, the detailed Python graph supplies the parity reference. The
benchmark then alternates Python summary and native full-call ordering for three
repetitions. The table reports a distribution of per-case medians, not a pooled
throughput or tail-latency guarantee. Native full calls include structural
validation, encoding, runtime lookup/hashing, FFI, fresh hardware execution and
result decoding. Prepared calls reuse only immutable input encoding; the hardware
state and full execution are fresh. Preparation cost and initial build/first-call
cost are recorded separately. File loading and process startup are excluded.
Prepared calls follow the alternating full calls, so this small observational
benchmark does not eliminate all host/cache-order effects.

## Reproduction and limits

Use the [native guide](../../native-frontend.md). Run the frozen development and
validation manifests with `frontend-parity --banks 4 --banks 8 --repetitions 3`,
and the additional RF8 manifest separately with `--banks 8`; use a new output
directory each time. Hashes identify the exact inputs. Do not regenerate a
manifest and continue claiming the old identity.

The independent RTL accuracy remains 30/40 within 5%, 38/40 within 10%, with
135/140 candidate pairs correctly ordered. The stronger requested coverage and
ranking target is still unmet. This port preserves current approximations;
causal queued-request edges, cache transaction calendars, shared-L2 arbitration,
full launch/flush intervals and arbitrary compiled control flow need additional
model work. Native parity and internal tests do not constitute user acceptance.
