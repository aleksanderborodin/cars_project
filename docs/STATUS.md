# Implementation and evidence status

Updated 2026-10-27. [Index](../README.md) · [Specification](EXPERIMENT.md) ·
[Roadmap](ROADMAP.md)

## Implemented foundation

| Component | Actual capability | Limitation for the proposed paper |
|---|---|---|
| `env/make.py`, `carracing.py` | State-only CarRacing factory; official standard dynamics/reward; deterministic track generation | No local observation protocol or visual training factory |
| `env/state.py` | Truth extraction, track geometry, StateObs v1 | Privileged: future centreline/curvature and global features |
| `policies/base.py`, `eval/runner.py` | Full-track reset and raw-state control evaluation | Unsuitable for information-restricted claims without a new interface |
| `policies/baseline.py` | Pure-pursuit expert and speed controller | Privileged expert, not a local baseline |
| `il/`, `policies/torch_mlp.py` | BC/DAgger and normalized MLP artifacts | Privileged teacher/data; no matched local supervision controls yet |
| `rl/`, `policies/sb3.py` | Feedforward PPO/SAC, normalization, action-repeat and throttle options | Neither recurrence nor a modern world-model comparison |
| `training.py` | Seeded collection, config/cost records, immutable run creation | No new disjoint partition manifests or `seen-inf` exclusions |
| `env/metrics.py`, `eval/episode_log.py` | Return, completion, departure/exit metrics; reproducible detailed logs | Truth-rich logs must be separated from future agent-facing data |
| `eval/report.py` | Mean/std and rliable IQM/bootstrap summaries | Groups only method/split; not ready for protocol/condition comparisons |

## Planned, not implemented

- Local geometry and fixed-view local vision observation contracts and leakage tests.
- Disjoint training/development/calibration/test manifests and historical overlap audit.
- Fitted vehicle dynamics, deterministic MPC, and scenario MPC with a simple prior.
- Learned distribution of road continuations, calibration, and observation-aware planning.
- Recurrent model-free baseline; TD-MPC2, DreamerV3, and Valdi adapters/reproductions.
- Visual reconstruction and same-supervision representation controls.
- Protocol-aware paired reporting, latency diagnostics, frozen final study, paper results.

## Evidence ledger

| Statement | Status |
|---|---|
| The original foundation passed its tests | Verified before archival: 48 tests passed |
| The retained foundation passes after cleanup | Verified: 47 tests passed; one Rust-specific test was archived with its implementation |
| New protocols/methods outperform competitors | Not measured |
| The candidate mechanism is novel | Unestablished; close prior art identified |
| Standard CarRacing SOTA is beaten | Not established; no authoritative matched ranking verified |
| Physical transfer or formal safety is demonstrated | Out of scope / not demonstrated |
| Referenced papers and official code were surveyed | Literature review completed; no external reproduction executed |

Validation commands and caveats are recorded in [DEVELOPMENT.md](DEVELOPMENT.md).
Tests validate infrastructure behavior, not research performance. This revision
does not run a new training campaign or inspect held-out results.

## Current decision

Proceed with Roadmap A–B only as the next implementation milestone. The local
information boundary and value-of-information pilot must pass before committing
to a learned hidden-road model. Budget and publication aspirations remain
proposals, not measured feasibility or promised acceptance.
