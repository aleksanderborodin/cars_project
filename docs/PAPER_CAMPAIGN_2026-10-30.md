# Paper execution ledger and next queued campaign

Updated 2026-10-30. [Intent](USER_INTENT.md) · [Blueprint](PAPER_BLUEPRINT.md)

**User steering:** execute the experiments needed for an actual scientific paper,
with realistic claims and complete documentation. **Assistant execution choice:**
finish the retained-recipe seed replications and queue a fixed full-development
same-proposer comparison. This ledger distinguishes executed work, executable
queued work, required unresolved studies and optional extensions.

## What is executing now

The [seed 1/2 replications](LOCAL_REPLICATIONS_2026-10-30.md) run serially, each to
its declared 2M-native-frame review. Both recipes/source snapshots were frozen
before launch. Their standalone scores select each policy checkpoint before any
search result exists. Prior seed 0 remains historical exploratory evidence.

The next [queue declaration](../configs/campaigns/paper-search-sparse-queue-2026-10-30.json)
freezes **three policy seeds × 200 development tracks × ten conditions**:

| Condition | Purpose |
|---|---|
| Direct frozen policy | Same-actor baseline |
| Fixed control correction | Transfer the previously frozen seed 0 correction; no new fitting |
| Random choice among 7 proposals, planning seeds 0/1/2 | Proposal-variation control |
| Random choice among 21 proposals, planning seeds 0/1/2 | Count-matched control for the primary 21-candidate method |
| Original search7 | Lower online-compute budget |
| Original search21 | Main retained programmed-simulator selector |

The same observation/history, programmed mechanics, estimated state/contact,
single tangent road continuation, score, three residual knots, 20-decision horizon
and repeat 4 are preserved. No new search algorithm, road predictor or private
ranking feedback is introduced. Random planning streams are fixed and reset
as before; three planning streams are not three independently trained policies.
The fixed correction was developed on seed 0; its transfer may be poor and does
not establish an optimal cheap correction for other actors.

Reuse seed 0's existing 200-track direct/fixed/search7/search21 results after exact
two-episode replay under the frozen queue code. New seed 0 random7 panels must
reproduce their original first 20 episodes. New seed 1/2 direct panels reproduce
the chosen checkpoint's existing first 50 evaluation episodes. Every method is
run and retained for every seed regardless of whether search helps.

**New cost caps:**5,208,000 native evaluation frames, 225,120,000 internal native
physics frames; zero training within this comparison. These include seed 0's
eight parity episodes. Seed0's four historical full panels are reused, not
charged as new interaction or counted as independent repetitions. All new random
and fixed/direct controls are costed even though they simulate no candidate worlds.
Reset physics ticks remain additional recorded infrastructure work; the legacy
evaluator does not persist a single aggregated reset counter.

The queue waits until both training seeds finish; it stops on dependency failure,
stale heartbeat, source/checkpoint mismatch, failed replay, resource cap or disk
floor. It runs from copied source and performs one condition at a time, with no
project training concurrently. The full 200-track development manifest is reused;
calibration/final-test identities and private ranks stay sealed. This is offline
control with measured latency; no real-time fallback or safety guarantee is added.

## Analysis frozen before outcomes

Primary: search21-minus-direct paired official return and completion, per training
seed, with every seed shown. Secondary: search7-minus-direct, search21-minus-mean
random21, search7-minus-mean random7, search21-minus-fixed, and search21-minus-search7.
Report individual random streams plus their within-track mean. Use descriptive
paired track intervals within each seed and explicitly separate across-training-
seed variability. Three exploratory actors, adaptive development selection and
historical seed 0 reuse do not establish final-test or headline five-seed validity.
Do not choose a checkpoint because its search gain is favorable.

## Remaining paper gates

| Work | Evidence/execution status | Required next step |
|---|---|---|
| Partitions, immutable configs, checkpoint/memory/parity tests | Implemented; 113 tests passing | Preserve at every new protocol |
| Original local search gain and cheap controls | Seed0 measured on 200 tracks | Seed1/2 plus count-matched random controls queued |
| Native recipe comparability | One full 4M-decision run, selected early checkpoint; late regression | Competence and training-seed robustness still unresolved |
| Strong learned/actor-free competitors | PPO adaptations measured; Valdi/iCEM sources audited; iCEM adaptation negative | Reproduce justified stronger competitors under explicit information/compute contracts |
| Current perception/future road/control decomposition | Fixed-controller geometry interventions and reference-tail study measured | Train/validate proposed V/L/F or narrower matched module study before those claims |
| Simulator/state/prediction validity | Approximate programmed model and executed-control diagnostics measured | Further controlled studies only where required by the final claim |
| Full final evaluation | Intentionally unopened | Freeze methods, selection, claims and exact matrix first |

The queue completes a substantial **local mechanism dataset**, not all paper
requirements. Strong-baseline and component-adequacy gates may require new work;
an unfavorable result must change our claims rather than be hidden.

Optional custom/GPU/independent simulator, learned road-distribution extensions,
agentic evolution and decision-space adaptation remain separately scoped future
ideas. The user requested that these intentions be retained; this does not make
every optional branch a prerequisite for the first paper or authorize silent
changes to current information boundaries.

## Realistic planning estimates

- Current two seeded blocks: roughly 50–70 minutes from their launch, based on
  measured 2M local blocks of about 25 minutes each; update from live progress.
- The queued full-development comparison: provisionally **12–18 hours after the
  training dependency finishes**. Earlier 200-track direct/fixed/search7/search21
  took 5.46 hours for one actor; two new actors plus random controls/parity dominate
  new cost. Ordinary machine load, policy episode lengths and failures can alter it.
- Core local curves/tables and audit: target **one to two days** if those jobs pass.
- Full paper-level experiment package: planning allowance **two to four weeks**,
  low confidence while stronger competitors and the final decomposition matrix
  are unresolved. This is a resource-planning estimate, not a promised completion
  date or a claim that SOTA/novelty will be established by then.

Status: declared. Queue preparation/checks precede waiting/execution. Remaining
unimplemented studies need explicit protocols, tests and frozen budgets before
their data can support paper claims; they are not reported as running.

Pre-execution correction: the initial waiting queue `paper_search_20261030` was
stopped with zero new environment frames because its command selected the reference
renderer, while the historical broad campaign used the validated sparse renderer.
The corrected `paper_search_sparse_20261030` declaration explicitly matches sparse
rendering and retains the same candidate engine/behavior. Original preparation
and stop reason are preserved; no failed or completed result was overwritten.

Corrected queue launch observed at `2026-10-30T13:29:11.279191+00:00`,
source/launch commit `5c0a2c6`. Live status:
`runs/queues/paper_search_sparse_20261030/status.json`. It is waiting for
`local_replications_20261030` and will execute/report automatically after both
training blocks complete. All 114 tests and manifest checks passed before launch.
Project training and evaluation measurements are serialized as declared; do not
launch additional training while this evaluation queue is active.
