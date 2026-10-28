# Frozen 200-track development comparison

[Index](../README.md) · [Specification](EXPERIMENT.md) · [Previous evidence](FROZEN_POLICY_RESULTS_2026-10-28.md)

Declared and launched 2026-10-28. This is a development campaign, not the final
paper evaluation. Live technical status is in
`runs/campaigns/broad_driver_20261028/status.json`; the full report is generated only
after all conditions finish and their provenance checks pass.

## Question and declaration

Does the original frozen driver's simulation-search improvement persist across
the complete 200-track development panel, including tracks outside the repeatedly
used first 50? Does it exceed the already fitted cheap constant correction?

The [immutable declaration](../configs/campaigns/broad-driver-2026-10-28.json)
fixes the checkpoint, source hashes, software versions, conditions and analysis:

| Condition | Simulated candidates per decision | New training |
|---|---:|---:|
| Original frozen recurrent policy | 0 | 0 |
| Same policy plus the previously selected mean-7 constant offset | 0 | 0 |
| Same policy with programmed-physics selection | 7 | 0 |
| Same policy with programmed-physics selection | 21 | 0 |

All use the same local geometry/ego/history interface and four native frames per
action. Search keeps horizon 20, planning seed zero, residual scales and utility
from the original study. Its sparse renderer has prior parity evidence. The
checkpoint remains the original 1,007,379-frame standalone-selected model. No
offset is refitted. The order is direct, constant offset, search 7, search 21.

Caps: 800 episodes, 800,000 real evaluation frames, 112,000,000 internal native
physics steps, zero training frames. Conditions execute sequentially with no
competing project training or racing. Ordinary workstation activity is not
controlled. Timing is measured again; the earlier renderer microbenchmark does
not replace these measurements.

The runner copies production source and executes children from that verified
snapshot. It also copies the runner, reporter, lockfile and manifest. Subsequent
workspace development cannot change an in-progress condition. After each
condition, source/dependency/checkpoint identities are checked. The first 20
episodes must reproduce their reference observations, actions and rewards exactly;
this reuses campaign traces and adds no environment steps. Failures stop the
campaign and retain its artifacts. No performance-based early stopping occurs.

## Analysis and uncertainty

Primary contrasts are search 7 minus direct and search 21 minus direct. Search 21
minus the fixed correction is a supporting contrast; search 21 minus search 7
measures the additional budget's return. Report official return, completion,
progress, off-road frames, exits, completion-conditioned lap time, latency and
all compute. Include paired track-bootstrap intervals, 5,000 draws, seed zero.
These are nominal unadjusted development intervals for one trained policy and
one planning stream, not uncertainty across trained models.

Prespecified panels are all 200, the original 20, the checkpoint-selection 50,
and the remaining 150. The first two small panels overlap. The remaining 150
were outside the current checkpoint selection, but remain development data;
historical project use and later adaptive analysis prevent treating them as a
final holdout. Report disagreement between panels instead of selecting the
most favorable subset.

**Assistant decision, high confidence:** broaden before changing planner settings;
preserve the cheap control, pairing, costs and exact replay checks.
**Uncertain:** whether the gain persists, whether another learned policy benefits,
and whether its added latency is justified. A fixed order can confound timing
with machine load; any latency claim needs later isolated/interleaved profiling.
Do not change this running comparison to resolve that uncertainty.

## Follow-ups under autonomous execution

The user asked to continue without routine confirmations and document uncertain
choices and their purpose. The assistant may execute bounded development probes
and revise plans from public evidence. Final-test identities/results and private
candidate-ranking outputs remain excluded until their respective protocol gates;
the private outputs never become agent feedback.

1. Finish and validate all four conditions. Preserve negative or mixed results.
2. Audit an established optimizer before its matched-work comparison. Count
   actual simulator evaluations across iterations and shifted-candidate reuse.
3. Independently test the already proposed lower-learning-rate baseline repair
   after this latency campaign, with its own declaration and frozen source.
4. Review both evidence blocks before extending training, choosing independent
   seeds, or changing the prediction/search representation. A successful probe
   is not automatically a stronger general baseline or a paper-ready result.

Broader independent-model comparisons, native-input reproduction, information
ablations, a complete study freeze and untouched final evaluation still remain.
This campaign alone cannot supply all data required for a SOTA claim.

## Queued baseline probe

The [lower-learning-rate declaration](../configs/campaigns/driver-lr3e5-block-2026-10-28.json)
is frozen and its queue launched. Status is
`runs/queues/driver_lr3e5_block_20261028/status.json`. This is queued, not trained.

The sole algorithm change from the original driver is initial learning rate
3e-5 instead of 1e-4. Retain ten PPO epochs, no KL stop, fresh training seed zero,
original architecture, observations, reward normalization and 8M-frame global
schedule. Stop at the initial 2M review. Evaluate approximately 200k, 500k, 1M
and 2M checkpoints on the same first 50 development tracks. The scheduled-frame
boundaries round up to a complete rollout. Caps are 2,008,191 training frames
and 200,000 development frames, plus the separately recorded reload check.

**Assistant rationale, moderate confidence:** smaller updates may reduce the
regression without discarding most optimization as the KL trial did. The exact
3e-5 value is an engineering choice with low confidence; it may simply slow
learning. A fresh common numeric seed is not independent replication. No
representation, reward or regularization sweep is bundled into this probe.
Select its checkpoint by standalone mean return, then completion, then earlier
frame count. Review the entire curve and retain every failed checkpoint/trial.
A winning development checkpoint does not automatically replace the original
frozen-search evidence or justify extending training to 8M.

The queue owns another immutable source/configuration/script snapshot and waits
for successful broad-campaign reporting. It fails closed on a failed or stale
predecessor. It then runs the full frozen-source regression suite before training.
This serialization avoids competing training/racing jobs during search timing.
Tests passed before the broad launch (80 passed); the queued repeat is an
additional gate. Training and its checkpoint evaluations may overlap each other,
as in prior blocks; their wall time includes that contention.

## Completed-condition snapshot requested during execution

The user requested a table before the full campaign finished. Direct, fixed
correction and search 7 each completed all 200 tracks. Their verified mean return
and completion are 862.42/35.5%, 865.95/39.5% and 873.05/58.5%, respectively.
Search 7 improves completion by 23 percentage points [16.0, 30.5]; return gain
+10.64 has interval [-2.56, 22.80]. These are paired, nominal development intervals
for one frozen policy. Search p95 is 136.77 ms, above the 80 ms action interval.

See the [completed-condition snapshot](../reports/broad_driver_completed_2026-10-28_1810/README.md)
for the table, all panels, costs, attribution checks and additional descriptive
contrasts. Search 21 remains excluded until it finishes. This requested interim
inspection does not change methods, the ongoing run, or the queued baseline probe.

A subsequent user-requested [188-track common-prefix preview](../reports/broad_driver_preview_2026-10-28_2004/README.md)
includes the still-running search-21 condition: 884.47 return and 72.3% completion
versus direct 865.32/35.6% on those same 188 tracks. This is explicitly an interim
descriptive panel, not the completed 200-track result or a trigger for method changes.
