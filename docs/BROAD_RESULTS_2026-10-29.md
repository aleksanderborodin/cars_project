# Completed broad comparison and lower-learning-rate probe

Reviewed 2026-10-29. [Paper blueprint](PAPER_BLUEPRINT.md) ·
[Declaration](BROAD_DEVELOPMENT_2026-10-28.md) · [Status](STATUS.md)

Both campaigns finished on 2026-10-28: broad reporting at 20:17 UTC and the
queued training/reporting sequence at 20:42 UTC. No jobs from this batch remain
active. Source/configuration/checkpoint and public result hashes were verified;
the generated figures were visually checked. The serialized regression gate
passed all 80 tests with three upstream SWIG deprecation warnings.

## All 200 common development tracks

Same original 1,007,379-frame recurrent checkpoint, local geometry/ego/history,
four-frame controls, planning seed zero, unchanged settings and sparse renderer.
The fixed correction was fitted earlier and not refitted. No new policy training
occurred within this comparison. Each condition's first 20 episodes replayed its
reference observations, actions and rewards exactly.

| Condition | Mean official return | Completion | Mean off-road frames/race | Pipeline p95 | Run wall time |
|---|---:|---:|---:|---:|---:|
| Direct | 862.42 | 35.5% (71/200) | 25.82 | 1.50 ms | 1.40 min |
| Fixed correction | 865.95 | 39.5% (79/200) | 25.86 | 1.53 ms | 1.40 min |
| Search 7 | 873.05 | 58.5% (117/200) | 4.53 | 136.77 ms | 93.03 min |
| Search 21 | 884.48 | 72.0% (144/200) | 3.03 | 362.11 ms | 231.14 min |

| Paired contrast | Mean-return difference, 95% interval | Completion difference, percentage points, 95% interval |
|---|---:|---:|
| Search 7 − direct, primary | +10.64 [−2.56, 22.80] | +23.0 [16.0, 30.5] |
| Search 21 − direct, primary | +22.06 [10.18, 33.35] | +36.5 [29.5, 44.0] |
| Search 21 − fixed correction, supporting | +18.53 [5.17, 32.15] | +32.5 [25.5, 39.5] |
| Search 21 − search 7, secondary | +11.43 [−0.85, 24.15] | +13.5 [7.5, 20.0] |

Intervals use 5,000 paired track-bootstrap draws, seed zero. They are nominal,
unadjusted exploratory development intervals conditional on this checkpoint and
planning stream. They do not capture independent policy-training variability or
adjust for the project's earlier development selection.

On the remaining 150 tracks outside checkpoint selection, direct/search 21 score
863.85/880.78 with 37.3%/72.7% completion. The paired return gain is +16.93
[2.48, 30.65]. Search 21 versus fixed correction on that subset is +13.80
[−2.77, 29.51], so that smaller-panel return contrast remains uncertain. All
20/50/150/200 panels remain development data; these are not independent holdouts.

**Interpretation:** simulation selection improves the tested frozen policy's
completion, and search 21 has a positive nominal return interval versus direct
on the full panel and remaining 150 subset. This justifies further investigation.
Search 7's return gain and search 21 minus search 7 return gain remain uncertain.
The larger population's completion benefit is clearer, with substantial extra cost.

The 80 ms action interval is exceeded by both search variants; every recorded
search decision missed that budget. The simulator waits for actions, with no
deadline fallback. These are not real-time results. Conditions ran sequentially,
but ordinary workstation load was not controlled and order was fixed. Pipeline
latency includes observation construction and policy/search, excluding real
environment stepping and disk output.

Comparison cost: **737,023 native evaluation frames**, **99,945,440 internal
native physics steps**, **23,921,439 actor-forward examples**. Actor training and
earlier trials/verification are additional costs. The full campaign/reporting
wall time was 19,644.56 s. Preserved sources:
[table](../reports/broad_driver_2026-10-28/summary.csv),
[subsets](../reports/broad_driver_2026-10-28/subsets.csv),
[analysis/provenance](../reports/broad_driver_2026-10-28/analysis.json),
[figure](../reports/broad_driver_2026-10-28/comparison.png).

## Lower-learning-rate training probe

Fresh seed zero; initial learning rate 3e-5 instead of 1e-4; ten epochs and no KL
stop. Same architecture/inputs, normalization and 8M global schedule, stopped at
the 2M review. All four scheduled checkpoints were evaluated on the same first 50
development tracks. This is not an independent replication of the original setting.

| Actual training frames | Mean return | Completion |
|---:|---:|---:|
| 204,800 | 314.56 | 0% |
| 507,876 | 692.18 | 8% |
| 1,007,562 | 622.33 | 0% |
| 2,006,916 | 624.51 | 0% |

The declared standalone selection rule chooses 507,876 frames. Its performance is
below the original selected driver's 858.11/30% on the same 50 tracks. **Assistant
decision:** do not adopt this repair or automatically extend it. Keep the original
checkpoint identities and all search results. Smaller updates may slow learning;
one failed 2M trial is not proof that this rate can never work or that PPO cannot
learn the task. The earlier KL-stop repair also failed to improve the reference.

Cost: 2,006,916 new training frames, 199,612 development evaluation frames and 8
reload-check frames; 1,435.11 s training-campaign wall time. Learning diagnostics
were finite. The training/evaluation overlap is included in that wall time.
[Checkpoints](../reports/driver_lr3e5_block_2026-10-28/checkpoints.csv),
[analysis/provenance](../reports/driver_lr3e5_block_2026-10-28/analysis.json),
[learning curve](../reports/driver_lr3e5_block_2026-10-28/learning_curve.png).

## What this changes

Preserve the positive local search result and unsuccessful baseline probes.
Strengthen baseline verification and implement the established-optimizer control;
the [paper blueprint](PAPER_BLUEPRINT.md) now consolidates the user's three-stage
diagnostic study and its prerequisites. A visual-perception conclusion, complete
forecasting decomposition, independent-policy robustness and native-input SOTA
comparison are still missing. Neither campaign opened final-test data or the
private candidate-ranking monitor.
