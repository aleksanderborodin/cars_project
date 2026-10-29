# Reference continuation: completed controlled pilot

2026-10-29. [Index](../README.md) · [Intervention design](INTERVENTION_STUDY.md) ·
[Preceding results](INTERVENTION_RESULTS_2026-10-29.md) · [Status](STATUS.md)

**Status: complete.** The exact replay and geometry checks passed. Replacing only
the scoring-reference tail raised mean return from 594.21 to 876.29 and completion
from 0/5 to 4/5. The original local-search mean remains higher at 906.99. All five
tracks, the remaining failure and costs are reported below; no new training ran.
The subsequent [20-track replication](REFERENCE_REPLICATION_2026-10-29.md) is complete:
+286.25 versus tangent, but −71.86 versus local search, with one primary counterexample.

**User instruction:** plan the next experiments and run them. The broader user
idea is to separate understanding/prediction from deciding through interventions.
**Assistant design:** test whether a mismatch between true simulated road and a
straight scoring reference contributed to the previous oracle-condition slowdown.
Preserving the entire visible prefix, extension extent and sampling is an
assistant refinement to isolate this contrast more precisely than replacing the
whole reference with a resampled full loop. This is a privileged diagnostic.

## One new condition, one replay gate

| Role | Road model | Objective reference | Work |
|---|---|---|---|
| Primary control | Existing oracle-derived road | True visible prefix, then 100 m straight tangent | Reuse five saved `both` episodes after replay |
| New condition | Identical oracle-derived road | Identical visible prefix, then 100 m along true directed road | Five common development tracks |
| Context only | Original local road prior | Original locally reconstructed reference | Reuse five saved baseline-search episodes |

The new mode is `reference=oracle_continued`; it requires the explicit global
diagnostic adapter. Match the same actor/checkpoint, ego/local masks/history,
programmed physics/observer, seven candidates, horizon 20, repeat four, planning
seed 0, residual scales and score weights. Keep the outside-footprint penalty and
the road world's own path unchanged. No new reward, termination, vehicle physics,
policy training, exact-state clone, learned scorer or private monitor access.

The helper anchors the final visible point onto the true directed polyline,
follows its segments, and samples at arc distances 1 through 100 m. The visible
samples are never resampled or altered. It handles loop-array wraparound and
duplicate vertices. Unanchorable endpoints fail explicitly. If no visible
reference exists, both modes retain the original local fallback and count it.
The 100 m extent is held from the earlier implementation for attribution; it is
not a newly optimized forecast range. This is still a sampled nearest-path utility,
not exact official-reward prediction or an optimal-control oracle.

## Checks before racing

1. Straight-road negative control: the true continuation exactly equals the old
   tangent when the supplied road remains straight over the extension.
2. Curved/wrapped paths: analytic segment distances, direction, duplicate vertices,
   vertex anchoring and map-array origin invariance.
3. Identical visible prefix, world queries and imagined observations. At a shared
   constructed root, verify all candidate action rollouts, initial predicted ego
   states and real actor memory match while scores can differ.
4. Check the visible prefix/length/anchoring on every recorded observation in the
   five primary-control trajectories. This uses public geometry, zero new real
   frames and zero physics rollouts.
5. Replay the primary control's first complete episode under the new code. Every
   observation, action, reward, termination and departure index must match exactly
   before the new condition starts. Five historical controls are reused; only one
   full episode is re-executed in this campaign. Preserve the earlier full replay
   evidence, rather than claiming five new replays here.

## Frozen budget and analysis

[Declaration](../configs/campaigns/reference-continuation-2026-10-29.json).
New condition: at most **5,000 native evaluation frames and 700,000 internal native
physics steps**. Replay: at most **1,000 / 140,000**. Total **6,000 / 840,000**;
zero new experiment training. Test-fixture training is separate infrastructure work.
Execute serially from copied, hash-verified sources with one CPU thread. Ordinary
workstation load remains uncontrolled; no competing project training/racing job.

Primary contrast: new true continuation minus the earlier `both` control, paired
on all five tracks. Report official return, completion, progress, departures,
sampled speed, braking, fallback counts and cost. Speed uses decision-end observed
samples weighted by action duration, then equal episode means. Compare the original
local-search result as context with its information difference explicit.

Retain every episode and paired difference, including losses. No five-track
significance, population-adequacy, general-information-value or SOTA claim.
Historical timing was measured in the previous campaign; do not call differences
paired latency improvements. Report new cost separately from reused historical cost.
Do not change weights/horizon or expand the panel based on interim outcomes.

If performance improves, this supports reference-continuation mismatch as one
limitation of this fixed controller. It does not isolate every score component
(the path affects both progress and centre distance) or prove the rest of the
system adequate. If it fails to recover performance, keep the result and design
a separate objective/horizon/imagined-input check. Either outcome still requires
strong baselines and appropriately trained V/L/F policies for the full paper.

```bash
uv run python scripts/run_reference_continuation.py --manifest configs/campaigns/reference-continuation-2026-10-29.json
```

Original IDs refuse overwriting. The technical status path is
`runs/campaigns/reference_continuation_20261029/status.json`. The campaign finished
at 2026-10-29 07:37:18 UTC (10:37 Moscow). All jobs in this batch are complete.

## Completed results

| Condition | Mean return | Completed | Mean progress | Sampled mean speed | Mean brake | Pipeline p95 |
|---|---:|---:|---:|---:|---:|---:|
| Original local search, historical context | 906.99 | 4/5 | 99.59% | 58.80 m/s | 0.0083 | 135.59 ms |
| Oracle road + tangent reference, historical primary control | 594.21 | 0/5 | 69.42% | 35.21 m/s | 0.0294 | 209.13 ms |
| Same oracle road + true reference continuation, new | 876.29 | 4/5 | 96.27% | 57.60 m/s | 0.0097 | 215.70 ms |

Primary paired return difference: **+282.09**, positive on every track.
Completion increased by four races; zero off-road frames and no playfield exits
occurred in either oracle condition. The new run still missed the 80 ms decision
interval on every decision; it is not a real-time result. Timing differences from
historical controls are descriptive, not a paired timing comparison.

| Development track, manifest order | Local search | Oracle + tangent | Oracle + true tail | New minus tangent | New minus local |
|---|---:|---:|---:|---:|---:|
| 1 | 917.30 | 667.61 | 921.00 | +253.39 | +3.70 |
| 2 | 879.66 | 516.95 | 713.56 | +196.61 | −166.10 |
| 3 | 914.30 | 604.70 | 916.00 | +311.30 | +1.70 |
| 4 | 907.90 | 571.87 | 909.20 | +337.33 | +1.30 |
| 5 | 915.80 | 609.90 | 921.70 | +311.80 | +5.90 |

The average difference from local search is **−30.70**, with the same four tracks
completing in both conditions. Track 2 remains an on-road timeout: progress 81.36%
versus local search's 97.97%, sampled speed 40.77 versus 55.48 m/s and mean brake
0.0282 versus 0.0119. There is no justification for excluding this case or reporting
only the four successful tracks as overall superiority. No fallback occurred.

## What this establishes and what remains open

**Measured conditional effect:** with the same oracle-derived road model, changing
only the reference continuation substantially improves this frozen controller on
these five development tracks. The visible reference, candidate rollout mechanism,
weights and model were fixed; the decline in braking and recovered speed accompany
the return gain. This supports reference-continuation mismatch as a real limitation
of the preceding oracle configuration.

**Limit:** the path affects both progress and centre-distance costs, so this does
not identify either cost term alone. It does not prove the objective predicts true
reward, the vehicle model is accurate for all candidates, or the policy uses perfect
perception. The score remains a hand-designed finite-horizon utility. These are
previously inspected tracks, one actor and one planning stream. The new privileged
method has not beaten the original local-search mean here, much less established
SOTA or a full-information learned-policy comparison.

The preceding audit's accurate oracle-road motion predictions for recorded actions
and poor oracle driving can now be understood more specifically: a substantial
part of the poor driving was recoverable by changing the scoring reference without
changing that road model. This does not rule out remaining objective, proposal,
observer or training limitations. The user's understanding-versus-deciding study
benefits from the control; it is not yet a validated three-way attribution of RL error.

**Historical next recommendation, now completed in the [separate replication](REFERENCE_REPLICATION_2026-10-29.md):** replicate the frozen tangent-versus-
true-tail comparison on the first 20 development tracks before tuning to track 2.
Declare both full conditions and a larger budget separately, retain the original
five as an explicit subset, and report the additional 15 separately as development
data. They are not a final holdout. A 40-episode comparison caps new evaluation at
40k native and 5.6M internal steps before any separately declared checks. Independent
planning streams/training seeds, strong baseline work and the established-optimizer
comparison still remain. Do not automatically activate all of these extensions.

## Verification, costs and artifacts

- **88 tests passed**, three upstream SWIG warnings. CLI discovery and local
  documentation links passed. Source, checkpoint, dependency and input hashes match.
- All **1,255** recorded primary-control observation states passed exact visible-
  prefix preservation, 100-sample tail length, finite geometry and anchoring checks;
  no visible-reference fallback. This was a data check, not new simulator work.
- One complete primary-control episode replayed exactly before new racing.
- New condition: **4,321 native frames**, **605,360 internal native physics steps**,
  **144,854 actor-forward examples**, 215.56 s evaluation wall time.
- Replay verification: **1,000 native frames**, **140,000 internal steps**,
  **33,500 actor-forward examples**, 48.62 s wall time.
- Total new campaign: **5,321 native / 745,360 internal frames**, **178,354 actor
  examples**, **277.38 s** including geometry checks/reporting; zero new training.
  Both total caps passed. Historical controls' compute is not counted again.

Public artifacts: [summary](../reports/reference_continuation_2026-10-29/summary.csv),
[all episodes](../reports/reference_continuation_2026-10-29/episodes.csv),
[executed behavior](../reports/reference_continuation_2026-10-29/executed_behavior.csv),
[paired differences, costs and provenance](../reports/reference_continuation_2026-10-29/analysis.json).
The frozen executable source and replay evidence remain in the campaign directory.
Final-test identities/results and private candidate-ranking data remain untouched.
