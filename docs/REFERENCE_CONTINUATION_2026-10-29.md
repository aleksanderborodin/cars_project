# Reference continuation: planned contrast and execution record

2026-10-29. [Index](../README.md) · [Intervention design](INTERVENTION_STUDY.md) ·
[Preceding results](INTERVENTION_RESULTS_2026-10-29.md) · [Status](STATUS.md)

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
`runs/campaigns/reference_continuation_20261029/status.json`. Completion/result
interpretation will be recorded below after all checks and five races finish.
