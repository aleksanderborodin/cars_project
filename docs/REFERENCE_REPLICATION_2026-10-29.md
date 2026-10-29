# Frozen reference-tail replication on 20 development tracks

[Index](../README.md) · [Specification](EXPERIMENT.md) · [Pilot](REFERENCE_CONTINUATION_2026-10-29.md)

## Origin, purpose and status

**User instruction:** continue the controlled intervention work; document choices,
uncertainty and idea origin. **Assistant design:** replicate the two unchanged
reference-tail conditions before tuning to the remaining pilot failure. This is a
fixed-controller geometry diagnostic within the user's understanding-versus-deciding
study. It does not train or establish the full V/L/F policy comparison.

**Validated and ready to launch.** The completed five-track pilot found +282.09 mean
return for true continuation versus a tangent tail with the same oracle road model,
but −30.70 versus original local search. This motivates replication, not a favorable
subset or a new parameter sweep. No production controller code changes in this batch.

Frozen declaration: [manifest](../configs/campaigns/reference-replication-2026-10-29.json),
SHA256 `ed70036b319e18531023e9ce8545deb87df90114dd0dbcccfbea7718aa135f2e`. New run IDs and files are distinct from every completed pilot.

## Conditions and information

Both newly evaluated conditions use the same frozen local recurrent actor,
checkpoint `frames_001007379` from `driver_main_20261028_s0_to2m`, the same
observation-initialized programmed vehicle model and oracle road outside the
current footprint. Actor inputs/history remain local; no true wheel/solver state,
environment clones or private candidate rankings are available.

| Condition | Objective reference | Road model | Role |
|---|---|---|---|
| Tangent | True visible prefix plus 100 m straight tangent | Oracle continuation, original raster inside footprint | Primary control |
| True tail | Exactly the same visible prefix plus 100 m true directed arc | Identical oracle road model | Only changed module |
| Local search | Reconstructed local path plus tangent | Local raster plus tangent prior | Historical, different-information context |

Seven feedback candidates, 20 imagined decisions, action repeat four, planning
seed zero, three residual knots, sparse imagined observations, score weights,
observer and root outside-footprint penalty all remain unchanged. True tail keeps
one-metre sampling. Both reference variants affect progress and centre-distance
scoring together; this experiment cannot identify either cost term alone.

The context reuses only the first 20 episode trajectories of
`broad_driver_20261028_search7`; its full-run metrics/config/cost are pinned for
integrity. It is not a third newly evaluated condition or an equal-information
control. Its historical timing cannot support a paired timing superiority claim.

## Fixed sample, analysis and budgets

- First 20 tracks in the existing development manifest, fixed before this run;
  same ordered tracks in every condition. The declaration records all 20 seeds.
- Report all20, original5 and additional15 separately, regardless of outcomes.
  Original5 informed this intervention. Additional15 were already used by the
  historical local controller and remain development data, not a final holdout.
- Primary contrast: true-tail minus tangent official return, paired by track.
  Also report completion, progress, off-road frames, playfield exits, sampled
  speed, braking, reference fallbacks and latency. Preserve all failures.
- Nominal paired-track bootstrap: 5,000 draws, report seed zero, all20/additional15
  only; original5 stays descriptive. These intervals condition on one frozen actor
  and one planning stream, with no selection adjustment or training-seed uncertainty.
  The three overlapping panels are not independent replications or formal tests.
- Forty episodes maximum: **40,000 new native environment frames / 5,600,000
  internal native physics steps**, zero new training. The original five episodes
  per condition are replay checks within these 40 episodes, not extra races.
- Serial tangent then true-tail execution, one CPU thread, no competing project
  training/racing. No early favorable stopping, tuning during the run, automatic
  expansion to 200, or final-test access. Stop and retain artifacts on integrity
  failure; a repaired or expanded run requires a new declaration.

**Assistant confidence:** high in the isolation and replication design after the
existing geometry/candidate/memory tests; uncertain whether the effect extends to
15 additional tracks or improves over local search. Twenty is a bounded diagnostic
budget, not a power calculation or a headline paper sample. One stream and a
locally trained proposer remain limitations even if the gain repeats.

## Execution and checks

```bash
uv run python scripts/run_reference_replication.py --manifest configs/campaigns/reference-replication-2026-10-29.json
```

The runner copies executable sources and scripts, verifies checkpoint, partition,
dependency and input hashes, and records child commands and live technical status.
As the original five episodes complete in each condition, compare every recorded
array with its pinned pilot counterpart except wall-clock timing. Stop on mismatch.
Observations, actions, rewards, termination, frame counts and off-road indices must
match exactly. Check complete episode counts and costs before publishing summaries.

Campaign directory: `runs/campaigns/reference_replication_20261029`.
Report destination after completion: `reports/reference_replication_2026-10-29`.
The report includes per-panel summaries, all paired episode results, executed
behavior, nominal intervals, a paired-return figure and source/artifact provenance.
No monitor rank/regret outputs or reserved test results are read.

Validation before launch: **90 tests passed** (three upstream SWIG warnings);
runner/reporter/evaluator CLI discovery, pinned input/configuration checks and all
32 local documentation files passed. No new environment work in these checks.

## Decision after results

If the tail correction helps the additional15, retain it as evidence that objective
compatibility mattered in this particular oracle intervention. If it fails, report
that rather than expanding the pilot claim. Either way, it does not prove perception
or prediction adequate, solve action selection, or establish a SOTA advantage.
Review the complete failures and costs before choosing one next diagnostic. The
competent-policy and established-optimizer comparisons remain necessary; neither
new global training nor a search sweep is implicitly activated here.
