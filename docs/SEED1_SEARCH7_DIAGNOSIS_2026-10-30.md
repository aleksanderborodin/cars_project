# Why seed1 search7 lost return: accounting and observed behavior

Reviewed 2026-10-30. [Paper ledger](PAPER_CAMPAIGN_2026-10-30.md) ·
[Public analysis](../reports/seed1_search7_diagnosis_2026-10-30/summary.json) ·
[Paired episodes](../reports/seed1_search7_diagnosis_2026-10-30/paired_episodes.csv)

**Origin:** the user asks how simulated selection can damage driving return.
The assistant performs a post-result analysis of the completed direct/search7
panels and permitted executed observations. This is descriptive development
analysis, not a new intervention, checkpoint selection or a causal identification.
No private ranks/oracle outcomes, held-out data or new simulator steps are used.
The queued conditions and their settings remain frozen.

## What actually lowered the reward

Same frozen training-seed1 actor, same 200 development tracks:

| Metric | Direct | Search7 |
|---|---:|---:|
| Mean official return | 865.42 | 842.87 |
| Completion | 30.5% | 38.5% |
| Mean fraction of distinct track tiles visited | 96.05% | 93.80% |
| Mean native steps | 950.565 | 951.735 |
| Mean off-road native steps | 15.455 | 10.770 |
| Playfield exits | 0 | 0 |

For these 400 no-exit episodes, the identity
`return = 1000 * visited_tile_fraction - 0.1 * native_steps`
matches every row within 2.1e-11. It is verified from the installed official
reward and our saved metrics; no assumption about a separate completion bonus is
made. The **−22.5561** mean-return change decomposes exactly into **−22.4391 from
less road coverage** and **−0.1170 from more native steps**. About 99.5% of the
loss is coverage. Planning wall time is not charged by this paused-environment
evaluation; the loss is not a latency penalty.

Search gains 29 completions and loses 13, giving 16 net additional completions.
On the 110 tracks where neither method completes, however, search loses 43.90
return on average. That group contributes −24.15 to the full-panel mean change;
the completion gains do not offset it. Across all tracks, search wins 75, loses
118 and ties seven. The nominal paired-track 95% interval for the mean difference
is [−33.37, −12.69], using 5,000 resamples and seed20261030. This is uncertainty
over this policy's development tracks, not over independent training seeds or a
final-test claim.

## A concrete failure, and why completion hides it

In post-selected episode188, both methods time out. Direct visits 73.18% of the
track and scores 631.78; search visits 30.03% and scores 200.29, a −431.49 change.
Search earns its last new-tile reward by native frame452, spends 50.8% of the
episode below 2 m/s, and has **zero off-road frames after frame500**. It is almost
stationary late in the episode while on the road. Completion labels both runs
unsuccessful and cannot express their very different coverage.

Other losses differ: episode163 stops gaining new tiles after frame588 but keeps
moving rapidly and accumulates 407 off-road frames. Thus a single story that
every loss is excessive caution would be wrong. The recorded four examples are
explicitly chosen after observing losses, not an independent confirmation.

Overall executed speed averages 54.96 m/s direct versus 50.26 with search, and
mean brake input is 0.00785 versus 0.01673. Those are trajectory associations:
the methods visit different states, so they do not establish braking as the
isolated cause.

## Why keeping the base candidate does not guarantee real improvement

Candidate zero is the unchanged proposer. Selection can therefore prefer a
candidate with no greater **predicted internal cost** than that modeled base.
It cannot guarantee a greater **actual full-episode official return**.

The frozen [rollout scorer](../src/cars/planning/policy_rollouts.py) uses estimated
centerline progress, clearance, center distance, slip and unknown-area penalties,
not the official distinct-tile episode reward. It has a 20-decision/1.6-second
horizon and no learned terminal value. Only the first action is executed before
replanning. Avoiding local cost can therefore beat moving in the internal score
while producing prolonged low progress in the real episode. Adding the same
constant per-step time penalty to every equal-length rollout would not change
their ranking; that alone is not a solution.

The [programmed model](../src/cars/models/physics.py) reuses vehicle equations but
estimates hidden wheel/contact state and rebuilds solver state. Its road outside
the observed footprint is a tangent prior. At 50 m/s, 1.6 seconds spans about 80 m,
well beyond the 30 m sensor radius. The recurrent actor also sees generated future
observations when candidates are rolled out. Errors in those elements can change
which candidate appears best. Search settings developed with seed0 need not
transfer to another learned actor's behavior. More samples may help or may exploit
model/score errors more strongly; monotonic real improvement is not guaranteed.

These are **known structural differences and plausible mechanisms**, not an
experimental attribution of this loss to one component. Exact reward accounting
and observed stalls do not identify whether the utility, model, imagined policy
responses or proposal family dominates. That needs a separately declared controlled
study after the current comparisons, with ordinary permitted observations and
no access to the user-only oracle stream. No module is declared solved or universally
limiting from these results.
