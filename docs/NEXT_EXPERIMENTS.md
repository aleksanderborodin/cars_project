# Next experiments after the first development pilot

Proposed 2026-10-28. [Index](../README.md) · [Specification](EXPERIMENT.md) ·
[Pilot evidence](PILOT_2026-10-28.md) · [Research log](RESEARCH_LOG.md)

**Architecture superseded:** the user's subsequent clarification makes a coded
simulator the main world model. Follow [SIMULATOR_SEARCH_PLAN.md](SIMULATOR_SEARCH_PLAN.md)
for active work. This document preserves the prior batch and conditional ideas;
it is not an instruction to train larger learned dynamics before implementing
programmed simulation. Partition, comparison and baseline-validity rules persist.

**Progress:** steps 1–2 are now complete; see the
[control-audit report](CONTROL_AUDIT_2026-10-28.md) for results and the proposed
60k-frame follow-up. The remaining steps below are proposed, not completed.

This records the original next execution plan. The user
requested the next moves; the ordering and budgets below are assistant proposals.
The immediate goal is to establish whether our working simulation/search loop adds
driving value, identify its limiting component, and build a credible comparator.
Training a special decision-loss predictor, hidden-road distributions, Bayesian
search and agentic evolution are not prerequisites.

## Evidence and scope

The revised search averaged 687.0 return and 20% completion across three fitted
models on 20 development tracks. Fixed-speed 60 achieved 613.4 and 30% completion.
The return difference interval includes zero. The 100k-frame recurrent policy
failed to learn useful driving. These results establish neither superiority nor
a competitive baseline. The full-map/raw-state expert's 90% completion shows
that these tracks are usually completable by that privileged controller; it does
not prove that our bounded candidate family can finish them.

Keep the current protocol and partitions: local radius 30 m, resolution 1 m,
ego velocity/yaw rate and previous action, continuous controls, repeat 4, and
official reward/physics/termination. Use the existing 20 development tracks for
debugging first. Calibration remains reserved; final test and legacy unseen/hard
data remain sealed. No agent reads real private ranking-monitor outputs.

## Ordered work and budgets

All frame limits below count native simulator frames. They exclude separately
reported internal-model transitions and synthetic checks. Episode limits imply
upper bounds because episodes can terminate before 1000 frames. Profile runtime
before promising wall-clock completion. No training was launched to write this plan.

| Order | Question and deliverable | Initial budget | Proceed criterion |
|---|---|---|---|
| 1 | Correct unknown-space dynamics semantics; paired rerun | Existing three models × 20 development tracks; at most 60k evaluation frames; no training | Synthetic consistency, deterministic replay, and an attributable before/after comparison |
| 2 | Does adaptive selection help within the same controller family? | 21 fixed controllers × 20 tracks; at most 420k evaluation frames; no training | Compare search with a development-selected constant from its exact candidate set; explain completion failures |
| 3 | Is the remaining limitation information, dynamics, or control? | One existing model; two privileged full-map controller conditions × 20 tracks, at most 40k additional frames | A documented diagnosis, including pace feasibility and representation confounds |
| 4 | Establish credible recurrent learning | One-seed 200k-frame engineering gate, then 500k and 2M checkpoints if correct; original-protocol reproduction separately budgeted | Verified training/reload, learning curves, sufficient training rather than a weak-baseline win |
| 5 | Make one targeted system improvement | At most two new dynamics variants initially; reuse training data before collecting more | Improvement in ordinary driving, with costs and failures reported; prediction error alone is insufficient |
| 6 | Check robustness beyond the debug panel | Three independent trained models, 200 common development tracks per condition | Stable comparison against tuned simple and learned references; then choose the paper study |

Steps 3 and 5 are conditional on the diagnosis. Preparation of step 4 can proceed
alongside the controller work. We do not launch every combination of model,
horizon, grid, sensor range and optimizer.

## 1. Correct an identified internal-model inconsistency

Source inspection and a synthetic probe found that rollout grip uses the road
raster with `cval=1` outside the square array. Unseen cells inside that square
are zero. Two points outside the circular sensor footprint therefore receive
different assumptions solely because of array padding: on a synthetic diagonal
road, positions (25,25) and (31,31) return grip proxies 0 and 1 respectively.
This is a confirmed implementation inconsistency; its driving impact is unknown.

Implement a shared road-hypothesis query: observed locations use the observation;
unobserved locations use the same assumed continuation for dynamics and planning
cost. Declare interpolation behavior at the visibility boundary. Initially retain
the constant-terminal-tangent hypothesis and current width prior. Do not use the
true hidden road or simply label all unknown space as road.

Test synthetic known-road/known-grass locations, unknown locations on/off the
assumed continuation, and rotated/edge cases within raster tolerance. Re-evaluate
the three existing mixed-data models with the unchanged bounded grid and horizon.
Preserve old artifacts and record a new implementation version/source hash. This
fix changes rollout interpretation, not the official simulator or observation
contract. A worse score does not justify retaining inconsistent semantics.

## 2. Separate search value from controller settings and failure modes

Keep fixed-speed 60 as a useful external reference. It is outside the revised
search grid and cannot alone isolate the benefit of choosing among that grid.
Add fixed speed/gain controls for the exact 21 combinations:
speeds 20,25,30,35,40,45,50 and gains 0.8,1.0,1.2. Use the same reconstruction,
continuation and low-level control implementation. A constant chooses one pair
for the whole campaign; search chooses a pair online.

Evaluate all 21 constants on the existing 20 development tracks. Select the one
with highest mean official return, breaking exact ties by completion and then
the recorded candidate order. Report all trials and completion, including any
return/completion tradeoff. This is development tuning, not an unbiased estimate
of the selected constant's generalization. Freeze that choice before the broader
200-track development comparison; those 200 tracks still are not final test data.

From public episode logs, distinguish playfield exits, on-road timeouts,
off-road timeouts, poor progress/stalling and failure to recover. Report progress,
speed and off-road occupancy alongside categories; predefine any stall threshold
before aggregating its counts. Do not claim these descriptive categories establish
causality. Check whether the speed cap itself prevents completion before blaming
road uncertainty or dynamics. If the family cannot achieve adequate pace, revise
the shared candidate family once and repeat the matched control study; count this
as another adaptive development round.

## 3. Diagnose with controlled, separately privileged experiments

Build a full-map version of the same planner and selected fixed controller, using
the same fitted ego dynamics, candidate parameters, horizon and scoring rules.
Keep the locally observed geometry identical where possible, replacing only the
unobserved continuation with known geometry. Declare any unavoidable changes in
path extraction or representation. Do not add wheel truth or exact-physics
rollout access. This is a separate privileged protocol, never a local competitor.

Compare ordinary whole-episode driving with the local counterparts. This does
not reopen the user-only candidate-ranking monitor: no real candidate oracle
returns, ranks or regret are accessed or recreated for selection.

- If the matched full-map condition improves driving, missing information or
  its representation is a plausible bottleneck; it does not prove forecasts will
  recover the gain, or isolate hidden-road information if representation changed.
- If both versions fail, first check pace, recovery, dynamics and the scoring
  objective. Failure is not proof that model-based control is unworkable.
- If a fixed controller matches search, examine the model and score before
  increasing search compute. The current expert alone cannot answer this question.

## 4. Build the learned baseline properly

Use the audited [pinned RL Zoo recurrent-CarRacing configuration](https://github.com/DLR-RM/rl-baselines3-zoo/blob/61956d90d9f8ac126043ed4024efa12a462d97e1/hyperparams/ppo_lstm.yml)
as a documented starting reference. It specifies 64×64 grayscale input, two-frame
stacking, eight environments, a 128-unit LSTM and four million training steps;
it is materially different from our short engineering pilot. Its training reward
normalization is a declared learner transformation, not a change to the official
environment reward; evaluation always reports raw official return.

Maintain two distinct experiments:

1. **Matched local adaptation:** our permitted local observations and repeat 4,
   with documented encoder/history and recurrent-PPO settings. Run one seed to
   200k native frames, then inspect scheduled 500k and 2M checkpoints if integration
   is correct. These are milestones, not a promise of convergence. Retain poor
   runs and log each tuning trial. Continue new resumable runs; the original
   100k checkpoint lacks the complete continuation state and is inference-only.
2. **Original-protocol reproduction:** audit upstream wrappers, action frequency,
   termination and dependency versions, then implement the published recipe and
   its full four-million-step budget. Record actual native frames separately from
   learner steps. Do not silently apply our repeat-4 protocol or label the local
   adaptation a faithful reproduction. Profile a short run before budgeting it.

Validate recurrent episode resets, action scaling, normalization persistence,
checkpoint reload/continuation and learning curves. Use development results for
selection. Compare sample-efficiency curves and account for all model collection,
labels, tuning and inference costs; our current 30-episode dataset is not a
frame-matched comparison with a multi-million-frame policy.

A reproduced recurrent policy is the first learned comparator, not by itself
evidence of SOTA. Next audit Valdi, the directly relevant actor-free CarRacing
reference, and choose a feasible modern world-model comparator from the
[reference roster](REFERENCES.md). Keep their original protocols separate from
matched adaptations. Freeze the eventual comparator roster before final test.

## 5. Improve the component supported by evidence

If permitted rollout diagnostics and ordinary driving point to dynamics, compare
the ridge model with a small MLP and, at most, a second MLP using four decisions
of permitted history. Use ordinary prediction losses and the same immutable
training corpus first; split by whole track and reset history each episode.
The history model tests temporal information separately from nonlinear capacity.
Choose exact architecture/optimizer and validation schedule in a run manifest
before fitting, not after seeing its outcomes.

Use one-step/multi-step errors as diagnostics, then evaluate driving with a fixed
planner. Logged-action rollouts with observed future grip do not establish
planning validity. If coverage is inadequate, collect a shared train-only corpus
with a declared 100k-native-frame target, record actual boundary overshoot, and
refit all compared predictors. Do not give only the proposed model extra data.

If failures instead implicate the objective, recovery or candidate pace, address
that component first with a controlled comparison. Avoid introducing a new model,
new score and new optimizer together and attributing all improvement to prediction.

## 6. Broader development, then the paper decision

After choosing configurations on the 20-track debug panel, compare the fixed
reference, local heuristic, search and recurrent policy on all 200 development
tracks. Use three independent training seeds for exploratory learned conditions;
deterministic references do not gain fake replications. These data remain
development data even if some tracks have not previously been evaluated.

Report mean return, completion, failures, progress, latency and cost; show learning
curves and paired track comparisons with training-seed uncertainty. Retain the
provisional practical target of +10 return without more than a two-percentage-point
completion loss from [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md). A small pilot cannot
resolve that completion margin. Use observed variance to plan precision and freeze
the final analysis before test; do not manufacture a significance gate from 20 tracks.

Only then prioritize additional complexity: road scenarios if information appears
limiting, or continuous/adaptive candidate search if the current family/optimizer
appears limiting. Compare one versus multiple worlds and fixed versus adaptive
search separately at matched rollout/latency budgets. Predictor-space selection,
regret-based objectives and agentic evolution stay optional studies; existing
private monitor results remain unavailable for all such work.

Headline evaluation remains five independent training seeds and a target of 500
common test tracks per model/primary condition, subject to the pre-test precision
plan. Freeze all methods, checkpoints, contrasts, budgets and failure handling
first. A simulation-only paper is within scope; SOTA and novelty remain unproven.
Before committing to a method paper, identify a substantive result beyond the
established idea of planning in a world model: for example, a reproducible driving,
data-efficiency or inference-cost advantage under the appropriate controls. If
that advantage disappears against strong comparators, revise the contribution
and report the negative evidence instead of weakening the comparison.

## Confidence and immediate deliverables

**High confidence:** correct the demonstrated semantics inconsistency, use a
matched fixed-candidate reference, train a credible baseline, preserve all failures
and keep final test sealed. These improve the validity of the experiment even if
our method loses.

**Medium confidence:** the full-map diagnostic and temporal dynamics comparison
will help locate a bottleneck; interpretation depends on controlled inputs and
adequate controller pace.

**Unknown:** whether these changes improve driving, whether probabilistic roads
or adaptive search help, and whether a publishable competitive advantage emerges.
Do not turn confidence in the experimental design into confidence in winning.

The first execution batch (steps 1–2) is complete: a tested world-query correction,
paired three-model reruns, the 21 fixed-controller table, and a public failure
report. It used 473,737 evaluation frames and no new model training. The baseline
audit is prepared. The control-audit report records the resulting decision and
next budget; subsequent experiments have not started.
