# Implementation and evidence status

**Current execution:** fresh local replications are complete: selected seed 1 869.23/32%, seed 2 −17.89/0%. The latter failed task competence at the declared review and remains in the active [paper comparison queue](PAPER_CAMPAIGN_2026-10-30.md). No training is active. The [comparator/novelty review](COMPARATOR_REVIEW_2026-10-30.md) corrects missing MRS.Q/APC-RL prior art and makes strong external comparisons explicit blocking gates. The ongoing dataset is exploratory mechanism evidence, not a SOTA result; 114 tests passed at the last code change.

**External comparison preparation:** [MR.Q/MRS.Q](MRQ_MRSQ_AUDIT_2026-10-30.md)
and [MPPI](MPPI_READINESS_2026-10-30.md) reference sources are now pinned and
hash-checked. Source inspection identified action/history/storage and stochastic
evaluation constraints, plus required accounting for MPPI's synthesized control.
No external algorithm was imported, integrated or executed; runtime and competence
gates remain open. Current evaluation source and measurements are unchanged.

Updated 2026-10-30. [Index](../README.md) · [Specification](EXPERIMENT.md) ·
[Roadmap](ROADMAP.md) · [Latest development results](SEQUENCE_SEARCH_2026-10-29.md)

The first development experiment and control-audit follow-up are complete. The final held-out paper
campaign has not started. User decisions are in [USER_INTENT.md](USER_INTENT.md);
attributed questions, ideas and probes are in [RESEARCH_LOG.md](RESEARCH_LOG.md).

**Active architecture correction:** the user requires a programmed simulator as
the main world model. [SIMULATOR_SEARCH_PLAN.md](SIMULATOR_SEARCH_PLAN.md) is the
active plan. Ridge dynamics below describe the existing comparison baseline,
not the intended main engine. The coded-physics prototype, bounded/wider campaign
and seven-candidate follow-up are complete. The full-map diagnostic and trainer
profiling are also complete. The first configured 2M-frame driver block completed
and regressed after its selected 1M checkpoint; unchanged training is stopped.
The six-condition frozen-checkpoint comparison and three fixed-correction controls
are complete. Search 21 improved this policy instance; a separate KL training
trial was worse and not adopted. The [200-track comparison and lower-learning-rate
probe](BROAD_RESULTS_2026-10-29.md) are complete; no jobs from this batch remain active.
Search 21 achieved 884.48/72% versus direct 862.42/35.5%, with 362.11 ms p95 latency.
The lower-learning-rate trial's best checkpoint was 692.18/8% on 50 tracks and was
not adopted. The [paper blueprint](PAPER_BLUEPRINT.md) now consolidates the proposed
three-stage intervention study and explicitly attributed ideas. No trained visual/
full-map diagnostic matrix, GPU simulator backend or agentic evolution has been implemented.

**Current work:** [controlled geometry interventions](INTERVENTION_STUDY.md) are
implemented: independent visible-reference and hidden-road substitutions around
the same frozen local actor, with unchanged objective weights/visibility penalty.
The five-track/four-cell development pilot is complete: baseline 906.99/80%;
visible-reference oracle 758.96/0%; hidden-road oracle 429.64/0%; both 594.21/0%.
All oracle conditions timed out on road, with lower speeds/more braking. A separate
15-window executed-action audit found better motion prediction with oracle road
(1.6-second velocity RMSE 0.00536 versus 31.35 m/s). These narrow results motivate
an objective/interface audit, not component-adequacy or SOTA claims. See the
[complete intervention report](INTERVENTION_RESULTS_2026-10-29.md). No jobs remain
active from that completed batch. The [reference-continuation control](REFERENCE_CONTINUATION_2026-10-29.md)
is now implemented with exact visible-prefix preservation, a declared replay gate
and five-track budget. It is complete: true continuation scored **876.29/80%**
versus the tangent control's **594.21/0%**, improving all five returns. Original
local search remains higher at **906.99/80%**; one on-road timeout accounts for
the new condition's mean deficit. This isolates a reference-tail limitation for
this controller, not a general information/SOTA advantage. All jobs in this batch
are complete. The [20-track paired replication](REFERENCE_REPLICATION_2026-10-29.md)
is also complete: true tail **813.85/60%**, tangent **527.60/0%**, historical local
search **885.70/70%**. The additional fifteen retain a +287.64 mean-return gain
versus tangent, with one counterexample, but −85.57 versus local search. Ten exact
replays passed. No jobs remain active; true tail remains a privileged diagnostic.
The subsequent equal-work optimizer comparison is reported below. Baseline
competence remains open; no larger sweep is activated.
The actor receives only its original local fields. Global training stays sealed.

**Latest completed experiment:** [matched sequence shooting/iCEM](SEQUENCE_SEARCH_2026-10-29.md)
finished all twenty tracks per new method. Colored shooting: **884.86/70%**;
iCEM adaptation: **634.13/15%**; historical knot search: **905.94/85%**. Primary
paired return difference −250.72 [−359.57, −147.31], with −240.99 on additional18.
Retain original knot search; neither new variant is promoted. All five replays and
100 tests passed. Total including smoke: 40,972 native / 17,220,000 internal steps,
zero new training. No jobs remain active from that completed batch. Baseline competence is the next priority;
the [factory audit](BASELINE_AUDIT.md) and [Valdi audit](VALDI_AUDIT.md) record the
remaining budget and protocol gaps. These are development findings, not SOTA.

## Implemented

| Component | Actual capability | Remaining limitation |
|---|---|---|
| Legacy environment/policies/IL/RL | Official standard dynamics, privileged expert, BC/DAgger, PPO/SAC | Legacy interfaces/splits are not the restricted research protocol |
| env/partitions.py | Disjoint namespaces, fixed dev/calibration manifests, test commitment, resumable training streams | Final-test runner intentionally absent; audit covers recorded historical training configs only |
| env/protocol.py | Restricted local road mask plus ego sensors; native RGB without numeric state; sanitized info; action repeat | Local geometry is an ideal sensor; fixed-view local RGB not implemented |
| models/dynamics.py | Observed-transition ridge dynamics, save/reload and integrated motion | Deterministic low-order model; no tire-state memory, ensemble or learned road belief |
| models/physics.py, planning/physics_search.py | Upstream vehicle equations and Box2D in independently reconstructed worlds; observation-corrected wheel estimate; simulated candidate selection | Approximate five-point wheel contact and latent-state initialization; no exact evaluation-state cloning |
| planning/ | Local pursuit, same-family fixed controller, 21-candidate/20-step search, shared road query for grip and margins | Single straight road continuation; fixed-controller gains unestablished; adaptive sequence search is separate; no multi-world planning |
| research/data.py | Partition-checked collection, model fitting and development diagnostics | No large offline corpus, calibrated predictor or modern pretrained model |
| research/recurrent.py | Local/native CNN-LSTM PPO adapter; recurrent reload; training-state continuation with fresh episodes | 100k pilot did not learn useful driving; strong reproduction remains open |
| planning/global_search.py | Explicit full-map geometry diagnostic with unchanged estimated vehicle state | Also changes centerline reconstruction; privileged, not equal-input superiority |
| planning/intervention.py, visible_reference.py | Separate visible objective-reference and hidden-road substitutions; exact local negative control; actor input/memory isolation tests | Privileged fixed-controller geometry diagnostic; not trained V/L/F policies or a visual-perception finding |
| planning/reference_extension.py | Privileged true-road tail appended to the unchanged visible reference; fixed 100 m/1 m extent and sampling | Reference-continuation diagnostic; utility/other planner limitations remain |
| research/driver_training.py and driver_* | Configured PPO, deterministic pooling, reward normalization, post-update checkpoints, native-frame schedule, CPU/GPU profiles | Local adaptation; original 1M checkpoint retained; KL-stop and lower-LR repairs failed to improve it; convergence/competitive baseline not established |
| env/native_recipe.py, research/native_* | Audited native resize64/grayscale/stack2/repeat1 CNN-LSTM; immutable train/checkpoint/evaluate path; CPU/GPU profiling and full-episode replay | One-seed 4M native campaign complete; selected 1M 843.82/38%, later regression; convergence unestablished; separate from local repeat-four search |
| planning/policy_search.py | Frozen recurrent feedback proposals, smooth residuals, isolated imagined memory, coded simulation selection and random control | Search 7/21 gains on 200 development tracks for one frozen checkpoint; high cost, no multi-seed evidence |
| planning/policy_rollouts.py, sequence_optimizer.py, policy_sequence_search.py | Shared feedback rollouts, matched colored shooting and two-batch policy-guided iCEM; owned first-state commits | Complete 20-track negative result; neither new variant promoted; adaptation, not original-protocol reproduction |
| planning/occupancy.py, policy_controls.py | Equivalent sparse occupancy renderer and frozen-policy fixed offsets | Sampled exact renderer parity; three fitted offset controls do not establish all cheap alternatives are inferior |
| research/evaluation.py, eval/paired.py | Common development tracks, traces/costs, protocol checks, paired and crossed model/track intervals | Small/adaptive study; no final paper campaign |
| monitoring/ | One-way online oracle-ranking sidecar, prefix replay, private live report | User-only outputs; no agent/learning/selection/assistant access to real ranking values |
| Run artifacts | Immutable configs, executable frozen campaign source, source/dependency snapshots, hashes, datasets/models/traces | Ignored large runs/private reports need separate backup |

The legacy Policy.reset(track) / StateObs v1 path remains privileged. New
restricted experiments use the research CLI; do not relabel legacy results or
seen-inf collection as the new protocol.

## What is trained, scored and selected today

For the new primary coded-physics variant, vehicle dynamics are **not trained**.
The observer carries simulated wheel state from executed history and corrects
the next hull velocity/yaw from observations. Decisions come from rolling out
the candidates, not a learned behavior-value head.

| Part | Active implementation | Training or selection rule |
|---|---|---|
| Vehicle dynamics | Locked upstream vehicle mechanics in reconstructed independent worlds; approximate wheel/contact observer | No fitted dynamics in the primary method; no true wheel/solver-state access |
| Behavior proposer | Frozen selected recurrent PPO checkpoint with local mask, ego and previous-action/history inputs | Ordinary PPO training and standalone development checkpoint selection; weights stay fixed during search |
| Visible road | Exact restricted road mask and heuristic reference reconstruction | This study bypasses RGB perception; no learned visual module |
| Unseen road | Terminal tangent and fixed-width prior | One deterministic hypothesis, not a learned or calibrated distribution |
| Baseline search | 7 or 21 smooth three-knot residual behaviors around imagined feedback actions; 20 decisions per rollout | Select minimum programmed model cost; retain the base policy among evaluated candidates |
| Sequence comparison | 21 full residual sequences in 13+8 batches; colored shooting versus a policy-guided iCEM adaptation | iCEM fits the online control-proposal distribution; actor/dynamics weights remain frozen |
| Execution | Execute the selected first control for at most four native frames, then observe and replan | Commit its first physical-state estimate; update real recurrent memory only from real observations |
| Predictor/configuration selection | Current engineered settings and explicitly recorded development trials | Future normally trained predictors may be selected by downstream driving; no private regret/rank feedback |

The search utility is not official reward: it combines predicted path progress,
road departure/clearance, centre distance, lateral velocity and unknown-space costs.
A better internal objective value need not improve real racing. The 20-decision
rollout covers 1.6 s; the real action interval is 0.08 s. The environment currently
waits synchronously, so latency above that interval is not a real-time control result.

The earlier ridge baseline remains available as a separate comparison. It fits
66 coefficients from 22 designed features to scaled one-step ego changes, with
ridge regularization 0.01 in its recorded runs. Its three development fits are
independent instances, not a probabilistic ensemble. They do not replace the
programmed engine or train on race return, candidate ranking or hindsight regret.
The optional predictor-selection study remains separate from special decision-loss
training, as the user requested.

Hindsight regret compares the chosen behavior's measured utility with the best
utility among the same candidate set, under a specified horizon and utility.
Computing that difference entirely from the current model's predictions would
produce the same choice as maximizing its predicted utility and would not provide
an independent accuracy signal. Real candidate-monitor values remain user-only
and unavailable for agent analysis, training or selection. Hindsight knowledge of
the realized hidden world also differs from information available at decision time.

## Measurements

- [Matched sequence-search comparison](SEQUENCE_SEARCH_2026-10-29.md): iCEM minus
  colored shooting −250.72 [−359.57, −147.31], 3/20 versus 14/20 completions.
  Colored also loses 21.08 versus historical knot search (17/20). Additional18
  retain the negative primary effect. iCEM has fifteen timeouts/two playfield
  exits; all failures retained. Both new p95 latencies exceed 380 ms. The trace
  audit is descriptive and uses only executed public trajectories, not oracle ranks.

- [20-track reference replication](REFERENCE_REPLICATION_2026-10-29.md): true tail
  +286.25 mean return versus tangent, nominal track interval [231.59, 343.19],
  12/20 versus 0/20 completions; gains on 19/20 tracks. Still −71.86 versus local
  search and two fewer finishes. Additional15 analyzed separately; one frozen
  actor/stream, development only. Total 38,031 native / 5,326,720 internal frames,
  zero new training, ten exact episode replays, 90 tests, 32.71 minutes.

- [Reference-continuation pilot](REFERENCE_CONTINUATION_2026-10-29.md): +282.09
  return versus tangent control, 0/5 to 4/5 completions. Mean −30.70 versus original
  local search; no overall local-search win. New cost including replay: 5,321 native
  frames and 745,360 internal native physics steps; zero training.

- Four-cell/five-track geometry pilot and executed-action prediction audit complete;
  see the [new report](INTERVENTION_RESULTS_2026-10-29.md). Better oracle-road
  prediction did not translate to better driving. Cost: 19,447 native evaluation
  frames and 2,729,240 internal native physics steps including the follow-up audit.

- Latest 200-track comparison: direct 862.42/35.5%, fixed correction 865.95/39.5%,
  search 7 873.05/58.5%, search 21 884.48/72%. Search 21 minus direct return +22.06
  [10.18, 33.35], exploratory track-only interval; p95 latency 362.11 ms.
- Lower-LR probe best 692.18/8% on 50 tracks; not adopted. Costs 2,006,916 training,
  199,612 evaluation and 8 reload-check frames. See the latest completed report.

- Frozen learned policy on 20 common development tracks: direct 863.55/40%;
  search 7 gives 885.70/70%, search 21 gives 905.94/85%. The latter return gain is
  +42.39 [14.36, 75.60], exploratory track-only interval; no SOTA/general claim.
- Best-return fixed-correction control: 860.45/60%, about 1.5 ms p95. Search 21
  remains higher on this panel; better cheap controls have not been excluded.
- Sparse imagined occupancy preserves two full race replays exactly and reduces
  median sampled times roughly 30–35%; still above 80 ms, not a real-time result.
- Fresh target-KL0.02 trial: best 697.04/4% on 50 tracks; not adopted. Cost
  2,006,884 training frames plus 195,829 evaluation frames and 8 reload-check frames.

- Full-map coded search: 923.15 / 923.60 mean return and 100% completion for
  7 / 21 candidates on 20 common tracks. This changes geometry/reconstruction,
  not vehicle-state access; no equal-information or SOTA claim.
- New configured driver checkpoints on 50 common development tracks: 200k gives
  584.9 return / 2% completion; 500k gives 734.6 / 6%; 1M gives 858.1 / 30%.
  The 2M checkpoint regressed to 779.2 / 2%; the predeclared rule selected 1M.
  Training stopped at the review. These are one-seed development results.
- Programmed physics with wider 21-candidate search: 737.3 mean return, 45%
  completion, 128.5 ms p95 decision latency. The seven-candidate variant: 697.8,
  40%, 48.9 ms. Best fixed controller across the wider family: 716.1 and 20%.
  Mean-return difference intervals include zero; no reliable advantage established.
- Programmed bounded search: 653.6 and 20%. The physics backend improves one-step
  ego prediction substantially on the same observed transitions but this does
  not establish better long-horizon decisions or driving.
- These two new campaigns used 247,428 native evaluation frames and 17,774,400
  internal physics frames, with zero new training. All 280 episodes reuse 20
  development tracks. Deterministic replay passed; isolated timings confirmed
  the seven-candidate cost reduction. Verification costs are reported separately.
- Original revised mixed-data/bounded-grid search: mean return 687.0 across three fitted
  models and 20 common development tracks; 20% completion in each run.
- Same-family fixed-speed reference: 613.4 return, 30% completion. The search
  difference interval includes zero and completion is lower: no established win.
- Corrected road query: 681.9 mean return, still 20% completion per fitted model.
  Best of 21 matched constants: speed50/gain1.2, 716.1 return and 20% completion.
  Selection used these development tracks; no reliable advantage is established.
- Control audit: 480 episodes / 20 common tracks, 473,737 native frames; no new
  model training. Across 60 search episodes, 38 on-road timeouts, nine off-road
  timeouts, one playfield exit and 12 completions. See the
  [follow-up report](CONTROL_AUDIT_2026-10-28.md) for limits and costs.
- Full-map/raw-state expert: 888.5 return, 90% completion; privileged reference.
- Recurrent PPO: 100,349 native training frames, poor driving; not a strong baseline.
- Native RGB: 2,048-frame training smoke, two-track evaluation and recurrent reload
  passed integration checks; not a visual performance result.
- Monitor isolation: actions, rewards and permitted observations identical over
  20 episodes with/without monitoring. Only public technical status/costs checked;
  real oracle-ranking values were not read.

Full tables, costs, adaptive choices and uncertainty are in the
[200-track report](BROAD_RESULTS_2026-10-29.md), the
[sequence-search report](SEQUENCE_SEARCH_2026-10-29.md) and linked earlier reports.
No final-test identities/results or legacy unseen/hard results were inspected.

## Open work

- The [next-moves plan](NEXT_MOVES.md) gives the active order, capped development
  experiments, stronger-driver training proposal and scientific decision gates.
  Its geometry diagnostic and first training block are complete. The selected
  checkpoint search comparison and cheap-control follow-ups are complete. The KL
  repair trial did not improve the baseline. The broader comparison and lower-LR probe are complete; the final study remains unstarted.
- Steps 1–2 of the [next-experiment plan](NEXT_EXPERIMENTS.md) are complete.
  Its 60k-frame pace/full-map proposal is deferred by the coded-simulator priority.
- Competent learned baseline and faithful upstream reproduction before headline runs.
  The [baseline audit](BASELINE_AUDIT.md) records current implementation gaps.
- Better model/control robustness and recovery, using permitted feedback only;
  the user-only oracle is excluded.
- Trained visual/local/full-map decomposition remains open. Completed fixed-controller
  substitutions and the reference-tail diagnostic do not replace those trained rows.
- Learned road distributions, calibrated uncertainty, scenario MPC, Bayesian
  behavior search and local visual perception remain proposals. The bounded iCEM
  adaptation completed its bounded development comparison and was not adopted.
- TD-MPC2, DreamerV3, Valdi and other modern baseline reproductions are not executed.
- Final frozen study, sufficiently powered comparisons, paper and release remain open.
- A [task-useful independent simulator](DECISION_USEFUL_SIMULATOR.md), GPU backend
  and driving-guided parameter/code adaptation are documented future ideas, not
  implemented. Realism is not an acceptance criterion for that proposed variant.
- Agentic evolution is future work; archived Rust/evolution tooling stays archived.

The latest completed full suite passed **100 tests** (three upstream SWIG deprecation warnings).
Pytest discovery is now restricted to active `tests/`, excluding saved run snapshots.
The original cleanup had 47 retained tests. New coverage includes partition sealing,
local isolation, protocol equivalence, fit/reload, continuation, paired statistics
and synthetic ranking logic. Documentation links, CLI discovery and the dependency
lock also validate. These checks establish infrastructure behavior, not superiority.
