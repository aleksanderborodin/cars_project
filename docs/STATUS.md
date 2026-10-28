# Implementation and evidence status

Updated 2026-10-28. [Index](../README.md) · [Specification](EXPERIMENT.md) ·
[Roadmap](ROADMAP.md) · [Actual development results](PILOT_2026-10-28.md)

The first development experiment and control-audit follow-up are complete. The full paper
campaign has not started. User decisions are in [USER_INTENT.md](USER_INTENT.md);
attributed questions, ideas and probes are in [RESEARCH_LOG.md](RESEARCH_LOG.md).

## Implemented

| Component | Actual capability | Remaining limitation |
|---|---|---|
| Legacy environment/policies/IL/RL | Official standard dynamics, privileged expert, BC/DAgger, PPO/SAC | Legacy interfaces/splits are not the restricted research protocol |
| env/partitions.py | Disjoint namespaces, fixed dev/calibration manifests, test commitment, resumable training streams | Final-test runner intentionally absent; audit covers recorded historical training configs only |
| env/protocol.py | Restricted local road mask plus ego sensors; native RGB without numeric state; sanitized info; action repeat | Local geometry is an ideal sensor; fixed-view local RGB not implemented |
| models/dynamics.py | Observed-transition ridge dynamics, save/reload and integrated motion | Deterministic low-order model; no tire-state memory, ensemble or learned road belief |
| planning/ | Local pursuit, same-family fixed controller, 21-candidate/20-step search, shared road query for grip and margins | Single straight road continuation; no established search advantage, CEM or multi-world planning |
| research/data.py | Partition-checked collection, model fitting and development diagnostics | No large offline corpus, calibrated predictor or modern pretrained model |
| research/recurrent.py | Local/native CNN-LSTM PPO adapter; recurrent reload; training-state continuation with fresh episodes | 100k pilot did not learn useful driving; strong reproduction remains open |
| research/evaluation.py, eval/paired.py | Common development tracks, traces/costs, protocol checks, paired and crossed model/track intervals | Small/adaptive study; no final paper campaign |
| monitoring/ | One-way online oracle-ranking sidecar, prefix replay, private live report | User-only outputs; no agent/learning/selection/assistant access to real ranking values |
| Run artifacts | Immutable configs, source/dependency snapshots, hashes, datasets/models/traces | Ignored large runs/private reports need separate backup |

The legacy Policy.reset(track) / StateObs v1 path remains privileged. New
restricted experiments use the research CLI; do not relabel legacy results or
seen-inf collection as the new protocol.

## What is trained, scored and selected today

| Part | Current implementation | Objective or selection rule |
|---|---|---|
| Car dynamics | Ridge regression with 22 designed features and three outputs: changes in forward velocity, lateral velocity and yaw rate | Squared error of scaled one-step changes on train-track transitions, plus coefficient regularization; ridge 0.01 in the current runs |
| Imagined position/heading | Mathematical integration of the predicted velocities and yaw rate | No fitted position model or separate training objective |
| Visible road | Exact restricted simulator road mask and heuristic path reconstruction | No learned perception module in this local-geometry study |
| Unseen road | Constant terminal tangent and width prior | No learned road forecast, probability distribution or calibration |
| Online decision | Simulate 21 speed/gain feedback candidates for 20 decisions (1.6 seconds); execute the first action for four native frames (0.08 seconds), then replan | Minimize hand-designed cost: departure/clearance, center distance, lateral velocity and unknown-space penalties minus predicted path progress |
| Predictor configuration selection | Current feature design and ridge setting are engineering choices; variants were examined through development diagnostics/driving | No automated downstream predictor-selection study or regret objective implemented |

The dynamics fit uses 66 coefficients, including intercepts, and predicts the
next 0.08-second ego transition. It is learned from actual simulator transitions;
the designed features and integration are mathematical components. Training loss
does not optimize candidate ranking, race return or hindsight regret. The planner
cost is not the official reward. Good one-step error therefore does not establish
good candidate selection or race completion.

The three dynamics models are independent development replications; deployment
currently uses one at a time. They are not combined into a probabilistic ensemble.
The optional future model-selection study trains predictors normally and selects
representations/configurations through downstream validation driving. It remains
distinct from explicitly training a predictor with a decision/regret loss.

Hindsight regret compares the chosen behavior's measured utility with the best
utility among the same candidate set, under a specified horizon and utility.
Computing that difference entirely from the current model's predictions would
produce the same choice as maximizing its predicted utility and would not provide
an independent accuracy signal. Real candidate-monitor values remain user-only
and unavailable for agent analysis, training or selection. Hindsight knowledge of
the realized hidden world also differs from information available at decision time.

## Measurements

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

Full tables, costs, adaptive choices and uncertainty are in the pilot report.
No final-test identities/results or legacy unseen/hard results were inspected.

## Open work

- Steps 1–2 of the [next-experiment plan](NEXT_EXPERIMENTS.md) are complete.
  The proposed next 60k-frame pace/full-map diagnostic has not started.
- Competent learned baseline and faithful upstream reproduction before headline runs.
  The [baseline audit](BASELINE_AUDIT.md) records current implementation gaps.
- Better model/control robustness and recovery, using permitted feedback only;
  the user-only oracle is excluded.
- Same-planner global/local diagnostic; the expert alone does not isolate information.
- Learned road distributions, calibrated uncertainty, scenario MPC, adaptive/Bayesian
  behavior search and local visual perception remain proposals.
- TD-MPC2, DreamerV3, Valdi and other modern baseline reproductions are not executed.
- Final frozen study, sufficiently powered comparisons, paper and release remain open.
- Agentic evolution is future work; archived Rust/evolution tooling stays archived.

The full suite now passes **63 tests** (three upstream SWIG deprecation warnings).
The original cleanup had 47 retained tests. New coverage includes partition sealing,
local isolation, protocol equivalence, fit/reload, continuation, paired statistics
and synthetic ranking logic. Documentation links, CLI discovery and the dependency
lock also validate. These checks establish infrastructure behavior, not superiority.
