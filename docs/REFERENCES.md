# Research references and reproduction register

Reviewed 2026-10-30. [Index](../README.md) · [Analysis](RESEARCH_PLAN.md) ·
[Comparison design](RESEARCH_PLAN.md#5-comparison-and-reproduction-plan)

Sources below are primary papers, proceedings, author project pages, official
repositories, or official documentation. Availability was checked during the
review. **SB3 recurrent PPO recipe adaptations have run, including the separate native 4M campaign; modern author-code MR.Q/MRS.Q, TD-MPC2, DreamerV3 and Valdi have not been reproduced here.** Code availability is not a verified installation, correct adaptation or matched published score. See the [current comparator/novelty gap review](COMPARATOR_REVIEW_2026-10-30.md).
The search is substantial but not proof that the candidate mechanism is novel.

## Closest work: read before claiming novelty

| Reference | What overlaps | Paper / author source | Official code and our status |
|---|---|---|---|
| Chang et al., *The Surprising Difficulty of Search in Model-Based RL*, ICML 2026 | Direct prior work on accurate-model/search failures and value overestimation; raises the novelty/comparator bar | [Publisher paper](https://proceedings.mlr.press/v306/chang26a.html) | [Official MRS.Q code](https://github.com/facebookresearch/MRSQ); paper/selected source review only; no reproduction |
| Rietz et al., *APC-RL*, ICLR 2026 | CarRacing behavior-prior composition; fixed track, symbolic simulator inputs and demonstrations require matching | [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cb22704ebc9bc4e0ef875d1959651012-Abstract-Conference.html) | [Author code](https://github.com/frietz58/apc-rl); protocol review only; no reproduction |
| Pezzato et al., *Sampling-based MPC Leveraging Parallelizable Physics Simulations*, RA-L 2025 | Simulator-driven MPPI is established prior work for our coded-model proposal | [Paper](https://arxiv.org/abs/2307.09105) | Prior-art review; no adaptation executed |
| Ha & Schmidhuber, *World Models*, 2018 | Vision, predictive memory, control; direct CarRacing experiment | [Project/paper and implementation links](https://worldmodels.github.io/) | Historical context; no reproduction selected |
| Freeman, Metz & Ha, *Learning to Predict Without Looking Ahead*, 2019 | Observational dropout and useful world models, including CarRacing | [Author project](https://learningtopredict.github.io/), [paper](https://arxiv.org/abs/1910.13038) | Context; no integration |
| Evans, Jordaan & Engelbrecht, *High-performance Racing on Unmapped Tracks using Local Maps*, 2024 | Local sensing → geometry → racing control | [Paper](https://arxiv.org/abs/2401.17732) | Paper-based mechanism comparison; no exact implementation verified here |
| Trisovic, Carron & Zeilinger, *Uncertainty-Aware Perception-Based Control for Autonomous Racing*, 2025 | Perceived-road curvature uncertainty and multi-scenario MPCC | [Paper, especially Section VII](https://arxiv.org/html/2508.02494v1) | Public implementation not verified; an adaptation must be labeled as such |
| Monninger et al., *MapDiffusion*, IROS 2025 | Multimodal probabilistic vector maps from vision | [Paper](https://arxiv.org/abs/2507.21423) | [Author code](https://github.com/tmonnin/mapdiffusion); related work / mechanism comparator |
| Wang et al., *CogniPlan*, CoRL 2025 | Completed unseen layouts and uncertainty-aware navigation | [Proceedings](https://proceedings.mlr.press/v305/wang25d.html) | [Author code](https://github.com/marmotlab/CogniPlan); related work / mechanism comparator |
| Manchingal et al., *Uncertainty-Aware Autonomous Vehicles: Predicting the Road Ahead*, 2025 preprint / ICRA 2026 | Uncertain future-turn prediction affects speed | [Paper](https://arxiv.org/html/2510.22680v1), [publication record](https://jkooij.github.io/publications/2026_uroadpred/) | Related work; no integration |
| Lindenberg & Chitta, *Valdi: Value Diffusion World Models*, July 2026 preprint | Stochastic latent diffusion MPC directly in CarRacing | [Paper](https://arxiv.org/html/2607.00917v1) | [Author code](https://github.com/Kit115/ValueDiffusionWorldModels); [source audit complete](VALDI_AUDIT.md), reproduction pending |
| Jia et al., *DriveAdapter*, ICCV 2023 | Privileged planning/perception separation and predicted-versus-ground-truth interface mismatch | [Paper](https://arxiv.org/abs/2308.00398) | Related work for intervention validity; not reproduced here |
| Seyam et al., *Model-Aware Generation of Safety-Critical Scenes for Robust End-to-End Driving via Distilled Planner Surrogates*, October 26, 2026 preprint | Explicit planning-versus-perception failure diagnosis and targeted planner improvement | [Version 1, especially IV-B](https://arxiv.org/html/2610.35448v1) | Paper states release is planned; no implementation/reproduction verified here |

Important distinctions from those sources:

- World Models demonstrates dream-only training/transfer in Doom; its CarRacing
  experiment should not be described as that same transfer experiment.
- Trisovic et al. share a control sequence across scenarios and constrain predicted
  positions to the perceived road. Learning truly unseen continuations and
  respecting future revelation is a possible distinction, not established novelty.
- Valdi's supplied wrapper extracts numeric proprioceptive channels from HUD
  pixels, rather than hidden simulator state. Its evaluation script runs a fixed
  600-step loop and ignores termination flags. This is a source finding, not proof
  of the script used for the paper's scores; those scores cannot be copied into
  our terminating 1000-frame table. See the [pinned source audit](VALDI_AUDIT.md).
- MapDiffusion and CogniPlan prevent claiming that multimodal map completion or
  planning over imagined maps is itself new. Their full systems solve different
  tasks; transplanting them is not automatically a fair racing comparison.

## Strong algorithmic comparisons

| Reference | Primary source | Official implementation | Planned role / caveat |
|---|---|---|---|
| Chua et al., PETS, NeurIPS 2018 | [Paper](https://arxiv.org/abs/1805.12114) | [Original code](https://github.com/kchua/handful-of-trials); [MBRL-Lib](https://github.com/facebookresearch/mbrl-lib) alternative | Dynamics-uncertainty diagnostic; original stack is legacy and MBRL-Lib is archived |
| Hansen, Su & Wang, TD-MPC2, ICLR 2024 | [Paper](https://arxiv.org/abs/2310.16828) | [Official code](https://github.com/nicklashansen/tdmpc2) | Strong latent-planning comparator; document history adaptation and episodic configuration |
| Hafner et al., DreamerV3, Nature 2025 | [Published paper](https://www.nature.com/articles/s41586-025-08744-2) | [Author code](https://github.com/danijar/dreamerv3) | Recurrent visual world-model comparator; profile a small configuration first |
| Fujimoto et al., MR.Q, 2025 | [Paper](https://arxiv.org/abs/2501.16142) | [Official code](https://github.com/facebookresearch/MRQ) | Additional modern model-free comparator; CarRacing adapter needed |
| Recurrent PPO implementation | [SB3-Contrib documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html) | [Official repository](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) | First practical memory baseline; match observations, labels, and frame budgets |
| RL Zoo recurrent CarRacing recipe | [Pinned configuration](https://github.com/DLR-RM/rl-baselines3-zoo/blob/61956d90d9f8ac126043ed4024efa12a462d97e1/hyperparams/ppo_lstm.yml) | Upstream commit pinned 2026-10-28 | [Factory/budget audit complete](BASELINE_AUDIT.md); separate native interface/full 4M run completed with version/partition adaptations and late regression; no matched published-score reproduction |
| Wang & Ba, POPLIN, ICLR 2020 | [Paper](https://arxiv.org/abs/1906.08649) | Paper's linked implementation; not installed here | Closely related policy-guided planning; prevents claiming proposer improvement by simulation is itself new |
| Pinneri et al., iCEM, CoRL 2020 / PMLR 2021 | [Proceedings](https://proceedings.mlr.press/v155/pinneri21a.html) | [Pinned official code](https://github.com/martius-lab/iCEM/tree/98c1c1fe2bfc94a87e4658b6c793f4e81bc30203); source audited, not reproduced | [Adaptation audit](SEARCH_OPTIMIZER_AUDIT.md), memory/correlated proposals and matched total-work comparison |
| Sikchi et al., LOOP, CoRL 2021 / PMLR 2022 | [Proceedings](https://proceedings.mlr.press/v164/sikchi22a.html) | [Author project](https://hari-sikchi.github.io/loop/) | Learned-dynamics/terminal-value contrast; not the default coded-simulator architecture |
| Hafner, Yan & Lillicrap, Dreamer 4, 2025 | [Paper](https://arxiv.org/abs/2509.24527) | No reproduction selected | Contemporary video-world-model context; full Minecraft-scale study is outside scope |

TD-MPC2, DreamerV3, and MR.Q are credible competitors, not verified current
CarRacing champions. Retained feedforward PPO/SAC and BC/DAgger provide useful
controls, but beating them alone would not support a strong world-model claim.

## Foundations, evaluation, and historical context

| Reference | Source | How we use it |
|---|---|---|
| Chen et al., *Learning by Cheating*, CoRL 2019 | [Paper](https://arxiv.org/abs/1912.12294), [code](https://github.com/dotchen/LearningByCheating) | Privileged training versus sensor-only deployment; equal-supervision controls |
| Chen, Koltun & Krähenbühl, *Learning to Drive from a World on Rails*, ICCV 2021 | [Publisher paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Learning_To_Drive_From_a_World_on_Rails_ICCV_2021_paper.pdf) | Ego/external-world factorization and distillation; related work, not a CarRacing reproduction |
| Renz et al., *PlanT*, CoRL 2022 / PMLR 2023 | [Proceedings paper](https://proceedings.mlr.press/v205/renz23a/renz23a.pdf) | Learned planning from compact object-level inputs; context for competent privileged-input controls |
| Donti, Amos & Kolter, *Task-based End-to-end Model Learning*, NeurIPS 2017 | [Paper](https://arxiv.org/abs/1703.04529) | Decision-focused prediction is established prior work |
| Bansal et al., TaskMet, NeurIPS 2023 | [Paper](https://arxiv.org/abs/2312.05250), [code](https://github.com/facebookresearch/taskmet) | Task-driven predictive metrics; public code needs an execution audit |
| Sunberg & Kochenderfer, continuous POMDP planning / POMCPOW, ICAPS 2018 | [Paper](https://arxiv.org/abs/1709.06196) | Belief planning and observation-dependent future actions |
| Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*, NeurIPS 2021 | [Project and paper](https://agarwl.github.io/rliable/), [code](https://github.com/google-research/rliable) | IQM, uncertainty intervals, and reproducible comparison |
| Henderson et al., *Deep Reinforcement Learning that Matters*, AAAI 2018 | [Paper](https://arxiv.org/abs/1709.06560) | Variability, reproducibility and standardized experimental reporting; informs pre-run gates |
| Tang, Nguyen & Ha, *Neuroevolution of Self-Interpretable Agents*, GECCO 2020 | [Author project](https://attentionagent.github.io/) | Historical visual CarRacing context, not a current SOTA target |
| Arockiaraj, Chang & Yee, *Transfer Learning for Customized Car Racing Environments*, 2026 preprint | [Paper](https://arxiv.org/html/2605.17928v1) | Recent simulator-specific context, not an authoritative ranking |
| Gymnasium CarRacing | [Official documentation](https://gymnasium.farama.org/environments/box2d/car_racing/) | Base task and standard input/action/reward contract; record installed version |

## Additional reading for search and downstream predictor selection

These sources inform the optional study in [PREDICTION_SELECTION.md](PREDICTION_SELECTION.md).
They do not make decision-focused predictor training the default.

| Reference | Primary source / code | Role |
|---|---|---|
| PlaNet, Hafner et al., ICML 2019 | [Paper](https://proceedings.mlr.press/v97/hafner19a.html) | Pixel-based learned models with online planning; actor-free precedent |
| DINO-WM, Zhou et al. | [Author project/paper/code](https://dino-wm.github.io/) | Reusable predictive visual features with test-time behavior optimization |
| Learning to Act by Predicting the Future, Dosovitskiy & Koltun, ICLR 2017 | [Paper](https://arxiv.org/html/1611.01779v2) | Action-dependent future measurements as an alternative prediction target |
| Objective Mismatch, Lambert et al., L4DC 2020 | [Proceedings](https://proceedings.mlr.press/v120/lambert20a.html) | Prediction scores and downstream control can disagree |
| Value Equivalence, Grimm et al., NeurIPS 2020 | [Paper](https://arxiv.org/abs/2011.03506) | Planning-relevant representations without exact reconstruction |
| Distributional Model Equivalence, Kastner et al., NeurIPS 2023 | [Paper](https://arxiv.org/abs/2307.01708), [code](https://github.com/tylerkastner/distribution-equivalence) | Risk-sensitive models need more than expected-value agreement |
| Calibrated VAML, Voelcker et al., 2025 preprint | [Paper](https://arxiv.org/html/2505.22772v2), [code](https://github.com/adaptive-agents-lab/CVAML) | Bias in sampled value-aware losses; not ordinary probability calibration |
| ARC-Bench, Zhang & Li, 2026 preprint | [Paper](https://arxiv.org/abs/2609.05461) | Candidate-ranking audits separate from closed-loop success |
| Beyond Next-Screen Prediction, Tang et al., 2026 preprint | [Paper](https://arxiv.org/abs/2610.24372) | Direct ranker as an attribution control; findings are from GUI tasks |
| Strictly Proper Scoring Rules, Gneiting & Raftery, 2007 | [Author-hosted paper](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) | Probability scoring and truthful predictive objectives |

Valdi's Section 3 and appendix explicitly omit a policy prior and use CEM;
therefore an actor-free search claim must compare with it, rather than claiming
that actor-free CarRacing world-model control is new. Its original input and
600-frame protocol still require separate reproduction. See the linked paper above.

## Required reproduction record

For every selected competitor, add a record with:

1. Exact paper version, upstream repository commit, license, and environment lock.
2. Original observations, privileged labels, history, reward/termination, action
   space/repeat, training data, and checkpoint-selection protocol.
3. Original-protocol replication outcome, with unresolved deviations disclosed.
4. Our adaptation patch and reasons; no mixing original scores with adapted scores.
5. Training seeds, partition hashes, native-frame/label/model-rollout budgets,
   tuning cost, wall time, inference latency, and artifact paths.
6. Reproduction status: not started, installing, smoke-tested, replicated,
   adapted, or blocked with a specific reason. A smoke test is not a replication.

Refresh the literature and novelty assessment before freezing the paper. Save
formal citation metadata from primary publisher/arXiv records at manuscript time;
do not invent venue details, bibliographic fields, benchmark rankings, or results.

The programmed-physics backend reuses the locked Gymnasium 1.4.0
[vehicle implementation](https://github.com/Farama-Foundation/Gymnasium/blob/v1.4.0/gymnasium/envs/box2d/car_dynamics.py).
Record its installed source hash. This supplies mechanics, not hidden initial
state, road geometry or an authorized clone of the evaluation environment.

The [paper blueprint](PAPER_BLUEPRINT.md#prior-work-and-novelty-assessment) maps
these references to specific novelty/identification claims. The October 2026
preprint's detected-threat analysis is not a causal proof that all relevant
perception/prediction is correct. Our proposed oracle interventions need their
own validation; the general perception-versus-planning claim is already prior art.
