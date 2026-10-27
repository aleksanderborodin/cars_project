# Proposed design decisions and confidence

Updated 2026-10-27. [Index](../README.md) · [Specification](EXPERIMENT.md) ·
[Research assessment](RESEARCH_PLAN.md) · [Status](STATUS.md)

This is the concrete starting configuration proposed after the research review.
It is not implemented, benchmarked, or the final frozen study. Numerical defaults
below may change using development evidence only; log the reason and freeze them
before final evaluation. The experiment specification remains authoritative.
User decisions and optional ideas are recorded in [USER_INTENT.md](USER_INTENT.md).
These numerical choices are assistant proposals, not user commitments. The main
project is the complete prediction/simulation/control system; the geometry model
below is one candidate implementation. Predictor-space/model selection after
normal training is optional. Agentic evolution remains future work.

Confidence describes whether a choice is a sensible research decision, not the
probability that our method wins. **High:** strong methodological justification.
**Medium:** reasonable starting choice with a real alternative. **Low:** central
empirical or novelty uncertainty that needs evidence. No acceptance/SOTA probability
is estimated from the available evidence.

## 1. Scope and claims

Use Gymnasium CarRacing-v3 only, with official standard reward, termination,
physics, continuous three-component controls, and the current workstation.
Run a controlled partial-observation study and an early native-pixel pilot.
The native-pixel experiment is required for the desired standard-benchmark claim;
a result confined to the local protocols supports a narrower claim.

Primary question: can a modular predictive model support competitive driving
through internal simulation and online search, at a practical data/compute cost?
Geometry uncertainty and future-observation-aware planning are component studies.
The user additionally proposes normal predictor training followed by downstream
validation selection, not special end-to-end driving-loss training by default.
No component or combination is assumed novel. Prior art includes
[Trisovic et al.](https://arxiv.org/html/2508.02494v1) and
[Valdi](https://arxiv.org/html/2607.00917v1).

Confidence: **high** in the controlled-study design; **medium** in a useful compact
system; **low** in a new algorithmic contribution or a native-benchmark advantage.

## 2. Data and randomization

| Choice | Proposed default | Confidence | Alternative / trigger |
|---|---|---|---|
| Training tracks | Reproducible fresh-track stream drawn only from a training partition | High | Fixed finite corpus for a separate offline/sample-efficiency study |
| Validation | 200 fixed, disjoint standard-generator tracks | High in separation; medium in count | Increase before freezing if development estimates are too noisy |
| Calibration | 200 additional fixed tracks for a prespecified uncertainty calibration procedure | Medium | Fit uncertainty within training folds if the extra split proves wasteful; never use final test data |
| Final standard test | Target 500 fixed tracks per model and main condition | High in pairing; medium in count | More tracks or training seeds if pilot variance cannot resolve the target effect |
| Legacy held-out sets | Keep existing unseen/hard sets sealed and separate | High | Use only in the frozen final campaign; do not inspect them to select designs |
| Training replications | One seed for engineering; three for exploratory comparisons; five for headline learners | High in independent replication; medium in sufficiency | Increase model seeds when between-training variability dominates |

These are requested capacities for a new manifest system, not existing lists.
Reservation and overlap checking must respect the prohibition on inspecting
held-out identities/results during design. Audit historical training-side records
before reserving new partitions. Whole tracks and related counterfactual variants
stay together. Use the same final tracks for every method, with memory reset on
every episode. Training randomness is separate from track randomness.

Use the same partitions across observation protocols where appropriate, but once
final results from any protocol are inspected, do not use them to redesign another
protocol on the same test tracks. Freeze the whole planned campaign first.

## 3. Information and control interfaces

| Protocol | Agent input at train/validation/test | Role |
|---|---|---|
| G: global | Full geometry and the declared vehicle sensors | Privileged reference, not a mathematical upper bound |
| L: local geometry | Boundary polylines clipped to the visible footprint, visibility mask, forward/lateral body velocity, yaw rate, previous action, observation history | Primary mechanism study |
| V: local vision | 96x96 local RGB image and mask covering the same footprint, same vehicle sensors/history | Effect of perception error |
| S: native pixels | Official 96x96 image stream, previous actions, permitted history; no extra numeric vehicle state or map | Standard-benchmark experiment |

For L/V, start with a circular top-down footprint of radius R around the car,
with the car orientation fixed in the observation coordinates. No occlusion model
is implied. Pilot radii are **15, 30, 60, 120 simulator metres**, with **30 m** as the
initial primary condition. Train one range-conditioned model uniformly across
these four radii and report each range separately. The sensor configuration is
known to every agent. Before the comparison campaign, check the radii against
measured braking distances; revise the grid only to cover informative regimes,
with that decision recorded, not after looking for a favorable winning condition.

Clipping must remove hidden connectivity, vertex IDs, global progress, seed,
track ordering outside the footprint, and reset metadata. Previously observed
geometry may be remembered. V has a fixed scale from frame zero; S preserves
the official image stream, including initial zoom and HUD. All policies have
access to the same history within a table. Actor inputs initially exclude reward
history; rewards remain available to training losses. Any baseline needing a
different input contract belongs in a declared adaptation or separate table.

Use the official three control channels. Initial matched decision interface:
**four physics frames per action**, hence a nominal 80 ms interval at 50 Hz.
Count native frames, not just decisions. Audit competitor action repeats; retain
original-protocol reproductions separately from matched-repeat adaptations.
Native input alone does not make scores from differing episode limits or budgets
comparable. See the [official environment specification](https://gymnasium.farama.org/environments/box2d/car_racing/).

Confidence: **high** in matched information/history and protocol separation;
**medium** in the footprint, range grid, and action repeat. Alternatives are an
explicit forward field of view, denser observations, or one-frame actions. Select
these based on information and control diagnostics before final comparisons.

## 4. Candidate geometry-based implementation

Build the basic system before studying alternative prediction spaces. The road
model and its forecast distance in this table are one engineering option, not
requirements for every variant. Existing world-model implementations may be
reused. See [PREDICTION_SELECTION.md](PREDICTION_SELECTION.md) for optional model
selection; do not make that investigation block the working control loop.

| Module | Initial implementation | Confidence | Alternative and evidence needed |
|---|---|---|---|
| Road representation | Curvature at 2 m arc-length intervals, predicting 100 m beyond the observed frontier; integrate to boundaries with known constant width | Medium | Splines or occupancy representation if loops/multiple visible segments make this representation fail |
| History | Recurrent state from permitted observations/actions; reset each episode | High | Explicit observed-map memory if recurrence loses useful already-seen road |
| Simple road prior | Conditional empirical/Markov curvature model fitted on training geometry | High as a necessary baseline | Smooth straight/constant-curvature extrapolation as still simpler controls |
| Learned forecaster | Two-layer GRU, width 128, five-component Gaussian mixture for sequential curvature increments | Medium in small autoregression; low in exact capacity | Small Transformer if underfitting; diffusion only if missing modes materially harm decisions at matched compute |
| Forecast training | Conditional likelihood, train-only normalization, checkpoint selected on validation | High as a transparent starting objective | Decision-weighted loss only as a separately controlled experiment; probability quality may worsen |
| Calibration | Fit one temperature for mixture logits on calibration tracks using likelihood; separately assess braking-critical event reliability and coverage | Medium-low | Variance calibration or fixed-bin event calibration if logits alone do not address observed errors; no guarantee of calibrated event risk |
| Dynamics | Fitted planar vehicle model with road-contact dependence; validate multi-step rollouts and candidate-action ranking | Medium-low | Three small residual MLPs, each two layers of width 128, if model error changes control choices; train from permitted histories |
| Visual module | Color/edge segmentation and visible spline fitting, then a small U-Net with widths 32/64/128 if necessary | Medium | Learned keypoints/vector decoder if raster reconstruction dominates error or latency |
| Initial planner | Warm-started CEM: 128 candidate sequences, 16 elites, three iterations, 20 decision steps, 16 persistent road samples | Medium in sampled MPC; low in exact runtime/candidate adequacy | MPPI or fewer samples if latency fails; longer horizon/terminal value if horizon limits competent control |
| Planning objective | Progress, road-departure and control-smoothness costs; expected loss plus CVaR over worst 25% of sampled roads | Medium-low | Expected loss alone; risk weights 0, 0.25, 1 as initial validation candidates, with cost normalization fixed first |
| Deployment target | Measure whole-pipeline latency; target p95 below 80 ms on the workstation | Low until profiled | Smaller planner, distilled policy, or a separately labeled slower decision protocol applied to all comparators |

The initial planner uses about 123,000 candidate-map decision transitions per
control update (128 x 3 x 20 x 16), before residual ensembles or other overhead.
No measured throughput is claimed. Profile before committing to the full study.
The horizon is initially 1.6 seconds; compare 0.8/1.6/3.2 seconds on development
data. Geometry horizon and dynamics horizon are different quantities.

For matched geometry ablations, share dynamics, solver, cost, horizon, training
corpus, and tuning effort. Each imagined road remains fixed throughout a rollout.
The planner never copies the live environment or gets its hidden geometry.
Its internal objective does not change the simulator's official reward.

## 5. Additional planning experiment after basic scenario MPC

First implement valid shared-open-loop scenario MPC. Then test a small shared
feedback controller inside rollouts: CEM optimizes its speed/control parameters,
and future actions can respond to synthetic local observations. Hidden sample
identity and unseen road are not inputs to that controller. Identical observation
histories must produce identical actions. This tests whether anticipating future
visibility improves the conservatism/performance tradeoff.

Compare open-loop and feedback rollouts at matched model-transition budgets.
Evaluate calibration both geometrically and on events relevant to braking.
This is an **optional mechanism study**, not a validated novel algorithm. If it merely
reimplements an established method, the contribution must be an independently
valuable result/analysis, or we must revise the method. A collection of standard
modules plus a small score increase is insufficient evidence for a strong claim.

Alternative: if online planning adds little or exceeds latency limits, train a
policy on the same frozen belief and compare it with MPC. A later frozen-world
imagination/distillation experiment may use the same collected corpus; count all
imagined transitions and teacher labels. Do not expand into a separate large
world-model project before the main question survives.

## 6. Supervision and the native-pixel bridge

Start L/V in a **structured-supervision track**: geometry labels from training
tracks are allowed. Initial shared collection budget is **200,000 native frames**
per training seed, with competent driving, perturbed controls, and recovery states.
Count all collection frames and labels; this allocation is included in a learner's
total budget. Train-only map generation without stepping physics is also counted
separately as extra supervised data.

Provide equivalent label/data access to comparison controls, including a recurrent
policy with geometry auxiliary supervision and a policy consuming the same frozen
representation. Baselines without labels remain explicitly observation-only;
they cannot by themselves isolate the benefit of our algorithm from supervision.

For S, the cleanest external comparison is **observation-only training and pixel-only
deployment**. Geometry must then come from images and image-derived targets, not
hidden simulator labels; vehicle motion/scale must also be inferred. Known rendering
or width priors are disclosed. This is a substantial adaptation, with **low-to-medium
confidence**, and not merely replacing the input layer.

In observation-only S, calibration must also use permitted observations or
image-derived targets, not simulator geometry labels. Select checkpoints on
validation task return under the same information rules as competitors. Hidden
truth may support separately reported evaluator diagnostics, but cannot fit the
calibrator or select the observation-only model. The geometry-likelihood
calibration default above applies to the structured-supervision L/V study.

Run its feasibility pilot early. If supervised geometry is necessary, report a
native-input, privileged-supervision track with equal-label controls. Do not present
it as an observation-only SOTA result. If gains appear only under restricted vision,
retain the limited-visibility claim and disclose the native result.

## 7. Comparator roster and ablations

| Experiment | Required comparisons | Additional comparison if warranted |
|---|---|---|
| Local geometry | Local pure pursuit/braking rule; deterministic MPC; simple-prior scenario MPC; learned-mean and learned-samples MPC; recurrent PPO with matched supervision; TD-MPC2 history adaptation | SAC/history and MR.Q if the model-free comparison remains weak |
| Local vision | Same simple/planning controls through matched perception; recurrent policy; DreamerV3; policy on our frozen representation | Trisovic-style uncertainty around visible curvature |
| Native pixels | Recurrent model-free baseline, DreamerV3, TD-MPC2 adaptation, Valdi and its deterministic comparator where reproduction succeeds | Historical methods for context, not an inherited leaderboard |

Pin upstream code and record faithful replication versus adaptation. Reproduction
failures must be investigated and disclosed, not used to silently discard a strong
competitor. None of the named methods is assumed the verified CarRacing champion.

Essential module swaps: exact/estimated visible geometry; simple/learned mean/learned
sampled future; uncalibrated/calibrated predictions; open-loop/observation-feedback
planning; fixed/learned dynamics residual; planner/policy on identical representations.
Run focused comparisons, not the full Cartesian product of every setting.

## 8. Budgets, endpoints, and gates

- Engineering runs: up to **100,000 native frames** for one seed and runtime checks.
- Development comparison checkpoint: **2M native frames**, including the 200k
  collection allocation where used; report earlier learning curves.
- Extend the shared comparison budget to **5M** when development learning curves
  show the 2M comparison would assess unfinished training. Larger needs trigger a
  revised resource plan, not a claim that the undertrained baseline is inferior.
- Report environment frames, labels, model transitions, accelerator/CPU hours,
  tuning trials, memory, and p50/p95/p99 latency. Frame matching is not compute
  matching; add performance-versus-cost comparisons.
- Primary task endpoint: mean official return on all episodes. Completion is a
  co-reported guardrail; completed-lap time is secondary with its denominator.
- Provisional practical target: **+10 return points** over the strongest predefined
  matched competitor, with completion no more than **2 percentage points lower**.
  Validate the meaningfulness and achievable precision of this target on development
  data before freezing; it is not a result or universally valid effect threshold.
- Use paired test tracks, five independently trained models, and intervals that
  respect both track and training-run variability. State the IQM estimand. The
  [rliable work](https://agarwl.github.io/rliable/) informs reporting, but does not
  make five seeds automatically sufficient.
- No real-time or rare-safety claim follows from offline return or a small number
  of failures. Report actual deadline misses and failure categories.

| Stage | Concrete output | Proceed only if |
|---|---|---|
| A | Local interfaces, partition manifests, provenance, leakage and equivalence tests | Information boundaries and reproducibility hold |
| B | Working internal model and online search; local/global expert diagnostics; native-input pilot | Competent control, valid observations, and measured runtime; geometry gaps guide only that component |
| C | Complete-system comparison with fixed-predictor controller/search controls | Gains are attributable and survive appropriate learned-policy/planning baselines |
| D | Optional predictor-space/model selection and focused planning/calibration ablations | Any added complexity improves validation driving at matched resources |
| E | Visual/native adapters and strong competitor runs | Claims survive appropriate competitors; failures are investigated |
| F | Frozen models, endpoints and all conditions; final evaluation and release | Test results have not influenced design; uncertainty supports the claimed effect |

Do not launch all final training runs before B/C. Time estimates remain conditional:
the first A/B sprint is roughly one to two engineering weeks; total publication
time cannot be credibly fixed before profiling and the basic control pilot.

## 9. What would change the plan

No hidden-information gap means reconsider the hidden-road component, rather
than manufacture a favorable benchmark or stop the whole system automatically.
A simple prior matching learning means keep the simple model and abandon that
learned-forecast advantage claim. Poor dynamics ranking means fix the internal
model before attributing failures to its geometry representation.
Native-only failure narrows the claim. Strong baselines winning changes the result;
it is not a reason to remove them. Weak novelty changes the method or the paper's
contribution, not the description of existing work.

## 10. Alternatives for the whole system

| System | Why choose it | Main limitation | Decision trigger |
|---|---|---|---|
| Recommended: modular predictive model plus online behavior simulation/search | Matches the main user goal; allows existing models and controlled component swaps | Integration, model exploitation, runtime, and unproven novelty | Establish working control, then compare complete systems under matched resources |
| Local geometry plus classical robust MPC and a simple road prior | Lowest model/training complexity; strong, understandable controller | A standard implementation is unlikely to provide an algorithmic paper contribution | Prefer it if learning adds no reliable advantage; retain it as a mandatory baseline regardless |
| Latent world model plus planning, following TD-MPC2/Valdi | Works closer to native pixels; avoids explicit metric map reconstruction | Harder to diagnose geometric uncertainty; established method family | Prefer this direction if explicit geometry is the bottleneck and a distinct latent-model mechanism emerges |
| Recurrent policy, optionally trained in a learned world or distilled from the planner | Cheap deployment; retains memory; closest to the train-in-an-internal-world variant | Model exploitation during imagination; representation/control gains can be confounded | Prefer deployment through a policy if MPC exceeds latency limits or adds little beyond the same belief |
| Controlled scientific/benchmark study rather than a new winning algorithm | Can explain when imagination helps or fails with matched information and counterfactuals | Needs an independently substantive finding; a negative result alone does not ensure publication | Consider if algorithm gains fail but the mechanistic evidence is strong and general enough to justify a paper |

Confidence in a classical or recurrent prototype working is higher than confidence
that the proposed structured method beats them. None of these alternatives earns
a novelty claim merely by being implemented in CarRacing. The recommendation is
to use the simpler alternatives as serious competitors and let the pilot decide
which research claim remains credible.


## 11. Additional user ideas, outside the current critical path

- **Expert/known-track experiment:** retain a competent privileged controller or
  full-map planner. Compare information conditions separately and label any
  demonstration supervision. Useful for locating perception/model/control limits;
  not a guaranteed optimal reference.
- **Predictor selection:** train normally; freeze each predictor; train/configure
  the downstream driving system; choose prediction spaces/hyperparameters using
  validation racing. No special driving-loss gradient is required.
- **Agentic evolution:** later explore code/controller/planner optimization,
  potentially in Rust, while keeping the predictor and information protocol fixed.
  Account for adaptive validation use and search cost. Do not restore the archived
  tooling or run evolution as part of the present implementation milestone.

These distinctions follow [USER_INTENT.md](USER_INTENT.md). Assistant critiques
and confidence in each idea are recorded there and in the optional research note.
