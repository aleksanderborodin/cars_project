# Predictor selection and useful predictions: optional research extension

Reviewed 2026-10-27. [User intent](USER_INTENT.md) · [Specification](EXPERIMENT.md) ·
[Design choices](DESIGN_DECISIONS.md) · [References](REFERENCES.md)

The main system is modular prediction, internal simulation, and online behavior
selection. This note investigates an optional extension: choosing a normally
trained predictor by downstream validation driving performance. It does not make
special decision-focused predictor training the main project or a prerequisite.
No new representation has been shown to work yet.

The user has also proposed a **future** regret-based selection study, with rank
secondary and potentially continuous/adaptive behavior optimization. Its purpose,
attribution and limitations are in [RESEARCH_LOG.md](RESEARCH_LOG.md), R8. It is
not active: the existing privileged ranking monitor is user-only and must not feed
any agent, training, tuning or selection process. Ordinary racing outcomes remain
the current permitted development feedback and system-level paper endpoints.

## 1. The distinction to preserve

The user proposes the following outer loop:

1. Choose a prediction target/space and a predictor configuration.
2. Train the predictor with its ordinary supervised/self-supervised objective.
3. Freeze the predictor. Train/configure the driving controller or planner on top.
4. Measure complete closed-loop driving on validation tracks.
5. Select the predictor/controller configuration from those validation results.
6. Freeze the selected system before final testing.

Predictor weights are fitted on training data. Validation chooses configurations
and checkpoints, including prediction spaces. No driving-loss gradient through
the predictor is required. A borrowed predictor can first be evaluated frozen;
adapting its targets or architecture may require retraining with a normal
prediction loss. That remains distinct from end-to-end decision-focused learning.

The outer objective is useful driving; the inner predictor objective can remain
likelihood, reconstruction, or self-supervised consistency. Report prediction
quality as well as driving quality, and identify whether a gain came from the
predictor, the planner, the controller training, or extra tuning effort.

## 2. What the literature establishes

| Primary reference | Relevant result / scope | Consequence here |
|---|---|---|
| [PETS, Chua et al., 2018](https://arxiv.org/abs/1805.12114) | Probabilistic dynamics and sampling-based planning | Borrowable planning/model baseline; broad architecture is established |
| [PlaNet, Hafner et al., ICML 2019](https://proceedings.mlr.press/v97/hafner19a.html) | Learns image-based latent dynamics and chooses actions by online planning | Learning a model instead of a direct actor is not new |
| [Valdi, Lindenberg & Chitta, 2026 preprint](https://arxiv.org/html/2607.00917v1) | Explicitly has no policy prior; uses CEM and learned reward/value models in CarRacing | Closest actor-free comparator; modified observations/600-frame limit require a separate reproduction |
| [DINO-WM, Zhou et al.](https://dino-wm.github.io/) | Offline visual-feature prediction enables behavior optimization at test time | Reusing a predictive representation for search also has precedent; its goal-reaching task differs from racing |
| [Learning to Act by Predicting the Future, Dosovitskiy & Koltun, ICLR 2017](https://arxiv.org/html/1611.01779v2) | Predicts action-dependent future measurements and selects actions using goals | Compact driving-outcome predictions are a credible alternative to reconstructing a complete road |
| [Objective Mismatch, Lambert et al., L4DC 2020](https://proceedings.mlr.press/v120/lambert20a.html) | One-step predictive likelihood need not track downstream control quality | Validation driving is a justified selection criterion; a better prediction score is insufficient |
| [Value Equivalence, Grimm et al., NeurIPS 2020](https://arxiv.org/abs/2011.03506) | Models can preserve planning-relevant values without reproducing every state detail | Different prediction spaces are legitimate hypotheses; no universal best space follows |
| [Distributional Model Equivalence, Kastner et al., NeurIPS 2023](https://arxiv.org/abs/2307.01708) | Preserving expected values is insufficient for general risk-sensitive planning | If decisions use failure tails, evaluate distributions rather than only means |
| [TaskMet, Bansal et al., NeurIPS 2023](https://arxiv.org/abs/2312.05250) | Learns task-sensitive prediction metrics | Relevant alternative beyond the user's normal-training selection proposal |
| [Calibrated VAML, Voelcker et al., 2025 preprint](https://arxiv.org/html/2505.22772v2) | Certain sampled value-aware surrogate losses have bias for stochastic models; analyzes corrections | Do not assume a special task loss preserves uncertainty; this paper's surrogate calibration is not ordinary probability reliability |
| [ARC-Bench, Zhang & Li, 2026 preprint](https://arxiv.org/abs/2609.05461) | Audits fixed-candidate rankings and reports failures masked by frequent replanning in studied JEPA systems | Measure candidate ranking as well as closed-loop success; not a theorem about our model |
| [Beyond Next-Screen Prediction, Tang et al., 2026 preprint](https://arxiv.org/abs/2610.24372) | In studied GUI tasks, a direct ranker is a strong control for the value of generated futures | Add a direct candidate scorer; do not attribute all gains to simulation |

Official code is linked by [CVAML](https://github.com/adaptive-agents-lab/CVAML),
[distribution equivalence](https://github.com/tylerkastner/distribution-equivalence),
and the existing reference register. Availability has been checked, not execution
or CarRacing compatibility. No external reproduction has been performed here.

## 3. Which prediction might be useful?

Let h be permitted observation/action history and b a candidate behavior: a
control sequence, or a feedback controller with a precisely specified continuation.
One useful target is a distribution q(Y | h, do(b)) over its consequences. The do
notation emphasizes what would happen if we executed that behavior; it does not
imply that a causal model has been identified from arbitrary logged data.

Three representation families are worth comparing after the basic system works:

| Representation | What it predicts | Strength | Limitation |
|---|---|---|---|
| Geometry and vehicle state | Possible road layouts and vehicle rollouts | Inspectable, reusable across planners | May spend capacity on irrelevant distant detail |
| Driving-relevant intermediate variables | Curvature demand, road-clearance margin, braking reserve, traction, time until road becomes visible | Compact, interpretable physical reasoning | Approximate mechanics and hand-chosen variables can omit important effects |
| Behavior-conditioned outcomes | Joint future progress/reward, departure events, speed and terminal condition | Directly supports behavior comparison; may be cheaper | Can extrapolate badly for unfamiliar controls; less reusable for different goals |

My recommendation for the optional first comparison is geometry versus a compact
behavior-conditioned outcome predictor, plus a direct mean-value/risk scorer.
Use ordinary predictive objectives for each. Keep the candidate behaviors,
information, data, history, and tuning budget matched. A hybrid can retain geometry
as a diagnostic without making detailed reconstruction the selection objective.
Confidence is medium that this comparison is informative; which space wins is unknown.

Useful outputs include cumulative official reward at several time horizons;
probability and time of leaving the road; minimum vehicle-to-road-boundary margin;
speed and controllability at the horizon. Lap time alone is insufficient because
failed laps cannot be discarded. A short-horizon gain that ends in an unrecoverable
state is also insufficient. A terminal value or longer continuation needs its
own validation and must use only permitted information.

For the structured-supervision study, true geometry/vehicle quantities can label
training examples. In observation-only native-pixel experiments, use rewards,
allowed observations, and image-derived targets; hidden-state labels change the
supervision category. A borrowed latent coordinate is not automatically a good
control metric merely because it predicts well.

## 4. How prediction connects to search

One frozen model scores many candidates. A planner selects b to minimize expected
loss, optionally plus a declared tail-risk penalty, then executes only the first
short action and replans. It does not pick the most favorable sampled world or
average the optimal steering from different worlds.

There are distinct distributions: belief about the environment, consequences
conditional on a behavior, and the proposal distribution used to sample behaviors.
Optimizing the last one must not make the predicted environment conveniently safer.
A static road does not change because we consider a different throttle action;
the car's path, observations, and consequences do change.

Road/world samples should persist over a rollout. Joint temporal/outcome structure
matters for cumulative risk. Per-behavior outcome marginals can suffice to compare
expected loss or CVaR; a full joint distribution across all behaviors is not always
necessary. Common world samples can improve paired comparisons, but giving two
black-box predictors the same random seed does not prove causal consistency.

When future actions adapt to predicted observations, hidden scenario identity is
never an input. Distinct future actions require distinct observable histories.
If a model only predicts a scalar action value, it is effectively a critic; compare
it with a direct learned critic rather than automatically claiming a new simulator.

## 5. Horizons, targets, and hyperparameters

A fixed 100 m forecast is not a project requirement. Useful horizons depend on
speed, braking, upcoming constraints, model error, and compute. A small initial
comparison can predict at 0.4/0.8/1.6/3.2 seconds and measure decision quality at
matched inference cost. These are development candidates, not frozen settings.
An adaptive horizon is a later alternative. Different horizons require consistent
terminal objectives; shorter predicted sequences cannot simply be scored as
cheaper because they omit later failures.

| Quantity | Meaning | How it is chosen |
|---|---|---|
| Prediction space and target variables | Structural model/design choice | Candidate designs compared on validation driving |
| Network weights | Learned model parameters | Normal training loss on training tracks |
| Road/trajectory probabilities, predicted outcomes | Model outputs for the current history | Inferred by the trained model; not manually tuned to favor an action |
| Width/depth, learning rate, loss weights, checkpoint | Hyperparameters/model-selection choices | Prespecified validation search with costs logged |
| Horizon, candidate count, sample count, optimizer iterations | Planner hyperparameters | Validation performance-versus-compute comparison |
| Sensor footprint and available history | Experimental information contract | Fixed within a comparison, not secretly optimized only for our method |
| Risk tolerance and tail definition | Objective/preference/protocol choices | Declared and frozen; report tradeoffs rather than choose after final results |
| Selected controls | Online decisions | Search under the frozen model and allowed current history |

## 6. Training and validation discipline

For each configuration, train on training tracks and fit the downstream system
using its declared training budget. Use validation return/completion to select
checkpoints and hyperparameters. Prediction metrics remain diagnostics. Keep the
uncertainty calibration split separate, and never use final test racing results
for model selection. Limit and record the number of trials; repeated adaptive
selection can overfit validation even when no gradients use it.

Use a two-stage screen: inexpensive prediction/action-scoring diagnostics discard
broken configurations, then actual closed-loop validation races choose among
competent finalists. Do not discard a lower-likelihood model solely on that metric
if it remains a plausible better controller. Compare equal tuning opportunities
and report the full search cost, including downstream controller retraining.

A clean attribution study has two parts:

- Freeze the same predictor and compare control/search variants, including a
  learned controller on the same representation and an actor-free search variant.
- Freeze the planner, candidate-generation procedure, and cost definition; compare
  predictor configurations selected by prediction quality versus driving quality.

Then report the best complete system separately. These controls prevent gains
from additional search, different candidates, or supervision being mistaken for
better prediction. A learned proposal policy is a labeled hybrid; if the headline
claims no learned actor, use an actor-free candidate generator in that variant.

## 7. Trustworthy distribution and candidate diagnostics

Use proper probability scores when fitting/evaluating predictive distributions:
likelihood where defined, Brier for events, and CRPS or a multivariate energy score
where appropriate. Separate marginal checks do not establish joint temporal
correctness. See [Gneiting & Raftery, 2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
Choosing a representation by validation races does not require inventing a special
loss. If task-weighted losses are tried later, label that as a separate extension;
they may distort probabilities, and ordinary temperature scaling is not a proof
of correcting the issue studied by CVAML.

On permitted development tracks, predict outcomes for a fixed candidate bank
before an isolated evaluator measures their consequences. Ground-truth branch
rollouts are a diagnostic, not a deployment input. Reset/replay must reproduce the
same physical state; Box2D objects cannot simply be assumed safely deepcopyable.
Count replay and branch frames. The policy cannot query the true evaluation
simulator to discover the best action before committing.

One realized hidden road gives one outcome per behavior, not the true conditional
probability distribution given partial observations. Matched-visible-history
families can provide controlled distributional diagnostics, but are separately
labeled synthetic conditions. The best action for the realized hidden world is a
clairvoyant reference, not necessarily an achievable decision from the available
history. Report expected decision quality under the allowed information and
separate irreducible uncertainty from model mistakes.

Training on privileged expert logs also needs care: the expert's action may
encode hidden road knowledge. Observational p(outcome | history, action) need not
match the consequence of imposing that action. Branch/randomized-action data can
help under a declared simulator-assisted training regime. Such reset/query access
is additional information even when raw geometry is not an input; count it and
keep it separate from ordinary online pixel-only comparisons.

Measure ranking/cost-difference error, probability scores, actual racing return,
completion, and latency. Test candidates outside the training proposal's narrow
support. More search may exploit prediction errors rather than improve decisions.
A direct candidate scorer with the same data is a required control for a claim
that explicit imagined trajectories add value.

## 8. Scope and paper language

The user-selected main project remains the complete driving system. Predictor
space/validation selection is an optional experiment. Special decision-focused
predictor losses are not the default. Expert known-track control and agentic
code evolution are separately recorded in USER_INTENT.md; the latter remains
future work and does not restore the archived Rust path.

A credible eventual result might show that one normally trained predictor space
supports better search at matched cost, or that a modular search system is more
sample-efficient than matched learned policies. Both need measured evidence and
comparison with established planning methods. Neither 'try all movements' nor
'no learning is needed' is accurate. A neutral working title is:
**Predict, Simulate, Drive: Control Through World-Model Search**.
