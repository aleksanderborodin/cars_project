# Next moves: establish useful simulation improvement over strong drivers

**Native implementation update:** the [interface/profile](NATIVE_RECIPE_2026-10-29.md) is complete, including the documented cross-device verification correction. Select CUDA/eight subprocess environments by cost. The [next native campaign](NATIVE_BASELINE_2026-10-29.md) declares one seed, 4M aggregate decisions, 0.5/1/2/4M checkpoints and first-50 development selection. It precedes a separate stronger matched-local trial; it does not replace the original local simulator/search study.

Updated 2026-10-29. [Index](../README.md) · [Specification](EXPERIMENT.md) ·
[Architecture](SIMULATOR_SEARCH_PLAN.md) · [Evidence](PROGRAMMED_SIMULATOR_RESULTS.md)

This is the active execution order after the programmed-physics pilot. It refines
the architecture plan's P1–P3; it does not replace the user's coded-simulator
decision. Numerical settings below are assistant proposals for development,
not user commitments or a frozen final study. No new campaign was launched while
writing this plan. Every stage requires its own saved, resolved manifest before runs.

**Execution update:** stage 1 and the trainer/profiling prerequisites in stage 2
are complete; the first 2M-frame block completed with regression after 1M. The
standalone-selected 1M checkpoint now has a completed positive stage-3 development
comparison. Three cheap fixed-correction controls did not reproduce its full gain.
Both KL-stopping and lower-learning-rate probes were worse; baseline repair
remains open. The [200-track comparison](BROAD_RESULTS_2026-10-29.md) is complete.
The [paper blueprint](PAPER_BLUEPRINT.md) consolidates the next-stage three-level
diagnostic study, source attribution and validity requirements. See the
[execution report](BROAD_RESULTS_2026-10-29.md); the original decisions below
remain recorded as the pre-execution plan.

**Latest user steering:** continue the needed experiments autonomously and document
choices/findings. The [geometry intervention](INTERVENTION_RESULTS_2026-10-29.md)
and [20-track reference-tail replication](REFERENCE_REPLICATION_2026-10-29.md)
are complete. Correcting the scoring tail helped its oracle-road condition but did
not beat original local search; the diagnostic is closed without promotion.

The [matched sequence-search comparison](SEQUENCE_SEARCH_2026-10-29.md) is also
complete: colored shooting 884.86/70%, iCEM adaptation 634.13/15%, historical knot
search 905.94/85%. Primary iCEM-minus-colored difference −250.72 [−359.57, −147.31];
the additional eighteen tracks retain the deficit. Five exact replays, all cost
checks and 100 tests passed. Keep the original local search; no larger sweep or
individual-track repair follows this negative result. No campaign jobs remain.

**Next bounded work, assistant decision:** establish baseline competence before
more search tuning. The [factory audit](BASELINE_AUDIT.md) finds that the selected
local actor used 251,904 learner decisions, only 6.30% of the reference recipe's
nominal 4M, with different encoder, action cadence and physical discount. The
[Valdi audit](VALDI_AUDIT.md) also identifies incompatible evaluation and compute
contracts. Neither external method has been reproduced here.

1. Implement and verify a separate native PPO recipe interface: audited pixel
   preprocessing/history, action cadence, normalization and partition-enforced
   streams. Current native RGB-plus-previous-action smoke is not that interface.
2. Profile its actual throughput/memory, then declare its full decision budget,
   versions, checkpoint selection and costs before starting training. Do not infer
   native runtime from state-only local training.
3. Declare a stronger matched local training trial using the audit's budget and
   discount/exploration-cadence findings. A changed recipe is a new trial, not a
   silent resume of the regressed actor. Extra training or cadence repairs helping
   is uncertain; report full curves and failed trials.
4. Once competence is demonstrated, freeze independently trained proposers for
   direct-versus-original-search comparisons. Trained V/L/F rows and modern
   learned-model reproductions remain separately gated; no final test is opened.

High confidence that these controls are needed, unknown which training repair will
succeed. CEM over the original knots, different noise, larger populations and
scenario forecasts remain possible later probes; the current experiment does not
identify which iCEM feature failed. Original staged budget proposals below are
retained as planning history; each new campaign still needs a resolved manifest.

## Objective and present evidence

The core question is whether simulation-based selection improves an already
competent frozen driving policy at a useful inference cost. The behavior proposer
may be learned; the primary world model remains programmed physics and the
selector uses simulated outcomes. We are not proposing to replace the engine
with a larger learned transition model or train a new value scorer by default.

Historical starting evidence for the original plan: on 20 repeatedly used
development tracks, wider
21-candidate search scores 737.3 with 45% completion, versus 716.1/20% for the best
fixed controller. Seven candidates score 697.8/40%, with 48.9 ms p95 latency.
No reliable mean-return advantage is established. A stronger proposer could improve
candidate quality, but it could also expose harmful overrides by an inaccurate
simulator, state estimate, road hypothesis or scoring rule.

Improving a genuinely competitive policy would be stronger evidence than improving
our present fixed controllers. It does not automatically establish novelty or
SOTA. [POPLIN](https://arxiv.org/abs/1906.08649) already combines policies and online
planning. Compare established search with the same simulator/access, and reproduce
any claimed benchmark competitor under its actual task protocol.

## Fixed foundations

- Keep official reward, physics, termination and 1000-native-frame episodes.
  Preserve the current strict-before-frame-1000 completion definition.
- Primary mechanism study: existing 30 m local road/visibility mask, ego sensors,
  previous action and permitted history; action repeat four. Privileged full-map
  diagnostics and native-pixel comparisons remain separate tables.
- Preserve the existing train/development/calibration/test partition mechanism.
  Training uses its recorded fresh-track stream. Use the first 20 development
  manifest entries for debugging, first 50 for initial checkpoint screening, and
  all 200 for broader development. These are nested development subsets, not new
  holdouts or additional independent evidence.
- Keep calibration reserved for a specific uncertainty-calibration procedure.
  Final-test target remains 500 tracks, unopened until the whole study is frozen.
  Keep legacy unseen/hard and user-only oracle outputs sealed.
- Mean official return across all episodes remains primary; completion, failures,
  progress, conditional lap time and latency are co-reported. Do not switch the
  primary endpoint to completion because the pilot looks better there.
- Record all attempted configurations, source/config hashes, training/checkpoint
  and selection costs, native data frames, internal physics frames and wall time.
  Reuse compatible artifacts; do not erase unsuccessful runs.

## 1. One bounded geometry diagnostic, then stop tuning the fixed grid

**Question:** does better road information materially improve this same planner?

Implement an explicit privileged geometry adapter. Keep the coded vehicle engine,
estimated wheel state, score, horizon and candidate definitions unchanged. Supply
full-map geometry without hidden wheel state, solver state or private oracle data.
Document any change in centerline/path construction; otherwise the comparison
would confound representation with geometry access.

Run wider 7 and 21 candidates on the existing 20-track development panel. Reuse
their current local counterparts after replay compatibility checks. New cap:
40,000 native evaluation frames and 11.2 million internal physics frames, plus
separately logged correctness checks. The full-map result is a diagnostic, not an
equal-input competitor or mathematical upper bound.

Do not use a significance cutoff on these 20 adaptive tracks as a universal gate.
Look at paired changes, completions/failures and uncertainty. A substantial geometry
effect motivates road representation/hypotheses; little effect shifts attention
to candidates, utility/horizon and state/contact errors. Either result leads to
the stronger-proposer experiment. No open-ended grid tuning follows this stage.

**Deliverable:** adapter tests, immutable manifest, paired diagnostic report and
one recorded component priority. Confidence: high in the need for a controlled
diagnostic, medium in how decisively this small experiment will identify a bottleneck.

## 2. Build a competent frozen learned driver

**First practical choice:** recurrent PPO based on the pinned
[RL Zoo CarRacing recipe](https://github.com/DLR-RM/rl-baselines3-zoo/blob/61956d90d9f8ac126043ed4024efa12a462d97e1/hyperparams/ppo_lstm.yml),
adapted to our local observations and repeat four. This is an engineering choice,
not a claim that PPO is current CarRacing SOTA. Our existing 100k-frame pilot is
underdeveloped and cannot be the headline comparator. See [BASELINE_AUDIT.md](BASELINE_AUDIT.md).

Before longer training:

1. Replace hard-coded training parameters with a resolved configuration, including
   encoder, recurrent state, action distribution, normalization and decision rate.
   Start from the audited recipe's 512-step rollout, batch 128, ten epochs,
   128-unit LSTM and state-dependent exploration. Keep discount semantics explicit
   under repeat four; this is an adaptation rather than a native-recipe replication.
2. Add reward normalization for learning, saved statistics and frozen evaluation.
   Report official unnormalized reward. Save scheduled checkpoints, optimizer,
   RNG (including CUDA when used), stream state and the global schedule position.
   Never restart normalization or the learning-rate schedule at a milestone.
3. Test recurrent reload, action transforms, normalization, continuation and
   seed/observation isolation. Save learning diagnostics, including failures.
4. Profile 2/4/8 workers and CPU/GPU execution with at least a full update per
   trial, at most 100k native training frames total. Keep profile artifacts separate
   from the final training seed. Hardware observed: Ryzen 5 7500F, six physical
   cores; RTX 5060 Ti, roughly 16 GiB VRAM; about 30 GiB host RAM. Availability
   varies. Do not assume GPU or eight workers are faster; record the chosen setup.

**Budget proposal:** one engineering training seed, a declared 8M-native-frame
maximum and checkpoints at 0.2/0.5/1/2/4/8M. Initially execute through 2M, then
inspect learning health and development performance. The schedule must already
span the declared 8M horizon so continuation does not silently change it.
Budget boundaries can overshoot to a complete rollout; actual counts are logged.

Evaluate scheduled checkpoints on the first 50 development tracks. Up to six
checkpoints cost at most 300k additional native evaluation frames per seed.
Select the standalone checkpoint by mean return, then completion, then earlier
checkpoint. Freeze that same checkpoint for all search variants; do not select
the primary proposer because it happens to benefit most from our search.

Continue toward 8M when the learner is healthy and remains undertrained. If it is
broken or fails to learn, diagnose it and declare a revised trial rather than
spending the entire cap blindly. A run stopped early is not a fair full-budget
competitor. No numerical score alone proves convergence; inspect curves and
compare against our stronger simple references on broader development tracks.

**Deliverable:** a demonstrably competent driver and complete learning curve.
If this recipe fails, repair/reproduce it before switching to a separately
budgeted second actor such as SAC. Do not substitute a famous algorithm name
for measured competence. Confidence: high that we need this control, medium that
this first recipe/budget will suffice.

## 3. Wrap the frozen driver in simulation search

For each real observation, the policy proposes its ordinary action. Generate
nearby smooth behavior variations, simulate their consequences with the coded
engine, execute the selected first action and replan. Keep the unmodified policy
behavior explicitly among the candidates. This provides a baseline candidate,
not a guarantee that a wrong simulator/utility cannot select something worse.

Initial implementation decisions:

- Use feedback policy rollouts, not a long fixed sequence assuming unchanged
  observations. Construct simulated road/visibility observations from each
  hypothesized world with the same sensor footprint and schema as training.
- Clone recurrent memory independently for imagined branches. Advance the real
  policy memory only with actual observations; never commit imagined policy
  history. Estimated physical wheel state remains a separate observer.
- Start with smooth residuals around the frozen policy's actions, parameterized
  by three time knots over a 1.6 s horizon. Clip the official three controls.
  The zero residual is the base behavior; retain all frozen network weights.
  Residual scales are development hyperparameters, not assumed optimal values.
- Begin with 7 and 21 candidate evaluations, one road hypothesis and the existing
  utility. Log both network inference and simulator work. Batch neural inference
  where practical; remeasure latency rather than borrowing the fixed-controller
  timings.
- Use separate planning RNG streams and identical candidate-generation rules
  for matched selection controls. No new teacher/value model is required.

Before racing: test branch independence, reset determinism, candidate zero
equivalence to the standalone policy, action bounds and simulated observation
boundaries. Profile memory and latency without training contention.

First comparison on the 20 development tracks, one frozen checkpoint:

| Condition | Purpose |
|---|---|
| Frozen driver directly | Base performance |
| Random selection from the declared candidate distribution | Variation without outcome-based selection |
| Coded simulation selection, 7 candidates | Initial lower-cost improvement attempt |
| Coded simulation selection, 21 candidates | More search at declared extra compute |

For the first random control use the same seven-candidate family, execute a random
first action and replan, with no unnecessary simulations. Count its lower actual
cost. Repeat this random control with three fixed planning streams rather than
treating one lucky random schedule as representative. Initial comparison cap is
120k native frames: six runs × 20 tracks. Search alone has at most 11.2M internal
physics frames; diagnostics have a separately declared cap. This is developmental
screening, not confirmatory evidence or a general stochastic-variance estimate.

**Deliverable:** paired standalone/search results and actual override frequency,
compute and failure breakdown. Override frequency says how often we intervene,
not whether an individual override was counterfactually correct. Private ranking/
regret remains inaccessible. Confidence: medium in proposal usefulness; unknown
whether our current scoring/rollouts improve an already good driver.

## 4. Improve the component implicated by evidence; compare established search

Do not simultaneously change road inference, behavior proposals, utility and
search and attribute the result to one invention. Predeclare at most six new
configurations for the first refinement round, screen on 20 tracks, then compare
the selected configuration on all 200 development tracks. Report all trials.
This cap limits initial exploration, not a reason to keep a known-broken method.

| Evidence | Next focused intervention | Required control |
|---|---|---|
| Strong full-map benefit; local search struggles | Road memory/reconstruction, then a small shared set of plausible continuations | Single-world planner with same engine/proposer; no hidden-road access |
| Long-rollout state errors in constructed known worlds | Observer/contact fidelity, including polygon contact if needed | Same geometry, actions, horizon and utility |
| Predicted progress disagrees with ordinary closed-loop racing outcomes | Compare declared task utility/horizons; examine recovery and terminal behavior | Same engine and candidate family; retain original score |
| Candidates lack braking/turn/recovery behavior | Smooth maneuvers or policy-distribution proposals | Same selection rule and equal rollout work |
| More candidates help but latency is poor | Warm starts, batching and established adaptive search | Equal total native simulation steps and measured runtime |

For road uncertainty, begin with four shared world hypotheses if justified,
score each behavior by mean simulated utility and preserve each world through
its rollout. Never select the most favorable imagined road. Report both equal-
candidate and equal-compute controls when particle count changes. A probabilistic
road model, if needed, is fitted normally on training tracks; prediction-space or
hyperparameter selection uses development racing as the user proposed. It is a
component option, not the new main contribution.

Add random shooting and an audited [iCEM-style planner](https://proceedings.mlr.press/v155/pinneri21a.html)
with the same programmed engine, observation access, proposal family and utility.
Specify total rollouts across all iterations, including refinement/rescoring;
matching population size alone is not compute matching. Select resolved optimizer
settings on development, label adaptations, and report return-versus-runtime curves.
Established optimizer changes are not claimed as our novel algorithm.

If the paper specifically claims that explicit multi-step simulation is better
than direct candidate scoring, add that scoring control with separately declared
training data and cost. It is a comparison, not the primary selector; it cannot
use the user's private rank/regret outputs. Do not make that stronger attribution
claim from standalone-policy versus search alone.

Maintain two performance views: unrestricted offline score and a declared latency
budget. An 80 ms target is useful, but p95 alone is not a hard real-time guarantee.
If claiming deadline-respecting control, define fallback-to-base behavior before
running that condition and count missed deadlines and fallback outcomes.

## 5. Establish that improvement survives stronger baselines and broader tracks

Move to all 200 development tracks before final claims. Train three independent
policy seeds for exploratory replication, using the same selected recipe and
budget; do not assume repeating a deterministic planner creates independent models.
The initial one-seed run may be reused if its protocol/configuration matches.
At 8M per seed this is at most 24M cumulative training frames, not 24M in addition
to already matching runs. Predeclare any hyperparameter/training-budget revisions.

Compare direct policy, selected policy-plus-search, a larger search budget and an
established planner with the same simulator. Four conditions × three policy
checkpoints × 200 tracks cap evaluation at 2.4M native frames. Reproduce random
selection and component controls at prespecified stochastic repeat counts in a
separately costed manifest. Do not silently omit the losing/broken trials.

Measure improvement across fixed earlier training checkpoints as a secondary
analysis: does search help only weak drivers, or also the best one? Select
milestones before evaluating search benefits; do not retrospectively rename a
favorable checkpoint "strong". Reuse saved weights rather than retraining them.

A claim of general improvement across policy families needs a second competent
actor family, initially SAC as a practical candidate, with a separate profiled
training manifest. It must learn successfully under the same input contract.
If only recurrent PPO is studied, limit the claim to that tested family rather
than implying that improving several checkpoints proves universal applicability.
The three-seed budget above covers the first family only.

For the same frozen checkpoint with/without search, estimate paired effects by
resampling checkpoint identities jointly and common tracks jointly. Extend our
reporter for that genuine pairing; the existing independent-model bootstrap is
not the correct paired-checkpoint estimator. For separately trained algorithm
families, preserve independent model draws with paired tracks. Stochastic planner
repeats remain nested within checkpoint/track, not independent trained agents.

**Proceed criterion:** reproducible benefit over a competent proposer at a useful
cost, and an advantage or substantive finding beyond established planning. Inspect
effect sizes and intervals, not just a development p-value. If gains occur only
for weak policies or vanish against iCEM, narrow the claim or redesign using
development evidence; do not declare the method generally superior.

## 6. External comparability and native-input work

Start the input/reproduction audit during baseline preparation; do not postpone
discovering an incompatible interface until after all local tuning. Scale visual
training after the local mechanism is informative.

- First reproduce the pinned RL Zoo pixel recipe as its own condition. Audit
  preprocessing, recurrent state, action frequency and actual native-frame budget.
  A local adaptation is not that reproduction; a published checkpoint with unknown
  training-track overlap cannot silently become a clean held-out paper model.
- Audit and attempt the direct CarRacing actor-free comparison
  [Valdi](https://arxiv.org/html/2607.00917v1). Its proprioception and 600-frame
  episodes require a separate reproduction; adapt to our task only with explicit
  changes and new measurements. Do not import its reported score into our table.
- Select one feasible modern whole-system comparator from the audited roster,
  initially [TD-MPC2](https://arxiv.org/abs/2310.16828); DreamerV3 is the visual
  alternative. Profile before committing large runs. No current CarRacing champion
  has been verified here. Freeze the final roster after reproduction feasibility
  checks, recording failures and reasons rather than hiding strong competitors.
- Compare original whole methods. TD-MPC2 already includes planning; its policy
  prior alone is an ablation, not the full competitor.
- Native-input search requires road/state inference from pixels/history and
  simulated policy observations compatible with that input. Our current geometry
  planner cannot be attached to a pixel policy while secretly receiving ideal maps
  or numeric state. Audit rendering/observation reconstruction and label costs.
- Declare known-physics access as an additional assumption. Match it for planning
  controls and report it explicitly against learned-only world models. Match
  inference budgets where meaningful and show cost curves where exact matching
  would conceal a real tradeoff.

A rigorous local-protocol study can support a narrower scientific contribution;
standard CarRacing SOTA additionally requires competitive results under genuinely
comparable native-input/task conventions. Success on a strong base policy alone
does not establish either novelty or the strongest standard-benchmark score.

## 7. Final freeze and paper

Only after development: freeze all input regimes, selected checkpoints, training
and planner seeds, methods, ablations, analysis, primary contrasts, multiplicity
handling and failure/deadline rules. Target five independent training seeds for
headline learned methods and 500 reserved tracks per checkpoint/condition. Reuse
matching development-trained checkpoints; increase the sample size only through
a pre-test rule based on development precision, not unseen results.

For illustration, four conditions around five paired policy checkpoints on 500
tracks means 10,000 episodes, at most 10M native evaluation frames. This excludes
additional independent comparators and stochastic repeats. Final internal-simulation
and wall-time budgets must be computed from the selected settings before launch.
Do not promise a calendar duration from environment-frame counts alone.

Write methods and related work while developing. Required evidence includes:
learning curves; same-checkpoint search improvement; performance versus base-policy
strength; return/completion versus compute; component controls; failures and held-
out uncertainty. No result sentences are written as facts before measurements.

If the final test changes our design, treat that as a new study requiring untouched
data. If only completion improves or latency is excessive, report that narrower
tradeoff. A weak result is evidence to understand, not permission to remove baselines.

## Immediate work queue and deferred choices

1. Broaden the original frozen-checkpoint comparison to all 200 development tracks
   before further planner tuning. Completed four conditions: direct, search 7,
   search 21, and the already fitted mean-7 constant correction. Keep weights,
   utility, residual scales, horizon, planning seed and offset fixed. Use the
   explicitly recorded sparse renderer for search, with its parity evidence.
   Cap: 800k native evaluation frames and 112M internal physics frames. This
   [manifest](../configs/campaigns/broad-driver-2026-10-28.json) was frozen and completed. The original 20 and the 50-track
   selection panels are reported as nested development subsets, not independent holdouts.
2. Implement the [audited iCEM-style adaptation](SEARCH_OPTIMIZER_AUDIT.md) using the same coded engine, information,
   proposer, horizon and utility; match total simulator work across iterations.
   Profile cost before a capped comparison. No novel-optimizer claim follows
   from adopting it. Preserve the current random-shooting result as reference.
3. Audit baseline training/reproduction after two unsuccessful single-factor probes.
   The lower-learning-rate block completed; best 692.18/8% did not improve the
   original 858.11/30% on the same 50-track selection panel. Keep the original
   checkpoint; do not automatically extend either failed repair or launch another
   optimizer sweep. Review implementation, competence and independent replications.
   [Full results and costs](BROAD_RESULTS_2026-10-29.md) retain both failures.
4. After broader evidence, run independent training seeds and prespecified
   planning-stream repeats; freeze each proposer by standalone performance.
   Continue the native-input reproduction audit before claiming comparability
   with published methods. Final-test data remain sealed.

Do not retrain a learned dynamics network by default. Do not launch all sensor
ranges, horizons, predictors and optimizers as a Cartesian sweep. Regret-loss
training, Bayesian allocation, Rust optimization and agentic evolution remain
documented future options. Rust becomes relevant if profiling identifies Python
simulation overhead as the obstacle; it must preserve physics and replay behavior.
The user's further [information/control study](INFORMATION_AND_CONTROL_STUDY.md)
adds future local/full-map learned/search contrasts and a three-module diagnosis.
It follows competent local controls; it does not open global training or private
oracle data now, and it must not replace the established-planner comparison.

The high-confidence decisions are frozen-checkpoint comparisons, information
separation, accounting and test isolation. PPO suitability, 8M sufficiency, residual
parameterization, useful horizon/utility and the eventual winning method remain
empirical uncertainties. A SOTA result or paper acceptance cannot be promised.

The user's [task-useful simulator option](DECISION_USEFUL_SIMULATOR.md) is deferred:
our own approximate/abstract engine, possible GPU execution, and optional code or
parameter adaptation by downstream driving. Realism is not required. The main
understanding-versus-decision question and frozen-proposer comparison remain the
priority; the completed campaigns remain fixed and the established-optimizer comparison remains planned.
A fidelity-preserving implementation optimization and a deliberately task-adapted
model are distinct experiments with different numerical acceptance criteria.
