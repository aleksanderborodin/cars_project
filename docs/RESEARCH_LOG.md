# Research questions, ideas, intent and probes

Updated 2026-10-29. [Index](../README.md) · [User intent](USER_INTENT.md) ·
[Implementation/evidence](STATUS.md)

This log preserves who proposed an idea, why, and what would test it. User wording
is paraphrased, not quoted verbatim. Questions are not treated as decisions.
Proposals, implementation choices and measured evidence have separate statuses.
Earlier project intentions remain in USER_INTENT.md; this log extends that record.

## R1 — Does the driving agent train with full information?

- **Author/type:** user question, 2026-10-28.
- **Context/intent:** understand what information the action-selecting agent gets,
  rather than conflate privileged simulator access with deployed controller inputs.
- **Current answer:** the main pilot uses a 30 m visible road mask, body velocity,
  yaw rate and previous action. Recurrent PPO can remember permitted observations.
  The search controller has a fitted dynamics predictor and no learned driving
  actor. The legacy full-map/raw-state expert is a separate privileged reference.
- **Assistant proposal:** compare the same planner under local and full-map
  information to distinguish missing-information failures from prediction/control
  failures. This is more diagnostic than comparing unrelated controllers.
- **Status:** local pilot and legacy full-map expert measured; same-planner full-map
  variant not implemented. The question did not authorize changing the main
  agents to privileged inputs.

## R2 — How does the system choose its final movement?

- **Author/type:** user question, 2026-10-28.
- **Context/intent:** distinguish sampling possible roads from searching possible
  behaviors, and understand whether the search is exhaustive or adaptive.
- **Current implementation:** 7 target speeds × 3 steering gains define 21 feedback
  behaviors. A single fitted ego-dynamics model simulates them for 20 decisions
  (1.6 s); the best predicted-cost behavior supplies the first action, held for
  four native frames (0.08 s), followed by observation and replanning.
- **Road assumption:** visible geometry plus a constant terminal-tangent extension.
  This is one assumed continuation, not a learned distribution over roads.
- **Status:** fixed finite grid implemented. Hundreds of road samples, CEM and
  Bayesian behavior search are not implemented. The grid is an engineering
  baseline, not a final paper algorithm or a novelty claim.

## R3 — Bayesian/adaptive search over worlds and behaviors

- **Author/type:** user future idea, 2026-10-28.
- **Context/intent:** use a distribution over the road ahead together with a
  distribution over promising behaviors/agent designs, to allocate computation
  more intelligently than the current fixed candidate grid.
- **Scope still open:** a distribution over action sequences, controller parameters
  or more general controller designs. Do not silently reduce the user's idea to
  one particular optimizer or promote it to a required component now.
- **Assistant interpretation:** distinguish environmental belief `p(world | history)`
  from a search/proposal distribution over behaviors. The latter is not a road
  probability or automatically a calibrated posterior. Bayesian optimization,
  adaptive sampling and CEM are different possible tools; CEM alone is not a
  claim of Bayesian inference.
- **Potential probe:** fixed versus adaptive candidate allocation, with the same
  world predictor, candidate family, observations and simulation budget. Separately
  test one-world versus multi-world evaluation and expected versus tail-sensitive
  objectives. Future scenario actions must not receive hidden sample identity.
- **Status:** future extension, not implemented or measured. It must fit the main
  modular model/simulation/search system and justify its additional compute.

## R4 — Rank the chosen candidate using privileged outcomes

- **Author/type:** user metric proposal, 2026-10-28.
- **Context/intent:** evaluate prediction as a tool for choosing behavior: would
  the selected trace still be preferred, and what would its rank be, if the true
  road/outcome were known? Geometry/prediction RMSE alone does not answer this.
- **Assistant implementation choice:** freeze all 21 candidate action sequences,
  their predicted costs and the chosen candidate. A separate worker resets the
  official development simulator and exactly replays the observed action prefix,
  then executes each frozen sequence. No live evaluation-world clone is passed
  to the driver. Count prefix replay and candidate rollout costs separately.
- **Metrics:** selected candidate's best/worst/midrank under ties, top-1 membership
  including ties, top-3 by best rank, hindsight regret, normalized regret, uniform
  candidate utility and Spearman agreement between predicted score and actual
  short-horizon return. Show all-tied states explicitly.
- **Why fixed candidates:** generating new oracle actions would compare different
  candidate sets and fail to isolate the quality of the original choice.
- **Interpretation limits:** the audit executes open-loop sequences generated by
  the nominal feedback rollouts; actual deployment replans every decision. Official
  short-horizon return also differs from the planner's handcrafted objective, so
  this measures overall choice quality, not prediction error alone. The hindsight
  oracle has information unavailable to the agent; nonzero regret need not imply
  a defective uncertainty model. States/candidates are not independent samples.
- **Status:** implemented as an optional online sidecar. Real ranking evidence is
  subject to the user's restriction below and is not read by the assistant.

## R5 — User-only oracle monitor, with no learning feedback

- **Author/type:** explicit user restriction, 2026-10-28; supersedes any earlier
  assistant suggestion of using the oracle diagnostic for method development.
- **Intent:** the user may watch ranking quality online, while no driving agent,
  research agent or learning/selection process benefits from privileged outcomes.
- **Implemented separation:** the driver exports frozen requests through a one-way
  queue. A separate simulator worker writes only to ignored `private_diagnostics/`.
  The main process does not read results or accept replacement actions. A private
  HTML page refreshes while monitoring runs.
- **Permitted assistant checks:** source inspection, synthetic metric tests, public
  technical completion/resource counts, and equality of ordinary driving actions
  with monitoring enabled/disabled. No real rank, oracle-return, regret or ranking
  correlation values may be read, summarized, indexed or used for tuning.
- **Documentation rule:** public research reports may describe the metric and
  monitoring mechanism; they must not contain the protected values. Any future
  change to their use needs an explicit change to this user restriction.
- **Boundary:** this is an enforced workflow/data-flow convention, not an OS sandbox
  against malicious code. No AI subagent has been given these outputs.

## R6 — Search may exploit dynamics outside the collection distribution

- **Author/type:** assistant problem diagnosis and probe, 2026-10-28.
- **Evidence/context:** ordinary public driving logs showed the initial search
  controller using 70–90 speed targets while the conservative training corpus had
  approximately 99% of forward velocities below 50. On 20 development tracks its
  mean return was lower than the local references. This observation uses permitted
  ego/action logs and driving outcomes, not the user-only oracle monitor.
- **Purpose:** avoid mistaking low one-step prediction error for a model that can
  rank aggressive controls reliably. Do not conclude that missing-road prediction
  alone explains the failure.
- **Probe:** conservative versus mixed-speed collection, crossed with wider versus
  bounded candidate grids, preserving the 21-candidate/20-step search size. The
  bounded grid also includes lower-speed choices, so this is not a pure upper-speed
  cap intervention. Collection uses equal episode counts, not exactly equal frames.
- **Follow-up:** repeat the unchanged mixed-data/bounded-grid condition with two
  additional collection seeds. All results remain adaptive development evidence;
  this is not a confirmatory or SOTA experiment.

## R7 — A weak learning baseline must not become evidence of superiority

- **Author/type:** assistant concern from the first recurrent-PPO pilot.
- **Context:** a 100k-native-frame default-configuration run failed to learn useful
  driving. Its score is an integration result, not a competitive baseline result.
- **Purpose/probe:** audit a published working configuration, preserve its exact
  source version, and separate faithful reproduction from our local adaptation
  before launching headline comparisons. Do not remove a strong competitor or
  advertise a win against an undertrained pilot.
- **Status:** adapter/reload/continuation tested; strong baseline reproduction and
  convergence remain open. Upstream configuration has substantial differences
  from the pilot and a much larger budget; see the baseline audit in the pilot report.

## R8 — Future prediction selection by regret, with continuous search

- **Author/type:** user future idea, 2026-10-28.
- **Context/intent:** use hindsight regret or candidate rank as the main metric
  for deciding what/how to predict, because prediction should support good driving
  choices. Also consider moving beyond the discrete candidate grid.
- **Status:** future option, not an activated objective or a change to the current
  user-only oracle-output restriction.
- **Assistant recommendation:** regret can be primary for this future prediction-
  selection study, with rank secondary. Rank discards mistake magnitude; a nearly
  tied second-best action and a costly second-best action differ in regret.
  Actual racing return/completion remain system-level paper endpoints.
- **Training interpretation:** this may select normally trained prediction spaces,
  models or hyperparameters through downstream decisions. It does not imply that
  we now train with regret gradients or replace ordinary prediction losses.
- **Continuous-search distinction:** steering/gas/brake are already continuous;
  the 21-point behavior parameter grid is discrete. Continuous optimization or
  adaptive proposals could search controller parameters or action sequences.
- **Fair probe:** fix the candidate set when isolating prediction ranking, or match
  optimization budgets when studying continuous search. Finite-candidate regret
  is relative to available candidates, not a proven global-control optimum; richer
  candidate coverage changes the reference as well as the choice.
- **Interpretation limits:** hindsight knows the realized hidden world. Nonzero
  hindsight regret may reflect irreducible partial information. Fix/report horizon,
  utility and candidate coverage, and validate ordinary closed-loop driving.
- **Access constraint:** existing private values remain unavailable to agents and
  training/tuning/selection. This future idea does not authorize reading them;
  permitted diagnostic use would need an explicit future change.

## R9 — Scientific validity and interpretation of every experiment

- **Author/type:** user requirement/reaffirmation, 2026-10-28.
- **Intent:** correct training/validation/test separation and run records are
  essential to the paper; results and what they mean must be thought through.
- **Implementation consequence:** every experiment records its information/access
  regime, data/compute budget, randomization, selection history, artifacts and
  interpretation limits. Distinguish infrastructure checks, adaptive pilots,
  exploratory replications and eventual frozen test evidence.
- **Current interpretation:** the pilot justifies further model/control and baseline
  work. It does not establish SOTA, reliable superiority, calibrated uncertainty
  or a full-race conclusion from short-horizon candidate ranking. Changes motivated
  by development results remain explicit; final test remains sealed.
- **Status:** required documentation practice. The first pilot report and campaign
  ledger implement it; user-only oracle values remain excluded from public analysis.

## R10 — Next moves should answer specific questions before larger runs

- **Author/type:** user request for the next work plan; assistant proposed ordering,
  budgets and confidence, 2026-10-28.
- **Intent:** make the next experiments useful to the paper and avoid spending
  training compute without knowing what each comparison establishes.
- **Plan:** [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md) records the immediate
  correction, matched fixed controls, failure diagnosis, learned baseline,
  conditional model improvements and broader development comparison.
- **Reasoning:** present results do not establish a search advantage, completion
  is lower, and the recurrent pilot is underdeveloped. More complicated road
  prediction is not yet justified as the first intervention.
- **Status:** plan documented; no new training or driving campaign launched in
  this planning task. Existing partition contracts and private-monitor restrictions
  remain in force. Confidence in useful experiments is not confidence in SOTA.

## R11 — Unknown-space dynamics must agree with the assumed world

- **Author/type:** assistant implementation finding and proposed correction,
  2026-10-28.
- **Evidence:** `planning/search.py` samples grip from the observed road raster
  with outside-array value 1. Masked unknown cells within the raster contain 0.
  A synthetic diagonal-road probe gave grip proxies [0,1] at (25,25) and (31,31),
  both outside the radius-30 sensor circle. No real track or private outputs
  were used in this probe.
- **Purpose:** remove an arbitrary difference between unknown regions caused by
  storage shape. Dynamics and scoring should query the same hypothesized road.
- **Probe:** test observed and hypothesized road/grass at rotated/boundary
  locations; rerun frozen models with all other settings unchanged.
- **Status/limit:** correction and paired reruns now complete; see
  [the control audit](CONTROL_AUDIT_2026-10-28.md). Mean return changed by −5.1
  with a wide interval and unchanged completion. No performance improvement established.

## R12 — Compare online choice with constants from the same candidate set

- **Author/type:** assistant experimental-control proposal, 2026-10-28.
- **Context:** the useful fixed-speed-60 reference is outside the revised search
  grid of speeds 20–50. This compares complete controllers but does not isolate
  adaptive choice within a shared set of behaviors.
- **Purpose/probe:** evaluate all 21 speed/gain constants on development tracks,
  select a campaign-wide constant, and compare with online search using the same
  low-level controller, representation and candidate set. Retain fixed60 as an
  additional reference and account for all tuning.
- **Interpretation:** selection on these tracks makes this a development study;
  broad development and eventual frozen test are separate stages. Investigate
  whether pace limits or recovery explain noncompletion before attributing it to
  prediction. Better mean return alone is not a completion improvement.
- **Status:** all 21 constants and failure categorization now evaluated; see
  [the control audit](CONTROL_AUDIT_2026-10-28.md). No search advantage established.

## R13 — Keep the main goal and architectural alternatives alive

- **Author/type:** user instruction/reaffirmation, 2026-10-28 (paraphrase).
- **Intent:** execute the next moves while remembering that the project's ideas
  may substantially change the architecture and research workflow.
- **Assistant response:** treat the current model and 21-candidate controller as
  a diagnostic implementation. Preserve the main observe/model/simulate/choose/
  act loop, existing-model reuse, prediction-space selection, probabilistic roads,
  continuous/adaptive or Bayesian search, expert experiments and future evolution.
- **Concrete consequence:** the first batch has a fixed scope and budget. Its
  report must state what to retain, replace or investigate next, rather than
  assuming that incremental tuning of the present controller is the paper.
- **Status:** recorded before launching the batch. Alternatives remain available;
  user-only oracle feedback and final-test restrictions remain unchanged.

## R14 — Completion failures motivate checking pace before larger predictors

- **Author/type:** assistant interpretation and next-probe proposal after the
  control-audit batch, 2026-10-28.
- **Evidence:** corrected search completed 12/60 episodes on 20 shared development
  tracks. There were 38 on-road timeouts, including 23 episodes with no fully
  off-road frames. The best constant used the upper speed/gain grid boundary.
  Correction of grip semantics did not establish a driving improvement.
- **Intent:** distinguish insufficient pace, candidate/controller limitations and
  missing information before attributing failure to road prediction or spending
  more training compute. These outcomes concern the current implementation.
- **Probe:** fixed60/gain1.2 versus already measured fixed50/gain1.2; separately,
  the same bounded planner and constant under declared full-map access. Proposed
  additional budget: at most 60k development frames; no new model training.
- **Limits:** endpoint road contact and tail speed do not prove the failure's
  cause. The selected constant is tuned on this panel, and the 60 episodes are
  not 60 independent tracks. Preserve the user's broader architectural options.
- **Status:** recorded after the completed batch; follow-up runs not started.

## R15 — Clarify prediction fitting versus decision scoring and regret

- **Author/type:** user questions and improvement priority, 2026-10-28.
- **Intent:** improve search and decisions substantially, understand what our
  predictor actually is, whether it is fitted for realism, and whether choices
  or predictor selection currently minimize regret.
- **Verified implementation:** learned ridge ego dynamics fitted to ordinary
  one-step transition error; mathematical pose integration; engineered visible
  reconstruction and straight unseen-road continuation. No neural dynamics,
  learned road distribution or regret-based fitting/selection is implemented.
  Online search minimizes a handcrafted rollout cost, not prediction error.
- **Assistant clarification:** distinguish (1) fitting prediction parameters on
  training data, (2) selecting predictor/control configurations using development
  racing, and (3) choosing the immediate action by simulated outcomes. These have
  different objectives. The user's earlier downstream-selection idea concerns
  the second stage and does not require changing the first stage to a regret loss.
- **Regret caveat:** subtracting each predicted utility from the same best
  predicted utility cannot improve the current ordering. Independent candidate
  outcomes can assess it, but existing private monitor values remain unavailable.
  This question is not authorization to use those values or change the loss.
- **Improvement direction:** test candidate pace/coverage and information first;
  then compare model capacity/history, objective/horizon and search alternatives
  with controlled changes. Judge the complete system by validation racing while
  reporting predictive errors separately. The goal is useful decisions, and the
  current implementation is replaceable; a larger predictor alone has no assured
  advantage.
- **Status:** clarification documented in [STATUS.md](STATUS.md); no new campaign
  or regret-based training launched by this explanatory task.

## R16 — Simulation, not a learned transition model, is the intended main engine

- **Author/type:** explicit user correction, 2026-10-28.
- **Intent:** use an algorithmic simulator written in code as the world model.
  Simulate behaviors and choose by expected simulated outcomes instead of
  training a model to make that behavioral choice. Compare against the underlying
  behavior model/controller from which candidates are proposed.
- **Consequence:** promote programmed vehicle physics to the primary engine.
  Preserve learned-dynamics pilots as baselines and archive the preceding source/
  plan checkpoint. Learned perception/proposers remain possible components; a
  learned transition network or critic is not the main default.
- **Assistant design:** reuse Gymnasium vehicle mechanics in separate worlds built
  from observations; estimate hidden wheel state; start with unchanged candidates
  and scoring for attribution, then improve maneuvers/search. The full detailed
  plan and uncertainties are in [SIMULATOR_SEARCH_PLAN.md](SIMULATOR_SEARCH_PLAN.md).
- **Novelty critique:** policy-guided lookahead and coded-physics MPC have prior
  art. Beating competent frozen proposers and established planners at declared
  information/compute budgets must be measured; SOTA is not promised.

## R17 — Execute autonomously and retain uncertain choices for discussion

- **Author/type:** user execution instruction, 2026-10-28, reaffirmed by “continue”.
- **Intent:** obtain actual results and information without repeated clarification
  interruptions; record choices where the assistant is unsure for later discussion.
- **Implementation:** coded engine plus synthetic tests, then a declared 20-track
  bounded/wider-candidate comparison. Record footprint approximation, latent
  wheel observer, discarded solver warm starts, straight-road prior and scoring
  limitations before interpreting results.
- **Boundary:** no private candidate-monitor feedback, no hidden evaluation-state
  cloning, no final-test access and no unsupported performance claim. Autonomous
  work remains within the established experiment protocol.

## R18 — Implement and measure the user's programmed-simulator architecture

- **Author/type:** assistant implementation and measurements under R16–R17,
  2026-10-28; the architecture intent comes from the user.
- **Purpose:** test the actual requested engine before investing in larger learned
  transition models. Preserve the existing candidate bank and score for an initial
  comparison, then separately widen candidate speeds.
- **Executed:** independent Box2D worlds using upstream vehicle equations, permitted
  local observations, estimated wheel state and declared footprint contact. P0's
  two conditions completed 40 episodes on 20 development tracks, within budget.
- **Evidence:** bounded physics 653.6 return / 20% completion; wider physics 737.3 /
  45%. One-step ego errors are substantially smaller than the frozen ridge models
  on exactly the same transitions, but bounded driving did not improve. Wider
  search also changes candidate pace, so its gain cannot be attributed solely to
  the engine. Its 128.5 ms p95 latency exceeds the 80 ms decision interval.
- **Critique:** the remaining problem includes candidate coverage, utility/horizon,
  geometry assumptions and latent state. One-step fidelity does not prove accurate
  counterfactual ranking. Do not read the user's private oracle to resolve this.
- **Record:** [combined report](PROGRAMMED_SIMULATOR_RESULTS.md), P0 manifest and
  public tables. No new model training or held-out evaluation.

## R19 — Test a smaller search bank and complete its fixed-controller controls

- **Author/type:** assistant adaptive design choice after P0, 2026-10-28.
- **Intent:** find a useful runtime reference and avoid comparing against an
  incomplete fixed-controller family. This is an engineering follow-up, not a
  new optimizer or a final user commitment to seven candidates.
- **Choice and uncertainty:** retain seven wider speeds with gain 1.2, selected
  from the earlier fixed audit; record the new manifest before execution. Reducing
  steering-gain coverage may lose useful behaviors as well as reduce compute.
- **Executed:** one physics condition plus eleven missing fixed conditions,
  240 new episodes on the same 20 tracks. Reuse ten existing fixed references.
- **Evidence:** seven-candidate search 697.8 return / 40% completion / 48.9 ms p95;
  best fixed across all 21 wider candidates remains speed50/gain1.2 at 716.1 / 20%.
  Mean-return difference intervals include zero. All these comparisons are
  development-adaptive and their intervals are not selection-adjusted.
- **Verification:** 66 tests passed; full episode replay reproduced all actions,
  rewards, observations and endings exactly. Isolated timing also confirms a cost
  reduction. Neither is independent evidence of better driving.
- **Decision:** retain both 7 and 21 as reference budgets. No final method chosen;
  all jobs finished. Costs, source/config provenance and limitations are indexed.

## R20 — Proposed next information diagnostic, with the same estimated car state

- **Author/type:** assistant proposal motivated by R18–R19, 2026-10-28.
- **Purpose:** determine whether improved available road geometry makes the current
  coded planner useful before adding more complex forecasting or search. This
  also advances the user's interest in known-track/expert experiments without
  treating a different expert controller as a clean information ablation.
- **Design:** explicit full-map adapter; same engine, estimated wheel state,
  utility and horizon; wider 7/21 candidates × 20 development tracks, at most
  40k native frames. Declare path-representation changes and an immutable manifest
  before execution; report a privileged table. No true wheel state or private
  candidate-ranking values.
- **Confidence:** medium. A geometry intervention may help locate a bottleneck,
  but a negative result will not uniquely identify state, scoring or search errors.
  It is not evidence of standard-input superiority.
- **Next decision:** strong improvement motivates permitted geometry reconstruction
  and shared road hypotheses; otherwise prioritize maneuvers, utility/horizon and
  state/contact diagnostics. Strong learned proposers and established planning
  comparisons remain required in either case.
- **Status:** proposed, not implemented or launched. The user's main architecture
  remains intact; regret objectives and agentic evolution remain future ideas.

## R21 — Test improvement over a stronger underlying driver; make the next plan concrete

- **Author/type:** user question and planning request, 2026-10-28.
- **Intent:** assess whether improving an already strong or SOTA driver would
  demonstrate that our simulation-selection method is useful, and obtain a
  thoroughly considered plan for the next moves.
- **Clarification:** the stronger underlying model here is the behavior proposer.
  Programmed physics remains the primary world model. The current experiment uses
  simple feedback controllers, not a strong learned policy. Stronger proposals
  may help, but wrong world estimates or utility can also harm a better driver.
- **Assistant recommendation:** preserve one capped same-planner geometry diagnostic,
  prepare a competent recurrent-policy baseline, freeze its weights and compare
  standalone behavior, random variation and coded simulation selection. Then add
  established search with the same simulator/access/budget and broader development
  replications. Preserve the original base behavior in the candidate set.
- **Critique:** policy-guided planning already exists, including POPLIN. Improvement
  over a verified strong driver is meaningful evidence under the tested conditions,
  not automatic novelty, general usefulness, or standard-input SOTA. Extra physics
  knowledge, latency and information advantages must be disclosed and controlled.
  A planner's policy-only ablation must not replace its full original method.
- **Concrete record:** [NEXT_MOVES.md](NEXT_MOVES.md) specifies implementation
  prerequisites, limited experiment budgets, a staged 2M/8M first-driver proposal,
  branching-memory requirements, paired-checkpoint statistics, reproduction/input
  audits, broader development and final freeze. Numerical choices are assistant
  proposals, not additional user commitments. Hardware was inspected; training
  throughput and final calendar cost remain unmeasured.
- **Status:** plan documented and indexed. No new training/evaluation started.
  Private oracle, calibration and final-test access remain unchanged.

## R22 — Execute the full-map diagnostic without changing vehicle-state access

- **Author/type:** user authorized execution with “lets go”; assistant implementation
  and measured result, 2026-10-28.
- **Purpose:** test the geometry/reconstruction bottleneck using the same coded
  physics, estimated wheels, utility, horizon and candidate controllers. It also
  addresses the user's known-track/expert idea without substituting a different
  expert controller and attributing its advantage solely to information.
- **Intervention:** explicit development-only global protocol with full road
  boundaries and ordered centerline; same local raster inside the sensor circle.
  Outside geometry and centerline reconstruction change together. No true wheel
  state, solver state or user-only oracle is supplied.
- **Evidence:** both 7- and 21-candidate conditions completed 20/20 tracks, at
  923.15 and 923.60 mean return; local counterparts completed 40% and 45%. New
  cost 30,651 native / 8,584,800 internal physics frames, within declaration.
  The shared-code local replay check matched exactly.
- **Interpretation:** important geometry/reconstruction bottleneck in this planner;
  not an isolated unseen-road effect, proof that search is necessary with full-map
  access, mathematical upper bound or native-input SOTA result.
- **Decision:** preserve the stronger-proposer comparison. Future local-map work
  should separate visible reconstruction from unseen-road continuation instead
  of immediately attributing the entire gap to a learned forecast.
- **Record:** [geometry/driver report](GEOMETRY_AND_DRIVER_2026-10-28.md); campaign
  manifest, public summary, uncertainty, outcome records and figure are indexed.

## R23 — Make stronger-driver training reproducible, then choose hardware empirically

- **Author/type:** assistant implementation decisions under the accepted execution
  plan, 2026-10-28. No claim that this PPO adaptation is current SOTA.
- **Intent:** obtain a competent frozen proposer with complete training and
  checkpoint provenance rather than reuse the failed 100k engineering reference.
- **Implemented:** resolved settings, learning-only reward normalization, native-
  frame global learning schedule, post-update checkpoint/optimizer/RNG/stream
  state and recurrent reload/continuation checks. Evaluation cannot update running
  normalization. The old trainer remains available as historical infrastructure.
- **Unexpected issue:** deterministic CUDA backward through adaptive-average pooling
  failed. Preserve all three failed profiles. An equivalent pooling implementation
  passed CPU/CUDA value-and-gradient checks before corrective profiling.
- **Choice:** CPU with four subprocess workers and one Torch thread, based on
  measured successful update throughput. Corrected CPU4 was about 1,457 native
  frames/s; CUDA4/8 were about 776/766. The single-update profile is approximate.
  Combined profiling used 90,112 training frames within the original 100k cap.
- **Execution:** first seed-zero block runs to 2M with an unchanged 8M global
  schedule; four scheduled checkpoints each receive 50-track standalone development
  evaluation. User authorization covers this execution; there is no new approval
  gate at each checkpoint. Review learning health before extending the budget.
- **Uncertainty:** convergence and competitiveness remain empirical. Early
  checkpoints show learning but do not establish a strong final baseline.

## R24 — Prepare simulation selection around a frozen recurrent policy

- **Author/type:** assistant implementation of the accepted same-proposer experiment,
  2026-10-28; user intent is to choose behavior by simulation rather than train a
  new behavior selector.
- **Choice:** three smooth residual knots around frozen feedback-policy actions,
  including a zero-residual base candidate. Initial 7/21 budgets, 1.6 s horizon
  and existing coded utility remain explicit development settings.
- **Boundary:** future local observations come from hypothesized worlds; imagined
  recurrent memories are isolated and never committed to the real policy. The
  physical wheel observer is separate. Random selection shares the proposal rule
  with a separate selection random stream and performs no simulation.
- **Evidence:** candidate-zero action/memory equivalence, branch isolation,
  action bounds, reset reproducibility and sensor-field checks pass. A train-only
  preflight used 320 native / 6,720 internal physics frames; contended snapshot
  timings were roughly 172–189 ms for 7 and 478–523 ms for 21 candidates.
- **Critique:** repeated neural inference and imagined observations are expensive.
  These timings overlap training/tests and are not isolated latency quantiles.
  No racing improvement has been measured for this wrapper; retain that distinction.
- **Statistics:** add a joint checkpoint/track bootstrap for the same frozen
  checkpoints with/without search. Independent-model resampling remains for
  separately trained algorithms; matching numeric training seeds is not pairing.
- **Status:** wrapper and reporting implemented; actual paired driving campaign
  awaited the standalone-driver health review/selection at this implementation
  checkpoint. It has subsequently launched under R25. No private oracle feedback.

## R25 — Preserve the best driver, stop an unhealthy extension, measure search

- **Author/type:** assistant execution and health-review decision under the user's
  "lets go" authorization, 2026-10-28. The user requested useful results and a
  record of uncertain choices, rather than an unexamined long training run.
- **Evidence:** four checkpoints on 50 common development tracks progressed from
  584.9/2% to 734.6/6%, then 858.1/30%, before regressing to 779.2/2% at 2M.
  Values are mean official return/completion. All diagnostics remained finite;
  the later checkpoint had substantially more off-road time. Approximate KL and
  clipping increased later in training, but these logs do not establish causality.
- **Decision:** apply the already declared standalone selection rule and freeze
  the 1,007,379-frame checkpoint. Stop the unchanged recipe at the 2M health review.
  No additional training is running. The selected policy is useful for a bounded
  exploratory comparison; it is not our final converged/strong baseline.
- **Costs:** 2,006,414 training native frames, 196,576 checkpoint-evaluation frames
  and eight reload-check frames. All four checkpoints and the regression remain
  reported. Earlier profile costs remain additional and are not hidden.
- **Next experiment:** execute the planned direct, random-variation (three planning
  streams) and simulated-selection (7/21 candidates) comparison, without choosing
  the checkpoint based on search benefit. Six conditions on 20 development tracks,
  cap 120k native and 11.2M internal frames; same frozen weights and permitted inputs.
- **Timing:** an additional selected-checkpoint preflight without project training
  took 148–161 ms for 7 candidates and 401–436 ms for 21; three snapshots, not
  quantiles. Cost 320 native / 6,720 internal frames. Run offline quality now;
  do not claim deadline-respecting behavior or add an undeclared fallback.
- **Statistical caution:** one selected model, repeatedly used development tracks,
  and fixed planning streams. Those streams reset per episode and are shared
  across tracks; they are not independent learned models or per-track seeds.
  Random controls are all retained; intervals are conditional track uncertainty.
- **Future repair probe:** separately declare a conservative PPO-update trial,
  possibly a target-KL cap, while retaining configuration/continuation and cost
  provenance. Do not call the suspected optimization issue a proven explanation.
- **Records:** [execution report](GEOMETRY_AND_DRIVER_2026-10-28.md),
  [health review](../reports/driver_first_block_2026-10-28/health_review.json), and
  [frozen-driver manifest](../configs/campaigns/frozen-driver-2026-10-28.json).

## R26 — Measured improvement around one frozen learned driver

- **Author/type:** assistant result review of the user's main same-proposer idea.
- **Evidence:** on 20 common development tracks, direct driving scored 863.55/40%
  return/completion, search 7 scored 885.70/70%, and search 21 scored 905.94/85%.
  The 21-minus-direct return difference is +42.39 [14.36, 75.60], an exploratory
  track-only interval conditional on one selected checkpoint and fixed streams.
- **Purpose:** test choosing by coded simulated outcomes without learning a new
  selector. The selected checkpoint was fixed from standalone performance first.
- **Limits:** reused development tracks, one model, no multiplicity/selection
  adjustment or native-input comparison. At 171.6/457.5 ms p95, the search is costly.
  Random variations drive very slowly; beating them is weaker evidence than the
  same-policy contrast. No private per-action correctness/rank signal was read.
- **Validation:** identical source/checkpoint hashes for six conditions, public
  trace/work consistency and exact replay checks passed. All controls are retained.
- **Decision:** preserve this positive but narrow result; broaden development
  and compare established search before claiming general algorithmic superiority.

## R27 — A controlled PPO repair did not work

- **Author/type:** assistant single-factor probe after the original regression.
- **Hypothesis:** the upstream target-KL stop might prevent harmful large updates.
  Fresh seed-zero training changed only that algorithm setting to 0.02, retaining
  the original architecture, schedule and data/checkpoint budgets.
- **Result:** best checkpoint 697.04/4%, final 570.17/0%; worse than the original
  selected driver. Early stopping reduced optimizer steps from 39,200 to 2,983;
  205/245 rollout updates stopped within their first epoch. All diagnostics finite.
- **Decision:** do not adopt or extend this setting. The original frozen actor
  and search table stay unchanged. A later lower-learning-rate probe retaining
  ten epochs is proposed, not launched. Do not infer universal KL harm or a proven
  cause of the original regression from this one trial.
- **Costs:** 2,006,884 new training frames, 195,829 development evaluation frames,
  eight reload-check frames; failed hypothesis retained in the public report.

## R28 — Improve computational work without changing simulated decisions

- **Author/type:** assistant profiling observation and implementation refinement.
- **Purpose:** reduce the high search cost before expanding evaluation. Instrumented
  cached training observations implicated imagined road rendering; it calculated
  margins and nearest-path distances not needed for binary policy observations.
- **Change:** explicit sparse-occupancy renderer, retaining the same world, coordinate
  transform and threshold semantics. Original rendering remains the default.
- **Evidence:** edge/random tests, sampled candidate actions/costs and two complete
  race replays matched exactly. Median three-snapshot times fell about 30–35%.
  Training contention and small samples prevent replacing the main latency table.
- **Limit:** both remain above 80 ms. This is a computational refactor, not a new
  prediction model or proof of real-time performance. No Rust/evolution campaign.

## R29 — Test whether a cheap constant correction explains the gain

- **Author/type:** assistant alternative explanation, tested before broad scaling.
- **Question/purpose:** could adding fixed steering/gas/brake offsets to the actor
  recover the benefit attributed to adaptive online simulation?
- **Design:** average executed-minus-base actions per episode and then equally
  across 20 public development episodes. Declare full mean-7, full mean-21 and
  gas-only mean-21 controls before racing. No private oracle data or new labels.
- **Evidence:** return/completion 860.45/60%, 763.67/10%, 738.93/5%. The first gives
  cheap completion improvement but no mean-return gain; none recovers search 21's
  905.94/85%. Costs: 56,896 native frames, zero internal physics, zero new training.
- **Limit:** offsets are fitted/evaluated on the same adaptive panel. This excludes
  these particular constant corrections, not all cheaper policies or scoring methods.
- **Record for R26–R29:** [completed report](FROZEN_POLICY_RESULTS_2026-10-28.md),
  linked declarations, tables, uncertainty, failure evidence, costs and next moves.

## R30 — User's future full-information and cross-information study

- **Author/type:** user future proposal and broader scientific ambition, paraphrased.
- **Intent:** show what trying behaviors in simulation offers when the entire road
  is known; also test whether restricted-information search can outperform a learned
  driver trained/deployed with full information. Separate road understanding from
  navigation, possibly into three modules, to explain benefits with imperfect parts.
- **Assistant design:** local/full-map × direct learned/same-frozen-policy search,
  with road inference, vehicle prediction and decision/control as diagnostic modules.
  Add competent full-map learning and established planning controls. Distinguish
  known geometry from hidden vehicle/solver state and known-map deployment from
  per-track optimization. Existing global diagnostics do not fill this matrix.
- **Critique:** a full-information optimal controller can ignore extra information
  and emulate a local method when allowed the same computation. A practical crossover
  would establish a result for tested algorithms/resources, not universal superiority
  over learned policies or a new field. PETS/POPLIN and MPC are relevant prior art.
- **Status:** documented future study; no global training, exact-state cloning,
  held-out access or private ranking feedback authorized by this entry. The user
  explicitly placed it in future work; current work remains the main local system.
- **Record:** [information/control study](INFORMATION_AND_CONTROL_STUDY.md), including
  author intent, precise contrasts, confidence, fairness and possible conclusions.

## R31 — Broaden before changing the successful search configuration

- **Author/type:** assistant experiment choice under the user's instruction to execute.
- **Purpose:** distinguish a repeatable one-policy gain from an effect confined to
  the repeatedly inspected 20-track panel, retaining the already fitted cheap control.
- **Decision:** four frozen conditions on all 200 development tracks; no new tuning,
  fixed checkpoint and planning stream, nested 20/50/150-track reports, paired intervals.
- **Implementation:** immutable executable source snapshot; sequential conditions;
  exact first-20 replay gate, dependency/hash checks and automatic public report.
- **Confidence/limit:** high confidence in the controlled comparison; uncertain
  generalization and latency value. One checkpoint does not establish training-seed
  robustness. All panels remain development data, and fixed order limits timing inference.
- **Status:** launched, full results pending. Caps 800k native / 112M internal frames.
- **Record:** [campaign declaration and analysis](BROAD_DEVELOPMENT_2026-10-28.md).

## R32 — User requests continued autonomous execution and explicit uncertainty

- **Author/type:** user's execution instruction, paraphrased.
- **Intent:** keep implementing, running and reviewing experiments toward a real
  scientific paper without waiting for repeated confirmation; document uncertain
  decisions, alternatives and ideas so they can be discussed later.
- **Assistant interpretation:** proceed through bounded, recorded development
  probes; preserve failures and revise plans using public evidence. Do not silently
  promise success, replace the main system with an optional idea, or skip validation.
- **Boundary:** final study remains unfrozen; no test access or private-oracle feedback.
  Independent baseline repair may be queued after the latency-sensitive campaign;
  methodological choices that depend on its results wait for the completed analysis.

## R33 — Queue a smaller-update baseline probe without changing the running study

- **Author/type:** assistant single-factor development decision under R32.
- **Purpose:** seek a stronger standalone learned controller after 2M regression
  and the failed KL-stop trial. Retain meaningful optimization instead of combining
  multiple unidentifiable changes.
- **Choice:** fresh seed zero, learning rate 3e-5, ten epochs and no KL stop; original
  architecture and 8M schedule, initial 2M review, four 50-track checkpoint evaluations.
- **Uncertainty:** the learning rate is a conventional scale change, not a proven
  fix; slower learning can look worse at 2M. One run cannot establish causality.
- **Alternatives deferred:** additional seeds at the original rate; representation
  or reward-normalization changes. None is silently combined with this probe.
- **Execution:** immutable source/config/script snapshot queued after the broad
  comparison and regression checks. Stop on failed/stale predecessor. No automatic
  8M extension, test evaluation or replacement of old checkpoint identities.
- **Status/cost cap:** queued, not trained; 2,008,191 training plus 200k evaluation
  frames, separate reload-check cost. Failed attempts remain part of development.
- **Record:** [declared probe](../configs/campaigns/driver-lr3e5-block-2026-10-28.json)
  and [rationale/queue contract](BROAD_DEVELOPMENT_2026-10-28.md#queued-baseline-probe).

## R34 — Audit established adaptive search before claiming its advantage

- **Author/type:** assistant comparison design, grounded in official iCEM source.
- **Intent:** test the user's central simulation-selection system against an
  established optimizer, not present finite enumeration as an algorithmic novelty.
- **Finding:** population size omits later iterations; within-decision elite reuse
  caches rollouts, whereas shifted candidates require new simulations. Very small
  elite counts can round the nominal reuse fraction to zero.
- **Proposed design:** retain the frozen feedback actor and model; compare full-
  sequence colored shooting with an adaptive residual-search implementation, then
  use the existing knot search as a separate representation reference. Explicitly
  label feedback residuals, incumbent retention and constrained populations as adaptations.
- **Confidence:** high on fair budget/memory/state accounting; low on the initial
  noise exponent, residual bounds and small-budget optimizer competence. Improved
  predicted cost can still worsen driving through model error.
- **Status:** pinned source and license audited; no optimizer implementation or
  driving evidence. The 200-track frozen campaign continues unchanged.
- **Record:** [audit, choices, alternatives and pilot gates](SEARCH_OPTIMIZER_AUDIT.md).

## R35 — User requests completed-condition results during the broad campaign

- **Author/type:** user status/results request; assistant verified interim analysis.
- **Scope:** three fully completed 200-track conditions, excluding unfinished search 21.
- **Evidence:** direct 862.42/35.5%, fixed correction 865.95/39.5%, search 7 873.05/58.5%.
  Search7-minus-direct completion +23 percentage points [16.0, 30.5]; return +10.64
  [-2.56, 22.80]. Lower off-road exposure but 136.77 ms p95 decision pipeline.
- **Meaning:** a completion gain for this frozen policy, with uncertain mean-return
  benefit and substantial runtime cost. No independent-policy or SOTA conclusion.
- **Integrity:** source/checkpoint/protocol/partition/cost and replay checks passed;
  all declared nested panels retained. Additional fixed-correction pairwise
  contrasts are labeled descriptive. No method or queue changes from this view.
- **Record:** [table, uncertainty and provenance](../reports/broad_driver_completed_2026-10-28_1810/README.md).

## R36 — User's future independent, fast simulator selected for useful decisions

- **Author/type:** user proposal followed by explicit clarifications, paraphrased.
- **Intent:** build our own simulator, possibly GPU-fast, adapt it through parameters
  or agentic rewriting/evolution, and evaluate its usefulness to final behavior
  choices. The user explicitly says it need not be realistic; usefulness is enough.
- **Motivations:** test whether improvement requires target simulator code or exact
  simulation, and motivate possible eventual real-world use through weaker assumptions.
- **Assistant design:** separate source reuse, model approximation, downstream model
  selection, backend speed and code evolution. Compare prediction-calibrated and
  driving-selected models with fixed control; compare parameter-only tuning against
  code adaptation with recorded feedback budgets. Count adaptation as optimization.
- **Critique/confidence:** high confidence in separating these claims; advantage
  unknown. We already know upstream code, so independent implementation is not
  proof of source-unavailable design. Gym success is not sim-to-real evidence.
  GPU throughput and task-guided models have prior art; no new result is asserted.
- **Status:** future option only. No new engine, GPU dependency, evolution agent or
  campaign. Normal predictive training remains the default for the earlier predictor
  study. Private rank/regret feedback is still forbidden; current runs are unchanged.
- **Record:** [intent, controls, sources and open decisions](DECISION_USEFUL_SIMULATOR.md).

## R37 — User reaffirms the main understanding-versus-decision goal

- **Author/type:** user clarification and request to preserve the project's ideas.
- **Intent:** distinguish understanding the world from deciding what to do; identify
  which parts are already sufficient and which remain limiting; demonstrate whether
  simply choosing among sampled variants improves the underlying driver's results.
- **Assistant interpretation:** freeze proposer/representation/model when testing
  selection, then use controlled component substitutions to attribute limitations.
  Include direct execution, random variation and simulated selection. The third
  diagnostic split remains road inference, vehicle prediction and decision/control.
- **Claim limit:** a successful complete system does not prove its understanding is
  solved, and an inaccurate model may still be sufficient. Adequacy is conditional
  on tested tasks, information and resources. Simulated selection need not be
  universally superior, nor can it enumerate all continuous behaviors.
- **Priority:** this is the main research question. The custom simulator, GPU,
  agentic evolution and predictor-selection ideas remain supporting future options.
- **Action:** foregrounded the goal in README, specification and USER_INTENT;
  linked the detailed future simulator proposal and retained existing study order.

## R38 — User specifies oracle-current, oracle-future and action-selection interventions

- **Author/type:** user's detailed paper formulation, 2026-10-29, paraphrased.
- **Intent:** distinguish three difficulties: current-scene understanding, prediction
  of future geometry/action consequences, and choosing the driving action. Train
  comparable policies on true current information and then true future road; compare
  each with selection among simulated variants of that same policy. Identify which
  difficulties remain and test strong policies as well as the initial PPO.
- **Further user ideas:** imperfect or independently implemented simulation may
  already suffice under partial information; prediction/model selection by final
  driving usefulness; optional evolution; untouched final evaluation. The desired
  action-selection-bottleneck explanation remains a hypothesis.
- **Assistant refinements:** distinguish retraining-under-information from controlled
  component replacement; maintain shared interfaces and test distribution shift;
  split hidden-road uncertainty from action-conditioned vehicle motion. Extra
  hidden-road knowledge can remove irreducible uncertainty, not merely fix errors.
- **Limits:** a remaining search gain depends on model, objective, candidate coverage
  and compute. It cannot establish that every RL policy controls poorly or that an
  end-to-end policy's perception is solved. No private oracle outputs are opened.
- **Status:** proposed study, no new oracle-training protocol or campaign launched.
- **Record:** [central blueprint and comparison matrix](PAPER_BLUEPRINT.md).

## R39 — User requests one consolidated idea register with explicit origin

- **Author/type:** user's organization and attribution instruction, 2026-10-29.
- **Intent:** collect ideas from the entire project discussion, use them to improve
  the paper, and clearly distinguish the user's ideas from the assistant's proposals.
- **Action:** central blueprint with 26 user ideas and 13 assistant refinements, plus
  explicitly attributed prior work. Origin and evidence status are separate fields.
  Conversational origin is not a claim of scientific priority. Detailed earlier
  rationale, corrections, failed experiments and archived work remain linked.
- **Rule:** future updates must preserve this distinction; do not silently convert
  an assistant numerical choice or an untested user hypothesis into an agreed fact.

## R40 — Completed broad evidence and failed smaller-update probe

- **Author/type:** measured evidence; assistant verification and continuation choice.
- **Results:** all 200 development tracks complete. Direct 862.42/35.5%; fixed
  correction 865.95/39.5%; search 7 873.05/58.5%; search 21 884.48/72.0%.
  Search 21 minus direct return +22.06 [10.18, 33.35] and completion +36.5 points
  [29.5, 44.0], nominal paired track-only intervals for one checkpoint/stream.
- **Cost/limit:** 99,945,440 internal native physics steps and 737,023 native evaluation
  frames; search 21 p95 latency 362.11 ms exceeds the 80 ms action interval.
- **Failed probe:** lower-LR trial best 692.18/8% on 50 tracks; final 624.51/0%.
  Cost 2,006,916 training plus 199,612 evaluation and 8 reload-check frames.
- **Assistant decision:** retain the original selected actor, preserve the failure,
  do not adopt/automatically extend the repair. Strong baseline verification remains
  necessary; this does not establish that learned action selection is intrinsically poor.
- **Verification:** both reports complete; input/checkpoint hashes checked, figures
  inspected; queued full regression suite 80 passed. Final test/private data untouched.
- **Record:** [completed review and artifacts](BROAD_RESULTS_2026-10-29.md).

## R41 — Close prior art sharpens the novelty target

- **Author/type:** assistant literature assessment responding to R38–R39.
- **Finding:** privileged-input driving, separating perception/planning, policy-guided
  online planning, distillation and task-useful models are established. An October 26,
  2026 preprint also explicitly diagnoses planning failures despite successful detection.
- **Assistant proposal:** seek a validated three-stage intervention study and measured
  residual decision headroom, not novelty for the general decomposition or search idea.
  Test whether the result survives competent policies, information/model controls,
  computational budgets and a direct-policy distillation comparison.
- **Confidence:** high that these controls matter; moderate potential for a substantive
  empirical paper; unestablished algorithmic novelty/SOTA or acceptance probability.
- **Record:** primary citations and claim-specific distinctions in
  [the blueprint](PAPER_BLUEPRINT.md#prior-work-and-novelty-assessment) and REFERENCES.

## R42 — Execute independent module interventions before new oracle-policy training

- **User origin:** proceed with the best practical intervention design and execute it.
- **Assistant design:** a fixed-controller 2×2 study independently replaces the
  objective's visible reference and the road model's hidden continuation. The
  previous global diagnostic changed both at once, so it could not attribute gains.
- **Implementation:** the same frozen recurrent actor receives only local inputs;
  oracle fields terminate at the explicit development adapter. Vehicle observer,
  candidate generation, horizon, weights and outside-footprint penalty stay fixed.
  A reference change cannot leak into road contact/margins or its prior.
- **Interpretation:** true visible directed centreline adds privileged geometric
  labels; it diagnoses reconstruction/objective sensitivity, not neural visual
  perception. Oracle-derived hidden road remains rasterized. No complete vehicle
  state, universal optimality, or exact three-way RL error attribution is claimed.
- **Sanity checks:** exact negative-control action/cost/memory parity; root
  observation invariance; separate positive effects of both switches; hidden
  detour invariance and missing-reference fallback. Full previous-episode replay
  gates the nonbaseline cells. No private candidate oracle is read or used.
- **Bounded pilot:** five existing development tracks × four cells, seven
  candidates × 20 decisions, seed 0, at most 20k native and 2.8M internal physics
  frames, zero new experiment training. All results reported, no significance or
  favorable-outcome expansion. Numerical settings/order are assistant choices.
- **Infrastructure finding:** unrestricted pytest discovery found duplicate tests
  in saved source snapshots. Discovery now targets only active `tests/` and
  excludes run/private directories; old artifacts remain unchanged.
- **Next:** review coverage and complete pilot outcomes, then a separately declared
  larger comparison and V/L/F training interfaces with competence/distribution
  controls. Baseline validation and established planning comparisons remain open.
- **Record:** [intervention design](INTERVENTION_STUDY.md) and
  [frozen declaration](../configs/campaigns/intervention-pilot-2026-10-29.json).

## R43 — Oracle interventions hurt driving despite better executed-action prediction

- **Origin:** measured results; assistant post-hoc audits and next-control proposal.
- **Pilot:** all four cells completed, five common development tracks. Baseline
  906.99/80%; visible-reference oracle 758.96/0%; hidden-road oracle 429.64/0%;
  both 594.21/0%. Exact five-episode baseline replay passed. No reference fallbacks.
- **Failure description:** all 15 oracle episodes were on-road timeouts. Sampled
  mean speeds 58.80 baseline versus 43.37/26.90/35.21 m/s; braking increased.
  These are executed outcomes, not private candidate ranks or regrets.
- **Assistant follow-up:** declare a bounded audit of recorded baseline actions,
  roots 50/100/150 on all five tracks, same estimated initial state and controls.
  At 20 decisions, velocity-vector RMSE fell from 31.34888 to 0.00536 m/s with
  oracle road. One-decision predictions matched exactly. No new real frames.
- **Inference boundary:** better recorded-action prediction and worse driving
  coexist here. This does not establish candidate-state fidelity or a unique
  cause; reference/progress utility, horizon and imagined-policy inputs remain
  possible limitations. Never call the three RL subproblems solved from this.
- **Next assistant proposal:** separately declare a full-road objective-reference
  control with the oracle road model, since the current `both` cell still extends
  its visible objective by a tangent. Then validate compatibility/competence before
  large V/L/F training. Do not hide failures or expand favorable subsets.
- **Cost:** 19,447 native evaluation and 2,729,240 internal physics frames including
  the 5,400-step prediction audit; zero new experiment training. All jobs complete.
- **Record:** [full results, provenance and caveats](INTERVENTION_RESULTS_2026-10-29.md).

## R44 — Execute the reference-continuation control

- **User instruction:** plan the next run and execute it, following the preceding
  discussion of reference/progress scoring.
- **Assistant refinement:** keep every true visible-reference sample unchanged;
  replace only its 100 m tangent tail with the directed true-road continuation,
  at the same one-metre arc sampling. Replacing the whole path would also change
  its visible parameterization and weaken attribution.
- **Fixed:** oracle road/vehicle model, actor/local inputs and memory, seven
  candidates, 20 decisions, repeat four, random stream, residuals and score weights.
  The path still affects both progress and centre distance; neither is isolated alone.
- **Checks:** synthetic straight/curved/wrap/duplicate cases; unchanged world,
  imagined inputs and candidate trajectories at a shared root; every recorded
  control prefix/extension length; exact one-episode control replay before racing.
- **Budget:** five new development races, at most 5k native/700k internal frames;
  replay separately capped at 1k/140k. No new training or held-out/private data.
- **Interpretation:** paired descriptive contrast against five saved controls,
  with local search as context; no outcome-based sweep or panel expansion.
- **Record:** [protocol and execution](REFERENCE_CONTINUATION_2026-10-29.md).
- **Completed outcome:** new true-tail condition 876.29/80%, tangent control
  594.21/0%; +282.09 return with gains on all five tracks. Original local search
  remains higher at 906.99/80%, with one new on-road timeout substantially worse.
  Speed recovered from 35.21 to 57.60 m/s; braking decreased. The path affects
  progress and centre distance together, so neither term is individually identified.
- **Verification/cost:** 88 tests; all 1,255 recorded geometry states passed;
  one full exact control replay. Total 5,321 native and 745,360 internal physics
  frames including replay, zero new training, 277.38 s campaign wall time.
- **Next assistant recommendation:** a separately declared 20-track paired
  replication before tuning to the remaining case; retain original-five/additional-
  fifteen development subsets. No automatic expansion or new training has launched.

## R45 — Replicate the frozen reference-tail contrast before tuning

- **User origin:** continue the intervention work autonomously and preserve idea attribution.
- **Assistant choice/purpose:** twenty fixed development tracks, both unchanged
  oracle-road reference conditions, to check whether the five-track recovery
  extends beyond the cases that motivated the correction. No tuning to track 2.
- **Declared analysis:** all20/original5/additional15 panels; official return and
  completion plus failures, executed speed/braking and costs. Nominal paired-track
  intervals for all20/additional15 only, one frozen actor/planning stream, no
  selection adjustment. Local search is historical different-information context.
- **Budget/checks:** 40k native / 5.6M internal physics frames, zero training;
  exact first-five replays in each condition included in that budget. Hash-pinned
  sources, checkpoint, input trajectories and runner; stop on integrity failure.
- **Uncertainty:** twenty is a bounded development budget, not a power calculation;
  original5 are selected development evidence, additional15 are not final test.
  No new global-policy training, held-out evaluation or private-ranking access.
- **Record/status:** [replication declaration](REFERENCE_REPLICATION_2026-10-29.md);
  90 tests and artifact/CLI/link checks passed; serial campaign is complete.
- **All20 measured outcome:** true tail 813.85/60%, tangent 527.60/0%, historical
  local 885.70/70%. Primary +286.25 [231.59, 343.19]; local context −71.86
  [−122.60, −28.95]. Intervals are nominal paired-track development intervals only.
- **Additional15:** +287.64 versus tangent [216.83, 362.18], gains on 14/15 tracks;
  −85.57 versus local. Track 13 is a counterexample (365.52 versus tangent 375.86).
  Eight corrected-tail timeouts remain, including two tracks completed by local.
- **Checks/cost:** ten complete exact replays; frozen/live source and artifact
  hashes and all 60 reported trajectories checked; 38,031 native / 5,326,720
  internal physics frames, zero new experiment training, 1,962.86 s total.
- **Assistant interpretation/next choice:** reference compatibility matters in this
  configuration, but corrected oracle search is still worse than local. Close this
  bounded diagnostic; retain local as the working method and return to the audited
  equal-work established-optimizer comparison. Its gain is uncertain; competence,
  independent seeds and V/L/F training still matter. No new run/200-track expansion,
  targeted failure tuning, private-ranking access or global training is launched.

## R46 — Compare adaptive search without conflating proposal representation

- **User origin:** continue the needed experiments without routine permission stops;
  document choices and findings. The main aim remains programmed simulation for
  selecting behavior from a frozen proposer.
- **Prior-work origin:** iCEM colored proposals, elite fitting/reuse and shifted plans;
  pinned official source and MIT attribution are retained.
- **Assistant decisions:** full 20-step residuals, matched 13+8 batches, 21 total
  evaluated trajectories including the unchanged base, β=1, ±3 channel-σ bounds,
  four elites, smoothing 0.1, one cached and one shifted elite. These are economical
  starting choices, not known-optimal settings. The finite-DFT noise convention,
  residual-space adaptation and global incumbent differ from upstream defaults.
- **Why:** the primary shooting control shares representation, bounds, batching,
  actual physics work and actor examples; comparison with old three-knot search
  otherwise confounds representation and optimization. No new value scorer is trained.
- **Critical checks:** real recurrent memory/root state cannot advance through
  imagined batches; execute the best evaluated candidate and commit its own first
  physics estimate even if it came from an earlier batch. Re-simulate shifted plans;
  never reuse previous-root costs. Synthetic tests cover these failure modes.
- **Staging:** one old episode replay plus two episodes per new method, then twenty
  per method under unchanged settings if technical checks pass, irrespective of
  performance. Cap 45k native / 18.9M internal frames across both stages, no training.
- **Status/record:** all 100 tests, historical replay, smoke gates and the fixed
  twenty-track comparison passed their technical checks under unchanged settings in
  [the resolved experiment](SEQUENCE_SEARCH_2026-10-29.md). The two-track smoke
  favored colored shooting over the adaptive variant (907.26 versus 568.97 mean
  return); this descriptive negative result did not trigger outcome-based tuning.

- **Completed outcome:** colored 884.86/70%, iCEM 634.13/15%, historical knots
  905.94/85%. Primary −250.72 [−359.57, −147.31]; additional18 −240.99. Colored
  also loses 21.08 versus knots. No outcome-driven setting changes occurred.
- **Failures and diagnostic:** iCEM has fifteen timeouts and two playfield exits;
  its three fast completed laps do not offset those failures. A clearly post-hoc
  executed-trace audit finds larger corrections, more departures and larger
  short-step prediction error on different visited states; cause remains unidentified.
  This audit uses no private ranks/regrets or unexecuted candidate outcomes.
- **Final verification/cost:** 100 tests, five exact replays, 80 reported trajectories
  and all pinned/frozen inputs verified. Including smoke: 45 new episodes, 40,972
  native frames / 17,220,000 internal physics steps, zero training, 58.35 minutes.
- **Assistant decision:** keep original knot search; close this bounded optimizer
  trial and preserve its negative result. No automatic 200-track expansion or
  targeted tuning. Prioritize competent baselines using R47's audits. No active
  jobs remain; strong-baseline, independent-seed, V/L/F and final-paper gates stay open.

## R47 — Audit baseline competence and comparator protocols while races run

- **User origin/purpose:** improve credible frozen proposers, compare fairly with
  existing work, and document choices and findings honestly.
- **Assistant action:** read pinned RL Zoo factory source and existing local
  training artifacts; audit the closest CarRacing learned-planning comparator,
  Valdi. These are source/artifact reviews, not new training or reproduced scores.
- **RL Zoo finding:** the default reference has no action repetition and requests
  4M aggregate learner decisions. Our selected checkpoint has 251,904 decisions
  (6.30%); the stopped trial has 501,760 (12.54%). Repeat four also changes the
  physical discount/noise cadence. Source hashes, counter meanings and caveats are
  in [the baseline audit](BASELINE_AUDIT.md). More training helping is unproven.
- **Valdi finding:** numeric sensors are extracted from HUD pixels; they are not
  hidden-state access. Its supplied evaluation ignores done/truncated flags over
  600 steps. Planning uses 5,120 candidates per replan and a learned terminal value.
  Source revision, action mapping, seed/resume differences, memory requirements
  and paper-version uncertainty are in [the source audit](VALDI_AUDIT.md).
- **Assistant decision/reason:** keep current search as a mechanism study and
  retain Valdi as relevant prior work. Verify a competent local proposer and the
  native PPO recipe before an expensive new learned-model reproduction. No SOTA,
  convergence or general limitation of RL follows from the current training trials.
  High confidence in source/accounting observations; performance and repair
  usefulness remain uncertain. No production settings, ongoing race conditions,
  final-test access or user-only rank/regret restrictions changed.

## R48 — Align native recipe and measure cost before larger baseline training

- **Origin:** user asks to continue experiments and establish strong comparators;
  assistant chooses the audited native interface and a bounded execution profile.
- **Purpose:** distinguish interface/budget shortcomings from learned-control
  limitations before interpreting the paper's component interventions.
- **Choice:** native pixels only, resize64/grayscale/stack2, repeat one, standard
  NatureCNN512 and actor LSTM128. Existing local repeat-four experiments stay intact.
- **Evidence:** official pixel/reward parity, vector history/reset, CPU/subprocess,
  post-update checkpoint/resume and frozen normalization checks pass; 105 tests.
- **Declared next measurement:** four serial CPU/CUDA × dummy/subprocess cases,
  eight workers, seed 0, three 4096-step updates each. Select execution cost only,
  never short-run return. See [declaration](NATIVE_RECIPE_2026-10-29.md).
- **Uncertainty:** high confidence in accounting/interface correction, medium in
  short-profile throughput transfer, unknown driving competence. Full-budget and
  independent-seed runs, stronger local proposer and trained V/L/F remain open.

### R48 execution update: profile complete, fixed native campaign declared

The assistant implemented the separate native interface and ran all four declared
profiles. Both GPU cases completed training but failed GPU-versus-CPU numerical
assertions at reload. The correction separates exact same-device serialization
and independent CPU-load repeatability from reported cross-device drift. The two
GPU cases alone were rerun under new IDs; no learning-rule change or score-based
selection occurred. All 106 tests pass. CUDA/eight subprocess environments is the
cost-selected setup: 543 frames/s, provisional 2.05 training hours for 4M decisions.
All six attempts, costs, failure artifacts and source checks are retained in the
[profile report](NATIVE_RECIPE_2026-10-29.md).

The assistant now declares the [one-seed native baseline](NATIVE_BASELINE_2026-10-29.md):
4M aggregate decisions, global linear schedule, four scheduled checkpoints and
fixed first-50-development selection by official return/completion/earlier step.
Full-episode evaluator replay precedes launch. This supports the user's comparison
goal but does not establish policy competence, generality or SOTA. Native pixels
and local repeat-four search remain separate protocols. No private rankings or
final-test data are accessed. Main uncertainty: whether the full recipe learns
competent driving and how well short-profile runtime predicts a longer curve.

R48 launch update: exact two-track/full-episode replay passed (two passes; 4,000
additional development frames, four resets). The fixed `native_baseline_20261029_s0`
campaign is now running on CUDA/eight subprocess environments; first target 0.5M,
then fixed first-50-development evaluation. Early training updates are finite.
This is launch evidence, not a completed native driving result. Runtime remains
provisional; the source/selection budgets are frozen and technical failures stop
the runner without an automatic hyperparameter retry.

## R49 — Native baseline regression and bounded deployment control

- **Origin:** user asks to continue rigorous experiments; assistant audits the
  completed native baseline and proposes the inference-device control.
- **Measured:** 4,001,792 native training decisions; 200 development episodes;
  2.42 hours. Selected 1M: 843.82/38%; final 4M: 645.80/4%. All 200 traces and
  544 source-file hashes, resume parameter hashes and optimizer/normalizer/stream
  clocks pass. The selected native driver is separate from the local proposer.
- **Interpretation:** later native regression persists after the full declared
  budget. It does not prove RL inadequacy, a perception/control decomposition,
  SOTA, or any native simulation-search improvement. Approximate KL increases and
  the learned SDE scale falls, but these are associations rather than causes.
- **Next assistant decision:** same checkpoints, same first ten development tracks,
  CPU versus CUDA inference; six historical CPU replay episodes, 30 new GPU
  comparison episodes and six exact GPU replay episodes. Cap 42,000 frames;
  no training, checkpoint reselection or private/held-out access.
- **Why:** train-on-GPU/evaluate-on-CPU numerical differences may amplify in
  closed-loop driving. High confidence in the diagnostic's validity value;
  effect size unknown. Preserve the frozen CPU-based 1M selection regardless.
- **References/status:** [baseline results](NATIVE_BASELINE_RESULTS_2026-10-29.md),
  [device-audit declaration](NATIVE_DEVICE_AUDIT_2026-10-29.md). Independent seeds,
  uninterrupted-training comparison and a stronger matched-local proposer remain
  later choices; no automatic hyperparameter sweep.

R49 completed: six historical CPU and six same-CUDA episode replays match exactly.
On the first ten tracks, CPU/CUDA means are 836.51/818.25 at 1M, 480.55/479.76 at
2M, and 549.89/616.92 at 4M. Later regression persists on both devices; device
sensitivity is not negligible. Cost 39,214 native frames, 42 resets, 203.59 s;
no new training. Retain original CPU-based selection; no native sweep is activated.

## R50 — Isolate PPO discount under the existing local action cadence

- **Author:** assistant experiment choice supporting the user's stronger matched
  proposer and simulation-selection goal; not a user-mandated hyperparameter.
- **Rationale:** the factory audit identifies different physical discount under
  repeat four. Test gamma=0.99^4=0.96059601 per decision, while explicitly holding
  reward-normalization gamma at the existing 0.99. Existing defaults remain intact.
- **Design:** same local information, repeat four, encoder, seed 0, CPU4, GAE/SDE,
  optimizer and 8M-native-frame LR schedule; stop at the bounded 2M review. Four
  checkpoints on the same first 50 development tracks; historical original local
  checkpoint is the reference. Exact two-episode historical replay precedes launch.
- **Confidence/limits:** high confidence in discount arithmetic and isolation of
  the configured normalization timescale, low confidence in driving improvement.
  Within-repeat rewards, GAE/exploration cadence and native interface still differ
  from upstream. This is not full recipe parity or causal component adequacy.
- **Alternatives not bundled:** larger decision budget, independent training seeds,
  encoder changes, GAE/SDE changes, uninterrupted-training controls. No automatic
  extension beyond 2M or parameter sweep. A better checkpoint still needs frozen
  direct-versus-original-search tests, independent seeds and stronger comparisons.
- **Status:** [declaration](LOCAL_DISCOUNT_TRIAL_2026-10-29.md); implementation and
  108 tests pass; historical replay is the remaining launch gate.

R50 launch update: the historical two-episode local replay passed exactly (1,803
native frames, two reset ticks), and 108 tests pass. Resolved-config audit confirms
only effective PPO gamma changes; normalization stays 0.99. Campaign
`local_discount_20261029_s0` is running on CPU/four workers under the declared 2M
review cap; early updates are finite. No new local driving improvement is yet
established, and the original frozen proposer/search remains the retained method.

R50 completed: the discount trial finished 2,007,013 native training frames and all
200 development episodes in 24.75 minutes. Trial-selected 0.5M: 691.74/2%; original
selected local: 858.11/30%. Paired selected-policy change −166.37, worse on 48/50
tracks. Saved PPO/normalization gammas and 544 source hashes verify correctly.
The variant is not adopted and no extension is launched. Original local
proposer/search remains retained. Next assistant proposal is a read-only audit
of effective action distributions, saturation and update behavior before another
independent-seed baseline or alternate comparator; it is not yet running.

## R51 — Observe effective action distributions before another training change

- **Origin:** user requests continued work; assistant implements the previously
  proposed offline actor/update audit after unsuccessful training variants.
- **Question:** do logged gSDE scale reductions imply less action variation, and
  do clipping/overlap or mean-versus-sampled-action differences accompany regression?
- **Design:** all four original-local, discount-local and native checkpoints;
  first five recorded development episodes on own and fixed common histories.
  Replay recurrent memory from zero; only permitted actor inputs enter inference.
- **Controls:** own-history actions must match saved evaluation; first 16 states
  per episode also match the standard predict API exactly. Analytic clipped-normal
  moments are checked against independent numerical integration. Common histories
  condition each actor on the same observations, not its hypothetical future drive.
- **Budget:** zero new racing, training or simulator steps; at most 60,000 offline
  decisions and 61,920 actor forward examples. No hidden ranks or held-out access.
- **Confidence and limits:** high confidence in the need to measure actual marginal
  variance; unknown mechanism. gSDE temporal correlation and driving benefits are
  outside this observational audit. No new controller or parameter selection occurs.
- **Status:** [declaration](ACTION_DISTRIBUTION_AUDIT_2026-10-30.md); implementation
  passes numerical and synthetic recurrent-parity tests; full regression checks
  precede execution. A separately declared independent-seed baseline is a possible
  next experiment if no actionable implementation fault is found.

R51 completed: all saved own-history actions and standard-predict checks match
exactly; 828 frozen source hashes verify. Original local effective steering/gas
marginal SD increases from 1M to 2M despite the learned scale falling; native marginal
variation decreases. No shared exploration-collapse explanation is established.
Cost 55,750 offline decisions, 57,670 actor examples, 34.23 s; zero new racing/training.
Keep current policies/controllers. Full conditional/clipping tables and limits are
in the [completed audit](ACTION_DISTRIBUTION_AUDIT_2026-10-30.md).

## R52 — Replicate the retained local recipe on independent seeds

- **Origin:** assistant execution choice supporting the user's credible same-
  proposer simulated-selection paper; independent-seed replication was already
  proposed in the project roadmap. The user did not prescribe seed 1/2 or 2M.
- **Purpose:** quantify reproducibility across initialization/training streams
  after failure of discount/KL/learning-rate variants and no actor-replay fault.
- **Frozen design:** only training seed changes from original local; seeds 1/2 run
  serially from frozen source/configs. Each stops at 2M native review under the same
  global 8M LR schedule, with 0.2/0.5/1/2M checkpoints and fixed first 50 dev selection.
- **Budget:** at most 4,016,382 new native training frames, 400,000 development frames
  and 16 train-only reload frames, plus resets. Prior seed 0 is historical exploratory
  evidence; no seed is omitted because of its score and no automatic extension.
- **Uncertainty:** high confidence in replication's validity role, low confidence
  it creates a stronger driver. Later direct/search comparisons freeze standalone
  checkpoints first. This does not establish convergence, SOTA or five-seed rigor.
- **Status:** [declaration](LOCAL_REPLICATIONS_2026-10-30.md), full checks and frozen
  launch precede execution. Private ranking and final-test data remain sealed.

R52 launch update: all 113 tests and manifest/config/source validation pass. Parent
`local_replications_20261030` is running seed 1, with seed 2 queued serially.
Both arms' production code, scripts and configurations were copied before launch.
No results are available yet. Each stops at its declared 2M-native-frame review;
no parameter changes, private ranks or final-test access are involved.

## R53 — Execute paper comparisons, retain all outcomes, and state realistic timing

- **User steering:** run the paper experiments and be realistic about readiness.
- **Assistant execution:** queue the full 200-development-track comparison for
  original seed 0 plus standalone-selected seeds 1/2. Ten conditions per policy:
  direct, transferred fixed correction, random7 and random21 under three planning
  streams each, original search7 and search21. No new hyperparameter selection.
- **Reuse/gates:** preserve four existing seed 0 full panels after exact two-episode
  parity; extend random controls with prefix checks; fresh direct panels must match
  the selected checkpoint's first 50 dev episodes. Queue waits for all training.
- **New caps:**5,208,000 native evaluation frames and 225,120,000 internal physics
  frames; zero training during comparisons, private/held-out access prohibited.
- **Analysis intent:** show every seed and method, paired ordinary driving
  outcomes, random controls, cost and latency. Actor selection uses standalone
  driving only. A weak or harmful-search seed is retained, not omitted.
- **Limits/timing:** exploratory three-actor data with historical/adaptive seed 0;
  not the complete paper or a final test. Queue runtime allowance 12–18h after
  training, inferred from measured 5.46h one-actor broad comparison. Core data
  target 1–2 days; full paper package planning 2–4 weeks has low confidence until
  stronger competitors and component-study budgets are fixed. Optional evolution,
  custom/GPU simulator and decision-space ideas remain preserved future work.
- **Status:** [paper ledger and declaration](PAPER_CAMPAIGN_2026-10-30.md); executable
  queue prepared with immutable source, method conditions and budget before data.

R53 technical preparation correction: initial `paper_search_20261030` queue stopped
while waiting, with zero new environment frames, after detecting reference-versus-
sparse renderer mismatch. Replacement `paper_search_sparse_20261030` explicitly
matches the historical broad renderer. Source/configs are re-frozen before driving;
replication training is unaffected. Original queue and stop reason are retained.

R53 corrected launch: all 114 tests and frozen declaration checks pass. Queue
`paper_search_sparse_20261030` is armed and waiting for both training seeds, then
runs the full fixed matrix and publishes tables/paired analyses automatically.
Actor selection is standalone only; every seed and condition is retained.
The [paper ledger](PAPER_CAMPAIGN_2026-10-30.md) explicitly lists unimplemented
required studies and remaining claim gates so this queue is not mistaken for
full paper completion. Project measurement jobs are serialized as declared.

## R54 — Do we have the strongest baselines and a SOTA-level research design?

- **User question/intent:** compare against the best relevant methods, use current
  research/deep reasoning, and be candid about prospects for a serious paper.
- **Assistant answer:** not yet. PPO adaptations and careful mechanism controls
  are useful but do not establish strong external-comparator coverage or novelty.
- **New primary review:** earlier roster missed MRS.Q/ICML 2026, directly concerning
  accurate-model search failure; it is now a close prior-work/comparison priority.
  APC-RL/ICLR 2026 adds a recent CarRacing protocol audit target. Existing POPLIN
  and physics-simulator MPPI prevent broad pipeline-as-novelty claims.
- **Protocol finding:** APC's author Appendix B.3 uses a fixed track, simulator-
  derived waypoints and demonstrations, so its scores are not copied into our
  local changing-track table. Named modern algorithms are not verified champions
  of our task. Source/code availability is distinct from a completed reproduction.
- **Execution priority:** independent competent SAC/MR.Q, same-engine MPPI/CEM,
  and modern MR.Q/MRS.Q readiness, then justified native visual comparisons.
  Current queue/source/selection is preserved; no competing training is launched.
- **Status:** [review and claim gates](COMPARATOR_REVIEW_2026-10-30.md), primary
  links and cached author-PDF provenance. New GitHub commit-pinning attempt timed
  out; no pinned audit, third-party execution or score reproduction is claimed.
- **Limits:** tests, seeds and clean splits do not supply novelty; broad RL
  inferiority, component adequacy and SOTA remain hypotheses requiring stronger
  evidence. Be willing to narrow claims or accept a negative result.

R52/R54 observed completion update: selected seed 1 mean 869.23/32%, seed 2 −17.89/0%.
Both reached the 2M-native-frame review with finite optimization and all four
scheduled evaluations, but seed 2 failed task competence. It is retained in the
ongoing paper comparison. The baseline gap is therefore empirical, not just a
missing algorithm name. Tests and implementation audits cannot substitute for
a competent independently credible actor. All external-comparator priorities
remain separate from the current immutable mechanism queue.

## R55 — Pin modern comparisons and resolve concrete adapter constraints

- **User steering:** continue the serious-paper work after the candid comparator
  review; no new algorithm or result is prescribed by this instruction.
- **Assistant execution:** resolve the earlier fetch failure, pin official MR.Q
  and MRS.Q revisions, and cache sixteen selected source/license files without
  importing their code. Pin two MPPI references and cache nine selected files.
  All 25 hashes and fourteen Python AST parses pass; no new simulator frames,
  dependencies, model execution or training are involved in this preparation.
- **Source findings:** symmetric external actions need an explicit gas/brake
  mapping; upstream Gym construction does not enforce our partitions; history
  differs from recurrent PPO. Source storage helpers and the MRS.Q pixel path need
  explicit resolutions. MRS.Q retains stochastic planning during evaluation and
  its effective RL target horizon differs from its named configuration field.
  These are source-derived constraints, not measured runtime failures here.
- **Resource intent:** default vector replay could consume 14.908 GB for the road
  plus ego/action observation alone. Compression and alternate encoders remain
  proposed. Default MRS.Q makes 9,264 latent transition examples per plan; those
  cannot be equated to Box2D ticks. Profile real update bursts before promising
  budgets or classifying a baseline as competent.
- **Planner choice and rationale:** use the separately pinned MIT MPPI reference
  for implementation readiness; the inspected author repository lacks an explicit
  license. Keep canonical action-space MPPI distinct from a feedback-policy
  residual adaptation. A weighted synthesized action needs its own simulated
  transition before committing our vehicle estimator, and that work is charged.
- **Uncertainty:** high confidence in provenance/accounting needs; unknown encoder,
  planning settings, performance and runtime. No external score, reproducibility
  claim or strongest-baseline gate is closed by this audit.
- **Execution boundary:** the frozen three-policy queue remains active and
  unchanged. Implementation/runtime profiles follow its measurements serially;
  no competing training or profiling, private ranks or held-out access is added.
- **Records:** [MR.Q/MRS.Q source readiness](MRQ_MRSQ_AUDIT_2026-10-30.md),
  [MPPI readiness](MPPI_READINESS_2026-10-30.md) and their public hash records.

## R56 — Implement the external boundary and verify official code on synthetic data

- **User steering:** continue autonomously toward the serious-paper comparison
  plan. The user does not prescribe a new encoder, algorithm repair or run budget.
- **Assistant implementation:** restricted local float32 vector/action codec,
  infrastructure-only environment wrapper, isolated RNG scope and hash-verified
  separate source staging. The actor receives only arrays; train/development
  remain the allowed partitions. This is history-one preflight, not native pixels
  or a recurrence-matched strong-policy claim.
- **Observed failure:** the first official MR.Q synthetic learner call failed.
  A minimal source check shows deterministic PyTorch fills unused `torch.empty`
  priorities with NaNs, which enter the masked cumulative sampling distribution.
  Preserve `runs/external_synthetic/mrq_20261030` and its failure unchanged.
- **Explicit correction:** initialize unused MR.Q priorities to zero in a new
  stage; preserve filled-slot priority formulas and record the changed source.
  Both algorithms have an explicit CPU observation-storage patch. Undefined
  upstream memory behavior is not claimed equivalent to the corrected version.
- **Narrow evidence:** corrected MR.Q and CPU-staged MRS.Q each pass one learner
  call on sixteen synthetic transitions with reduced declared dimensions, finite
  and changed encoder weights, fixed-RNG action repeat, and exact CPU save/reload.
  Zero driving environments or internal physics frames. GPU, pixel, ring-wrap,
  exact continuation and driving competence remain unvalidated.
- **Software checks:** first suite 132 passed/one CUDA-only skip; after replay
  correction and late-CUDA-init guard, full CPU suite 133 passed/one skip. Keep
  both check records. Late CUDA initialization fails rather than silently leaving
  an untracked GPU random stream.
- **Scheduling choice/revision:** permit bounded software checks on one CPU at
  nice19 with CUDA hidden while the frozen queue runs, rather than restricting
  all work to source reading. Record their overlap with seed1 search7 and keep all
  raw latency. Require separate clean timing before an efficiency claim. This
  does not authorize competing driving training/profiles or change queued methods.
- **Next gate:** complete training/checkpoint/evaluation integration and replay
  boundary/ring-wrap/resume tests, then declare a serial measured driving-readiness
  profile after the current queue. Do not promote an actor as strong from synthetic
  software checks. No private diagnostic or final-test access is added.
- **Record:** [implementation and limits](EXTERNAL_INTERFACE_2026-10-30.md), with
  source, failed/successful check configurations, provenance and timing windows.

## R57 — ETA check discovers disk exhaustion; preserve and recover the fixed study

- **User request:** give the current ETA. The assistant checks processes and
  finds both original PIDs dead despite the stale status file saying evaluating.
- **Observed cause:** root filesystem full while saving the next episode NPZ;
  even failure/status writes were interrupted. Eighteen full condition panels and
  58 complete seed1 search7 episodes are saved. No performance-based omission.
- **Assistant recovery:** prune unreachable package cache without deleting data;
  usable root space remains inadequate. Use the existing mounted secondary volume
  with about 16 GB free. Preserve original artifacts, declaration, methods, selected
  actors, all 69 source hashes and frozen analysis; verify completed-panel hashes
  and development identities before reuse. New run directories have recovery suffixes.
- **Validity/cost:** rerun interrupted search7 fully and require exact equality
  for its 58 saved prefixes. Charge a conservative failed-attempt upper bound of
  56,488 native and 7,908,320 internal frames; declare extra recovery allowance
  rather than silently discarding compute. Check free space during each run.
- **Checks/launch:** recovery preflight passed; full CPU suite 133 passed/one
  CUDA-only skip, with temporary storage on the secondary volume and no concurrent
  driving. Restarted at 2026-10-30 15:15:55 UTC; processes and heartbeat verified.
- **ETA:** about 12–15 hours after restart for this batch, from measured search7
  pace and the historical search21 ratio, including the full interrupted-condition
  rerun. This does not estimate completion of every remaining paper experiment.
- **Record:** [recovery declaration, storage, checks and limits](CAMPAIGN_RECOVERY_2026-10-30.md).
  The original status JSON is historical; use the new live status path. No private
  diagnostic, final-test access or new training method is involved.

## R58 — Why can simulated selection worsen official return?

- **User question:** explain the negative search7 return on the newly trained actor
  and how search can damage behavior despite retaining the base candidate.
- **Assistant analysis:** paired completed 200-track direct/search7 outcomes,
  official reward code and allowed executed observations; no new environment work
  or private candidate outcomes. This is post-result descriptive analysis.
- **Measured accounting:** return change −22.5561 = −22.4391 from fewer distinct
  tiles plus −0.1170 from extra native steps. Zero playfield exits in both panels;
  the identity holds on every episode. Wall-clock planning is not charged here.
- **Failure heterogeneity:** 29 gained and 13 lost completions, but much poorer
  coverage in unfinished runs. One illustrated timeout stalls on-road for about
  half its episode; another keeps moving off-road. No single mechanism follows
  from a completion percentage or an aggregate braking/speed association.
- **Structural explanation:** the base candidate protects modeled short-horizon
  proxy cost, not actual full-episode return. The score, terminal treatment,
  estimated contact/state, tangent road and imagined-policy responses can misrank
  behavior. Reusing mechanics code is not exact-state/world cloning.
- **Limits/next gate:** causes remain hypotheses until controlled substitutions.
  A uniform time penalty cancels across equal-length rollouts; more candidates
  are not guaranteed to fix model/score errors. Preserve all current settings and
  finish the frozen matrix before another declared development intervention.
- **Record:** [diagnosis and public paired artifacts](SEED1_SEARCH7_DIAGNOSIS_2026-10-30.md).

## R59 — Align search with actual driving and keep the paper truthful

- **User request:** change search to make it useful for the final driving goal,
  while retaining scientific honesty. No specific score, alpha or trial count is
  prescribed by the user.
- **Assistant design:** a bounded score-only intervention. Compare original
  nonprogress penalty strength1, quarter strength0.25 and zero strength, holding
  progress term, programmed model/state/road estimate, proposer, candidates,
  renderer, horizon, control rate and random stream fixed. This remains a proxy,
  not an exact official reward model or a guaranteed fix.
- **Validity:** new explicit `policy-objective` method; old defaults/campaign are
  untouched. Direct parity and twenty-track alpha1 parity gate variants. Official
  return is primary; completion/coverage/departures/stall-speed estimate co-reported.
  Retain every actor/arm, negative outcomes and development costs. No oracle/private
  rank feedback, official environment change or final-test access.
- **Scope and selection:** two useful existing actors on the repeatedly used
  first20 development tracks, three search settings each. Seed2 remains in the
  original full matrix; this narrow pilot is not headline seed robustness. A
  setting can be screened for broader validation only if point return beats
  direct and original search for both actors; no automatic promotion/expansion.
- **Costs/schedule:** at most124,000 new native frames and16,800,000 internal physics
  ticks, including direct parity; zero new training. Freeze all code/configs first,
  wait for successful current-matrix completion, then run serially on the secondary
  volume. Approximate allowance1–1.5h after dependency completion.
- **Checks:** full CPU suite140 passed/one CUDA-only skip; same-rollout score
  isolation, exact legacy-cost/action behavior at alpha1, recurrence/state-commit
  integrity, CLI separation and negative/missing-arm reporting tested. Software
  checks overlap old search21; timing caveat is recorded rather than hidden.
- **Prior work/uncertainty:** MRS.Q and MPC already establish search/model/utility
  difficulties; a weight change alone is not claimed novel. High confidence in
  interpretability of this one-axis test, low confidence in an improvement before
  driving. More complete progress/continuation objectives remain future tests.
- **Record:** [design, declaration and evidence limits](OBJECTIVE_ALIGNMENT_2026-10-30.md).

## R60 — October31 readiness/ETA check and scoped sleep prevention

- **User request:** are we ready to continue, what remains to run, and what is
  the ETA? The assistant verifies both controllers/processes, live output and storage.
- **Observed status:** main matrix19/30 complete, seed1 search21 near200 tracks;
  all ten seed2 conditions remain. Objective pilot is alive and waiting, with no
  pilot driving frames. No manual restart is needed.
- **Verified interruption:** systemd-suspend logs show sleep from October30
  23:41 to October31 10:50 Moscow, about11h09m; monotonic/calendar elapsed clocks
  corroborate it. Earlier ETA assumed continuous operation. Completed data,
  method settings, splits and failed outcomes are retained.
- **Assistant operational choice:** a temporary sleep/idle inhibitor monitors
  only the two technical status files and releases when they finish/fail/exit.
  No permanent power setting or scientific source is changed. Command, PID and
  helper hash are recorded; no private diagnostic or held-out data is accessed.
- **Revised estimate:** roughly6–8h awake for remaining main conditions, followed
  by1–1.5h objective pilot; combined target this evening, approximately18–21 Moscow.
  This is conditional on continued operation and ordinary timing variability.
- **Remaining paper gates:** actual strong external baselines, controlled
  component comparisons, adequate independent seeds and final freeze/evaluation.
  Current-job completion is not full-paper readiness; no firm paper ETA is claimed.
- **Record:** [progress, source evidence and limits](PROGRESS_2026-10-31.md).

R60 completion update: seed1 search21 finished during the check at845.37/54%,
versus direct865.42/30.5%; return change −20.05. The queue advanced into seed2
conditions. The completed-condition hashes/costs are recorded; no partial score
was promoted. The objective pilot's numerical settings and analysis remain as
frozen before this result.


## R61 — Completed replication and unsuccessful score-only repair

- **User intent:** report actual results, improve official driving return and
  distinguish understanding from deciding without inventing a positive outcome.
- **Measured:** all30 main conditions complete. Search21 gains22.06 on historical
  actor0 but loses20.05 on fresh useful actor1. Actor2 gains6.65 but remains failed
  at−13.43/0% completion. Across-seed mean2.89 has descriptive95% interval
  [−21.10,23.69]; no robust improvement is established.
- **Assistant intervention, now measured:** the previously declared quarter/zero
  penalty pilot finished all arms on20 reused development tracks. Quarter-scale
  point gains over original search are13.39/4.57, both intervals crossing zero;
  seed1 still loses14.58 versus direct. Zero penalties are worse. Neither setting
  passes the requirement to beat both controls on both actors; no promotion or
  expansion occurs. User's usefulness objective and assistant's experimental
  implementation remain separately attributed; prior planning work is not novel.
- **Integrity:** every saved condition's metrics/config hashes verified, report
  copies byte-identical, all failed/negative outcomes retained. No private or
  held-out feedback used. Direct and alpha1 parity gates passed. Pilot charges
  115,198 native frames/15,631,840 internal physics ticks and zero training.
- **Next choice and uncertainty:** complete competent independent baseline
  integration and declare a bounded diagnosis distinguishing score/model/proposal
  limitations before another method run. Aggregate results do not isolate cause;
  do not initiate an unrestricted weight sweep or final evaluation.
- **Operational status:** both controllers complete; temporary sleep guard released.
  [Results, full reports and limitations](RESULTS_2026-10-31.md).
