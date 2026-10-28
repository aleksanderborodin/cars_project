# Research questions, ideas, intent and probes

Updated 2026-10-28. [Index](../README.md) · [User intent](USER_INTENT.md) ·
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
