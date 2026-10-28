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
