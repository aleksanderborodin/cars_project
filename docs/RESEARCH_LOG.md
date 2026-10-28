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
