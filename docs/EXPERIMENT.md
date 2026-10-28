# Experiment design: modular prediction and control by search

Updated 2026-10-29 following the research discussion. The project now targets a
scientific paper about modular world models and online control, using Gymnasium CarRacing
only and the current workstation. Physical robots and other simulators are out
of scope. Restricted local/native interfaces, programmed physics and deterministic
search now exist; see [STATUS.md](STATUS.md) and the
[latest completed development report](BROAD_RESULTS_2026-10-29.md).
The [frozen 200-track development campaign](BROAD_DEVELOPMENT_2026-10-28.md) and
lower-learning-rate probe are complete. The broader learned road distributions,
scenario planning, modern comparisons and final campaign below remain proposed.

The [paper blueprint](PAPER_BLUEPRINT.md) consolidates the user's three-stage
perception/forecasting/action-selection study, attributed ideas, critiques and
comparison design. It proposes oracle-current and oracle-future training regimes;
these are not implemented or authorized to bypass existing information boundaries.
Read [USER_INTENT.md](USER_INTENT.md) for the user's decisions and paper goal.
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) details the geometry branch;
[PREDICTION_SELECTION.md](PREDICTION_SELECTION.md) reviews optional predictor
spaces and downstream model selection. Proposed numerical defaults,
confidence, and alternatives are recorded in [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md);
they remain development choices until frozen. The previous experiment
is preserved on the [archive branch](ARCHIVE.md). Retained legacy code and
historical artifacts keep their earlier protocol and interpretation; the new
restricted research path is separate.
The [run-readiness plan](RUN_READINESS.md) specifies what to freeze before data
collection, what to select in pilots, and what must pass before expensive runs.
The [programmed-simulator plan](SIMULATOR_SEARCH_PLAN.md) is the active architecture
following the user's correction: programmed physics is the main world model;
candidate behaviors are selected by simulated outcomes.
The [next-moves plan](NEXT_MOVES.md) refines its remaining execution stages and
records the proposed stronger-proposer experiments, budgets and decision gates.
The earlier
[next-experiment plan](NEXT_EXPERIMENTS.md) records the preceding work. Its first batch is complete; the
[control audit](CONTROL_AUDIT_2026-10-28.md) reports the correction, matched fixed
controls and failure evidence. Further diagnostics and baseline training remain open.

## Research question

Separate **understanding the situation** from **deciding how to act**. Distinguish current-scene perception, future prediction and action selection. Which
road/state/prediction components are already sufficient for the tested task and
which limit performance? Does simulation-based selection among a frozen proposer's
sampled behavior variants improve driving over executing that proposer directly?
These are the user's central questions, not a requirement to make every internal
prediction physically faithful. Use controlled component substitutions and compare
strong learned policies and established planners under matched information,
supervision and budgets. Component adequacy is conditional on this setting; no
whole subproblem is declared universally solved.

The main loop is observe, reconstruct/estimate, simulate candidates in a coded
world, choose, execute briefly, and replan. The main dynamics engine reuses
programmed vehicle mechanics in independent reconstructed worlds. Learned
transition models are comparison alternatives, not the default main engine.
A behavior proposer may be engineered or learned; the central comparison is
the same frozen proposer with and without simulation-based selection.
Hidden-road reconstruction is one component/controlled study, not
the sole task. Existing perception/proposal models may be reused. An optional study trains
predictors normally, freezes them, trains/configures downstream control, and
selects prediction spaces/hyperparameters/checkpoints by validation driving.
It does not require driving-loss gradients through the predictor.
The user's proposed future [information/control study](INFORMATION_AND_CONTROL_STUDY.md)
crosses local/full-map inputs with learned/search control and separates road,
vehicle-prediction and decision errors. It remains proposed; current global
diagnostics do not constitute a full-information learned-policy comparison.

The user additionally proposes a [task-useful programmed simulator](DECISION_USEFUL_SIMULATOR.md),
possibly independently implemented, GPU-accelerated or adapted through an agentic
loop. **Realism is optional in that future variant; actual driving usefulness is
the selection objective.** Prediction accuracy remains a diagnostic. This is a
separate future model-construction experiment, not an active evolution campaign
or a change to current frozen runs or normal predictor-training defaults.

Modularity, world models, and MPC are established ideas; no novelty, SOTA, or
safety claim is established. Search explores a finite set/parameterization of
behaviors; it does not enumerate every continuous action or guarantee the best
real-world action. Agentic evolution remains future work, not a prerequisite.

## Environment and information contracts

- Python 3.12; use `uv run`, and `uv add` for dependencies.
- Standard tracks retain official CarRacing-v3 reward, termination, and physics:
  continuous three-component actions, 50 Hz physics, 1000-frame TimeLimit,
  `lap_complete_percent=0.95`.
- Observation wrappers may restrict information. Every modified observation
  protocol has its own version and result table.
- Custom diagnostic geometries and perturbations are separately identified;
  their results cannot be reported as standard CarRacing scores. The existing
  hard generator/split remains frozen.
- Setting A stays state-only without rendering. A separate visual setting will
  explicitly enable rendering.
- Deterministic episodes must reproduce within the documented software/hardware
  configuration.

| Protocol | Policy inputs | Purpose |
|---|---|---|
| Global geometry | Full track and declared ego state | Privileged reference, not an equal-information competitor |
| Local geometry | Visible car-frame road boundaries/mask, declared ego sensors, permitted history | Isolate forecasting and planning |
| Local vision | Fixed-view local RGB/mask and declared ego sensors/history | Complete modular system |
| Native pixels | Unmodified official pixel observations | Separate external-comparability experiment for any standard-CarRacing claim |

Local policies must never receive `TrackInfo`, track identity, global tile index,
hidden geometry-derived features, seeds, or the environment object. Truth is
available to the evaluator for scoring. Memory resets each episode and may
accumulate only observed information. The current `Policy.reset(track)` and
StateObs v1 are privileged interfaces; their roughly 140 m future-road view is
not a limited-view sensor. The native renderer also has an initial zoom;
strict local vision needs a fixed observation construction from its first frame.

## Supervision and planning access

Geometry supervision can use training/development environments only. Count
labels and collection frames. Include controls receiving identical auxiliary
labels or the same frozen reconstructed representation. Distinguish privileged
supervision from observation-only learning in every comparison.

Deployment planning uses programmed mechanics with a state/road estimate built
from permitted observations and priors. Knowing the mechanics does not authorize
access to true wheel state or full road. Copying the evaluation environment also copies its hidden road and
is forbidden for local methods. Any simulator oracle is a separate diagnostic.

Sampled road geometry persists throughout an imagined rollout. Evaluate actions
across plausible roads; never select the road that makes an action look best.
Future actions may depend on future simulated observations, but not on hidden
sample identity. Separate uncertainty in geometry from uncertainty in dynamics.

## Methods and scope

Keep the expert, BC/DAgger, PPO/SAC, logging, and evaluation as useful
infrastructure. Retire Rust/Codex evolution and the complete track-count K-grid
to the archive branch. Historical runs remain preserved locally; they do not
establish results under the new protocol.

Build the programmed simulation/control system first. Prioritize local pure pursuit
with a visibility/braking rule; coded-physics local MPC and same-proposer selection
controls; scenario MPC with a simple road prior; matched planners with learned
single/multiple road forecasts; a strong recurrent model-free controller;
TD-MPC2 with documented history adaptation; and DreamerV3 for visual comparison.
Audit and attempt the directly relevant Valdi CarRacing reproduction. PETS-style
dynamics ensembles and MR.Q are additional comparison candidates. Adaptations
are labeled as adaptations, with original-protocol replication kept separate.

Include actor-free planning references (PETS/PlaNet-style methods and Valdi), a
learned controller using the same representation, and direct candidate scoring
when claiming that explicit simulation adds value. Compare controllers with a
fixed predictor and predictors with fixed control/search to attribute gains.
Keep the expert/full-map experiment as a privileged diagnostic or declared
teacher. Rust/agentic evolution can be reconsidered as a separate later controller
optimization route; its code remains archived now.

## Splits, fairness, and evaluation

- Never inspect or use `unseen` or `hard` seeds/results during method design.
  This planning task does not authorize a held-out evaluation campaign.
- Before training, implement disjoint training, development, calibration, and
  test manifests. Development/calibration use permitted seen-side data. Audit
  overlap against historical training records; `seen-inf` needs exclusions.
- Freeze the environment/information contract and partition mechanism before
  paper data collection. Train/dev/test use disjoint tracks from the same
  generator. Keep track, learning and planning random streams separate; log the
  actual training tracks and reserve final identities without inspecting them.
- Split supervised datasets by whole track; related diagnostic variants remain
  in one partition. Do not randomly split adjacent frames.
- Select checkpoints, budgets, sensor ranges, risk settings, and ablations on
  development data. Freeze models and the complete condition manifest before
  one final evaluation campaign per model.
- Use five independent training seeds for headline learned methods. Small pilots
  and three-seed exploratory ablations are explicitly preliminary.
- Match sensors, history, action space, and decision frequency within tables.
  Report native frames separately from decisions, internal model transitions,
  labels, expert actions, training/tuning compute, and evaluation compute.
- Retain official return, success, progress, completed-lap time, playfield exits,
  off-road events, and off-road frames. Off-road events are road departures,
  not physical collisions. Document the current strict-before-1000 success rule.
- Report lap time together with completion; never drop failures from the main
  performance comparison. Use common evaluation tracks, per-training-seed
  results, means/std, IQM, and 95% intervals with independent units stated.
- Extend reporting to group by protocol, supervision, budget, and condition.
  Episodes from one model are not independent training runs. Rare-failure
  claims need enough independent test tracks per model.
- Add forecast proper scores/coverage, curvature-event calibration, dynamics
  rollout errors, action-risk calibration, latency quantiles, and deadline misses.

## Implementation order

1. Information boundaries, partition manifests, and evaluation metadata/tests.
2. Programmed internal simulator with observation-based state reconstruction,
   same-proposer control/search, expert/global diagnostics and native pilots.
3. Complete-system comparison with strong learned policies and planning baselines.
4. Focused model/planner ablations; optionally select normally trained predictor
   spaces and hyperparameters by downstream validation driving.
5. Complete visual/native experiments and resolve identified component failures.
6. Freeze, run final evaluation, and write evidence-supported claims.

Revise the algorithm claim if gains depend on extra information or disappear
against competitive baselines at matched cost. A simple prior matching a learned
road forecast retires that component claim, not automatically the whole system.
Likewise, no hidden-road information gap changes the geometry experiment rather
than making all model-based driving research invalid.

All runs retain immutable `runs/<method>/<run_id>/config.json`, models, cost
records, and episode logs. Add protocol versions, partition hashes, supervision,
sensor/history settings, and planning budgets. Preserve failures. Finish code
changes with `uv run pytest -q`.

The optional privileged candidate-ranking monitor is **user-only**. Its real
outputs must never be read by agents or consumed by training, validation selection,
tuning, or assistant method analysis. Only technical status/costs and synthetic
tests are inspectable. See [RESEARCH_LOG.md](RESEARCH_LOG.md), R4–R5.
