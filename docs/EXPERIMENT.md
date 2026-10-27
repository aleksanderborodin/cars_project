# Experiment design: planning with uncertain road geometry

Updated 2026-10-27 following the research discussion. The project now targets a
scientific paper about partially observed control, using Gymnasium CarRacing
only and the current workstation. Physical robots and other simulators are out
of scope. The new protocols, models, planners, and experiments below are
**proposed, not implemented**.

Read [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the literature review, novelty
assessment, method, comparisons, and execution gates. Proposed numerical defaults,
confidence, and alternatives are recorded in [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md);
they remain development choices until frozen. The previous experiment
is preserved on the [archive branch](ARCHIVE.md). Existing code and artifacts
still implement that earlier protocol and retain their original interpretation.

## Research question

When does an explicit distribution over unobserved road geometry improve
closed-loop driving relative to deterministic reconstruction, simple statistical
priors, and recurrent learned controllers, under matched information, supervision,
interaction budgets, and inference cost?

Separate visible-road reconstruction, belief over unseen geometry, vehicle
dynamics, and planning. Investigate decision-relevant uncertainty and planning
that respects when new geometry becomes observable. Modularity, world models,
and uncertainty-aware MPC are established ideas; no novelty, SOTA, or safety
claim is established for this project yet.

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

Deployment planning uses an internal model built from permitted observations
and priors. Copying the evaluation environment also copies its hidden road and
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

Prioritize local pure pursuit with a visibility/braking rule; deterministic
local MPC; scenario MPC with a simple road prior; matched planners with learned
single/multiple road forecasts; a strong recurrent model-free controller;
TD-MPC2 with documented history adaptation; and DreamerV3 for visual comparison.
Audit and attempt the directly relevant Valdi CarRacing reproduction. PETS-style
dynamics ensembles and MR.Q are additional comparison candidates. Adaptations
are labeled as adaptations, with original-protocol replication kept separate.

## Splits, fairness, and evaluation

- Never inspect or use `unseen` or `hard` seeds/results during method design.
  This planning task does not authorize a held-out evaluation campaign.
- Before training, implement disjoint training, development, calibration, and
  test manifests. Development/calibration use permitted seen-side data. Audit
  overlap against historical training records; `seen-inf` needs exclusions.
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

1. Information boundary, partition manifests, and evaluation metadata/tests.
2. Deterministic local control, simple prior, and global diagnostic reference.
3. Compact road-distribution model with a matched-planner comparison.
4. Calibration, future-observation-aware planning, and strong learned baselines.
5. Visual reconstruction after the local-geometry hypothesis survives.
6. Freeze, run final evaluation, and write evidence-supported claims.

Revise or stop the algorithm claim if a simple speed rule matches it, a learned
prior does not improve decisions, gains depend on extra information, or the
advantage disappears against competitive baselines at matched cost.

All runs retain immutable `runs/<method>/<run_id>/config.json`, models, cost
records, and episode logs. Add protocol versions, partition hashes, supervision,
sensor/history settings, and planning budgets. Preserve failures. Finish code
changes with `uv run pytest -q`.
