# Active design: programmed simulation improves proposed behaviors

Updated 2026-10-28 after the user's explicit correction. This is the active
architecture plan. [User intent](USER_INTENT.md) · [Specification](EXPERIMENT.md) ·
[Earlier diagnostic results](CONTROL_AUDIT_2026-10-28.md)

P0 and the first P1 follow-up are complete. The
[programmed-simulator report](PROGRAMMED_SIMULATOR_RESULTS.md) records results,
costs and the proposed next diagnostic. Neither campaign establishes a return win.
The [next-moves plan](NEXT_MOVES.md) now gives the detailed execution order and
budgets for the remaining P1–P3 work, especially stronger frozen-policy comparisons.
Its full-map diagnostic and first 2M driver block are complete. The selected 1M
checkpoint is frozen for a running search comparison after later training regressed;
see the [current execution report](GEOMETRY_AND_DRIVER_2026-10-28.md).

## User intent and the central experiment

The world model should be an algorithmic simulator written in code. Candidate
behaviors may come from a controller or a learned proposal model. Simulate their
consequences and choose by the simulated outcomes; a newly trained behavior-value
model is not required to make that choice. The core experiment is a frozen
behavior proposer alone versus the same proposer improved by simulation.

This supersedes using learned ridge/neural transition dynamics as our main
engine. The ridge pilots remain useful historical controls. Normal training of
optional perception, road hypotheses or proposal models is separate from using
simulation for the decision. Do not silently turn the main project back into
learning a transition network or a learned critic.

The user also authorized autonomous execution and asked us to record uncertain
choices for later discussion. The choices below are assistant implementation
decisions, not additional user commitments or established scientific findings.

## Architecture

1. **Observe and estimate:** construct a local road map and car-state estimate
   from permitted sensor/history inputs. Retain uncertainty where observation
   cannot identify the hidden state.
2. **Construct internal worlds:** use programmed vehicle/road mechanics with
   observed geometry plus declared hypotheses for unobserved geometry/state.
3. **Propose behaviors:** retain the original proposer behavior and add varied
   maneuvers, initially from simple feedback controllers. A frozen recurrent
   policy can later supply proposals and becomes an explicit paired baseline.
4. **Simulate and score:** run each candidate in the same set of internal worlds.
   Score the outcomes with a declared coded task utility. The simulator executes
   the consequences; the proposal network's own value estimate does not select
   the winner in the primary variant.
5. **Execute and replan:** apply the selected first control for four native
   frames, incorporate the new observation, and repeat.

For multiple equally weighted sampled worlds, the proposed estimator is
`J_hat(behavior) = mean_k utility(simulate(world_k, behavior))`. For a weighted
scenario set, use its declared weights. Choose the behavior with the largest
estimated expected utility, not the world that makes it look best. A single
assumed world is a deterministic special case, not a calibrated distribution.
More particles reduce sampling error, not bias from incorrect world assumptions.

The distinction between a **world model** and a **behavior proposer** matters:
we compare behaviors under the same world assumptions. Selecting a world model
because it predicts the most optimistic driving outcome is not valid selection.

## Programmed engine and access boundary

Reuse the installed, locked Gymnasium `Car` vehicle implementation and Box2D
solver. The [upstream vehicle source](https://github.com/Farama-Foundation/Gymnasium/blob/v1.4.0/gymnasium/envs/box2d/car_dynamics.py)
implements wheel dynamics, steering joints and tire/road forces. The initial
backend invokes those equations at 50 Hz with the official solver iteration
settings. It creates independent worlds from observations; it never clones the
evaluation environment or reads its hidden road, seed, wheel state or contacts.

Knowing the physical equations does not give the agent the hidden initial state.
Wheel rotation, steering-joint state, contact geometry and numerical solver state
affect rollouts. With the current local sensors those must be estimated, not read
from the simulator. Full-map geometry diagnostics remain separately privileged;
they must not silently receive extra vehicle-state truth.

Version 1 uses official vehicle equations but approximate reconstructed contact:
a wheel is on road if any of its four vertices or center has road occupancy at
least one half. This can miss/alter geometric intersection near boundaries. It
is not an exact reproduction of the full CarRacing world. A later polygon-fixture
backend is the reference if this approximation limits decisions.

Version 1 carries the chosen candidate's **first** simulated transition's wheel
state into the next decision, then corrects hull velocity/yaw using the actual
permitted observation. It does not carry the end-of-horizon state. New candidate
worlds reconstruct joints and discard solver warm-start impulses. No parameters
are fitted and no extra sensors are introduced by this observer.

## Utility and uncertainty

The first engine comparison keeps the existing progress/clearance/center/slip/
unknown-space score. This isolates the simulator substitution before changing
the objective. That score is a coded heuristic; it is not the official reward.

The [official reward](https://gymnasium.farama.org/environments/box2d/car_racing/)
depends on visited tiles and the complete track's tile count. A reconstructed
local world cannot silently receive that hidden bookkeeping. The later coded
utility should prioritize new forward road coverage and completion/pace, account
for visited local history, and penalize departures and wasted time. Its precise
normalization and weights need declared development selection. Actual experiments
always evaluate unchanged official reward and completion. Exact scoring in a
known full track is a separate information setting.

First use one road hypothesis and one vehicle-state estimate. Then test small
scenario sets for hidden state and unknown road separately. Use common hypotheses
across candidates and keep each hypothesis consistent through its rollout. A
feedback candidate can react to future simulated sensor observations, but cannot
receive its scenario ID or geometry outside that simulated sensor footprint.

## Search progression

Keep the original proposer behavior explicitly among candidates. This makes
improvement over the base proposer testable; it does not guarantee improvement
when simulations, state estimates or finite-horizon scores are wrong.

1. **Fixed bank:** the existing 21 feedback controllers, to establish the effect
   of replacing fitted dynamics with programmed physics.
2. **Structured maneuvers:** time-varying speed/steering perturbations around the
   base behavior, including braking followed by turning and acceleration after
   a turn. Keep smooth controls and explicit recovery candidates.
3. **Adaptive allocation:** compare random shooting with a small CEM/iCEM-style
   refinement, warm-started from the previous chosen maneuver. Match candidate
   rollout work before claiming a better optimizer. Do not call CEM Bayesian.
4. **Frozen learned proposer:** evaluate the same competent trained policy
   directly and with simulation selection. Clone recurrent memory per imagined
   branch; imagined observations cannot update the real policy memory.

Suggested development grids are 8/21/64 candidate evaluations, 0.8/1.6/2.4-second
rollout horizons, and 1/4 world hypotheses. These are staged alternatives, not a
full Cartesian experiment. Start from measured bottlenecks and report native
physics steps, world construction, inference latency and wall-clock cost.
Code simulation trades training/scoring approximation for inference work; it is
not free compute and cannot be assumed to meet our 80 ms decision interval.

## Experiments and execution order

### P0: correctness and engine substitution — complete

- Synthetic checks against the upstream vehicle code under identical constructed
  starting states and on-road conditions; candidate independence; reset and
  first-transition observer carry-over.
- First condition: 21 bounded candidates, unchanged score, horizon 20 decisions,
  same 20 development tracks as earlier runs. Compare with all three frozen
  ridge runs and the selected same-family fixed controller.
- Second condition: 21 wider-speed candidates under the same programmed engine.
  This is a separate candidate-family intervention, not a pure dynamics effect.
- Budget: two conditions × 20 tracks, at most 40k native evaluation frames and
  16.8M internal physics frames. Zero new model training; the engine is deterministic.
- Manifest: [programmed-physics-2026-10-28.json](../configs/campaigns/programmed-physics-2026-10-28.json).
  Record complete runs and failures. Do not claim 40 independent tracks.

### P1: find the useful search/estimation budget — first follow-up complete

After P0, choose one limited follow-up from observed evidence: improve current-state
estimation, contact geometry, candidate pace/maneuvers, scoring/horizon, or search
allocation. Save the choice and a new budget before running it. Do not optimize
the whole pipeline at once and attribute the gain to one component.

The first follow-up restricted the wider bank to gain 1.2 and completed the
full wider fixed-controller comparison. Seven candidates produced 697.8 return,
40% completion and 48.9 ms p95 decision latency; 21 produced 737.3, 45% and
128.5 ms. The selected fixed reference produced 716.1 and 20%. These adaptive
20-track comparisons establish no mean-return advantage. The next proposed
diagnostic gave the same planners declared full-map geometry while retaining
estimated vehicle state (two conditions, at most 40k native frames). It is now complete; the execution report
records the declared adapter, manifest and results.

Measure one-step prediction errors using permitted observed ego transitions from
the same evaluated trajectories. These are ordinary state-prediction diagnostics,
not the user-only candidate oracle. Repeated rollout tests of known synthetic
worlds establish implementation behavior, not racing superiority.

### P2: prove value over the proposer

Use identical frozen proposer checkpoints, inputs and control frequency. Compare:

| Condition | What it tests |
|---|---|
| Original proposer, default action | Underlying behavior performance |
| Random sample from that proposer/candidate family | Effect of sampling without simulation selection |
| Simulation selection from the same candidate bank | Increment attributable to simulated outcome selection |
| Adaptive search with the same simulator | Increment from search allocation/coverage |
| Established planner with the same simulator/access/budget | Whether our particular planner improves on existing search |

Test at least a simple controller and a competent learned proposer. A win over
our failed 100k PPO pilot is insufficient. Count proposer training/pretraining
and observation reconstruction costs even when the selector itself is untrained.
Learned direct candidate scoring or a learned terminal-value variant can be a
clearly separate attribution baseline; it is not the default selector.

### P3: broader development and paper study

Use 200 fixed development tracks and independent trained proposer seeds. Five
training seeds remain the headline default for learned methods; do not fabricate
five independent replications of a deterministic coded planner. Freeze the complete
study before the reserved final test. All prior partition and privacy rules persist.

For standard native-pixel claims, the reconstruction/state estimate must come from
those pixels and permitted history. A local geometry result does not establish
standard CarRacing SOTA. Declare known physics access and compare with established
planners given the same simulator, as well as learned driving methods at reported
data/inference budgets. Extra simulator knowledge is an assumption to disclose.

## Prior art and claim discipline

- [POPLIN](https://arxiv.org/abs/1906.08649) already combines policy proposals with
  online planning, including action or policy-parameter optimization. Improving
  a base policy through simulated lookahead is not a new general idea.
- [iCEM](https://proceedings.mlr.press/v155/pinneri21a.html) supplies established
  memory and temporally correlated sampling techniques. Use it as a comparator
  or implementation foundation, not a new contribution of ours.
- [LOOP](https://proceedings.mlr.press/v164/sikchi22a.html) combines lookahead with
  learned dynamics and terminal value; it is a useful contrasting learned hybrid.
- [Valdi](https://arxiv.org/html/2607.00917v1) is direct actor-free CarRacing planning
  prior art with learned dynamics/reward/value. Its extra proprioception and
  600-frame setting require a separate reproduction; it is not an interchangeable
  standard-task score or a verified current champion.

The potential paper must establish a substantive advantage or finding: improvement
over competent frozen proposers, useful return-versus-compute scaling, or a justified
new treatment of partial-state reconstruction and simulation search. We cannot
promise SOTA or infer novelty from using coded physics instead of a neural model.

## Decisions for later discussion

| Choice | Confidence / unresolved issue |
|---|---|
| Coded simulator as the main world model | Explicit user decision; preserve this architecture intent |
| Reuse upstream vehicle equations | High engineering confidence; avoids inventing different vehicle dynamics |
| Separate reconstructed internal worlds | High methodological confidence; preserves observation boundaries |
| Carry wheel-state estimate and correct hull motion | Medium-low; hidden state may remain inaccurate or unidentifiable |
| Five-point wheel contact in first prototype | Low as geometric fidelity; explicit approximation to test against fixtures |
| Straight road continuation initially | Low as forecasting quality; useful controlled first hypothesis |
| Preserve old score for P0 | High for attribution; low confidence that it is a good final racing objective |
| Seven candidates as a low-cost reference | Measured latency benefit; driving tradeoff remains uncertain and coverage also changed |
| Same-planner full-map diagnostic | Complete: large geometry/reconstruction effect; does not separate visible reconstruction from unseen-road information |
| CEM/iCEM-style behavior refinement | Medium as established search tools; benefit and runtime here unmeasured |
| No learned selector/critic in primary system | Matches user intent; finite-horizon bias may require longer base-policy rollout |
| Eventual improvement over strong proposers/SOTA | Unknown; empirical target, not a promised outcome |

Existing private rank/regret outputs remain user-only and are not used for any
of these choices. Selecting a winner by internal expected utility does not require
hindsight-regret training. Regret-based studies and agentic evolution retain their
previous future-option status and access restrictions.
