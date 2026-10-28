# Future study: can online search compensate for less road information?

2026-10-28. [User intent](USER_INTENT.md) · [Specification](EXPERIMENT.md) ·
[Current evidence](FROZEN_POLICY_RESULTS_2026-10-28.md) · [Next moves](NEXT_MOVES.md)

**Status:** future experiment requested by the user during the current execution
batch. No new global-policy training, exact-state oracle or final-test evaluation
is launched by this document. Current `global` observations remain restricted to
development diagnostics; a future training interface needs its own explicit
protocol, tests, partition enforcement and budget.

## User's idea and purpose

User proposal, paraphrased: compare methods when the road is fully known, so the
system can try different behaviors in simulation and choose the best. Also compare
our restricted-information search against a learned driver trained and deployed
with full information. Separate understanding/predicting the track from navigating
it, possibly into three components, to explain where imperfect components still
produce better driving. The broader ambition is an important family of approaches
that allocates work to online simulation rather than learning all control directly.

**Purpose:** distinguish an information limitation from errors in vehicle prediction
or action choice, and test whether online computation can compensate for some
limitations of a learned controller. These are hypotheses to test, not conclusions
that a learned model cannot match search or that all possible actions are enumerated.
The user explicitly places this extension in future work; it should not interrupt
the current restricted-policy comparison or silently replace the main system.

**Subsequent user clarification:** the understanding-versus-deciding distinction
is the main scientific goal now; the full-map matrix is one future way to examine
it. Identify components sufficient for our tested setting, not universally solved
problems. Simulator realism is optional if useful choices survive in actual
closed-loop driving; see [the future task-useful simulator study](DECISION_USEFUL_SIMULATOR.md).

## Define what is fully known

Start with **full road geometry**: the entire ordered track/map is available at
deployment, with the same declared ego sensors and history as local methods.
Train and evaluate the full-map learned policy with that input; do not merely
give a local-trained network extra fields at test time.

Full geometry is not complete simulator state. Wheel rotation, steering-joint
state, contacts, solver impulses and state-estimation error can still matter.
Our existing full-map diagnostic uses estimated wheels and approximate contact;
it also changes centerline reconstruction. Call it full geometry, not a perfect
world model or proven upper bound. An exact-state diagnostic would be a separately
designed privileged protocol, not a silent extension of the present adapter.

Distinguish **an unseen track whose map is supplied at episode start** from a
fixed track optimized or memorized before evaluation. For the first study, use
disjoint train/development/eventual-test tracks from the same generator. Supplying
the map is declared deployment information, not permission to train on test tracks.
If later allowing per-track optimization, count its startup/adaptation compute
and give comparison methods the same opportunity.

## Two tasks, or three modules

The user's two-task description is useful: construct an internal understanding
of the world, then decide how to drive within it. The assistant proposes splitting
the first task for diagnosis:

| Module | Responsibility | Controlled intervention | Evidence |
|---|---|---|---|
| Road inference | Estimate visible geometry/topology and plausible unobserved continuation from allowed observations/history | Same controller with observed reconstruction, forecasts, or declared full geometry | Geometry/topology errors; uncertainty calibration if a distribution exists; actual driving |
| Vehicle-state and motion prediction | Estimate latent vehicle state and simulate actions in each road hypothesis | Same road/actions with different observer/contact approximations in constructed known worlds | Executed-action motion errors over declared horizons; actual driving and compute |
| Decision/control | Propose behaviors, simulate, score, choose, execute and replan | Same frozen proposer/model with direct execution, cheap controls, established search and our search | Official return, completion, failure breakdown and computation cost |

These modules interact; their effects need not add independently. A factorial
comparison or targeted interventions are more informative than saying a total
score difference belongs entirely to perception or planning. For a pure information
intervention, preserve representation/path construction where possible; otherwise
report the extra topology/reconstruction changes explicitly.

Vehicle mechanics remain programmed in the main system. A learned transition
model is a separately labeled alternative. Proposal generation belongs within
control; using a learned proposer still counts its training cost. Known-map
experiments bypass road uncertainty but do not automatically remove the other
two modules' errors.

## Proposed information-by-control matrix

| Input at deployment | Strong direct learned policy | Same frozen policy plus programmed simulation selection |
|---|---|---|
| Restricted local geometry/history | L-direct | L-search |
| Full road geometry, same ego sensors/history | G-direct | G-search |

Train competent local and global policies separately, with declared architecture,
capacity, data and optimization budgets. Select each checkpoint by standalone
development performance, then freeze it for its paired search conditions. Match
the actual representation supplied within each row. Include the same known-physics
access for planning comparisons and disclose that physics code is prior knowledge
when comparing against a learned-only method. A famous algorithm name or full-map
input does not establish that a baseline learned successfully.

At least three kinds of contrast are useful:

1. **L-search − L-direct; G-search − G-direct:** benefit of online selection around
   the same frozen actor under each information regime. A fixed actor does not
   itself perform an online candidate search, but it may learn equally good actions.
2. **G-search − L-search; G-direct − L-direct:** effects of map access within each
   method family, with representation, training and compute differences declared.
   These are not same-checkpoint contrasts when the actors are separately trained.
3. **L-search − G-direct:** the user's cross-information hypothesis. A positive
   result would show that this restricted-information planner beats this tested
   full-map learned driver at the reported resources. It does not establish that
   information is harmful, that road prediction is unnecessary, or that an optimal
   full-information controller would lose.

An ideal full-information controller can ignore extra observations and emulate a
restricted controller if it is allowed the same computation. Therefore a practical
cross-over measures approximation, training and inference tradeoffs; it cannot
reverse the value of information for otherwise unconstrained optimal control.

Add a fixed full-map controller and established sampling/MPC optimizers to avoid
mistaking easy known-map driving for a unique benefit of our search. The existing
privileged hand-engineered expert is not a full-information learned comparator.
A map-trained policy should also be assessed at more than one justified training
budget before explaining a loss as a fundamental limitation of learned control.

## Fairness, statistics and access

- Same official physics/reward/termination, tracks, decision frequency and action
  space. Full-map and local conditions have explicitly different information.
  Native-pixel benchmark claims remain a separate experiment.
- Report training/tuning frames, map/teacher labels if any, online simulator work,
  latency and startup/adaptation work. Show quality-versus-cost curves; a slower
  planner winning offline is not evidence of equal-latency superiority.
- Pair common tracks. Pair checkpoint identities only within direct/search rows.
  Separately trained local/global actors require a declared independent-model or
  explicitly coupled-training analysis, not merely matching integer seeds.
  Use multiple training seeds for headline claims and fixed planning repetitions.
- Retain all attempted configurations and competent learned and planning
  comparators. Select on development; freeze before the reserved final campaign.
- The user's private ranking/regret stream remains user-only. This study does not
  authorize using it for method analysis, fitting or selection. Component evidence
  uses permitted observations, executed outcomes, declared geometry diagnostics
  and synthetic/constructed-world checks instead.

## Novelty and confidence

The assistant judges the comparison structure useful with **high confidence**;
whether local search beats a strong full-map learned driver remains **unknown**.
The available positive result is only L-search versus one L-direct instance on
20 development tracks. G-direct and the paired G-search condition do not yet exist.

Online planning and learned-policy proposals have substantial prior art.
[PETS](https://arxiv.org/abs/1805.12114) combines learned probabilistic dynamics with
trajectory sampling; [POPLIN](https://arxiv.org/abs/1906.08649) combines policy
networks with online planning and also studies distilled direct policies. Our
programmed dynamics are a different modeling choice, not proof of a new field.
The paper needs a specific supported contribution: a better method, a useful
cost/quality tradeoff, or a careful finding about which inference/control errors
online computation can compensate for. The user's broader ambition stays recorded
without converting it into an established novelty or universal-superiority claim.
