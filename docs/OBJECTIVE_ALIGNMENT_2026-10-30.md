# Align search with measured driving: bounded score-only pilot

**Completion update, October31:** both campaigns finished successfully. See
[completed results](RESULTS_2026-10-31.md). Neither penalty variant passed the
pilot screen; no setting is promoted and no automatic expansion follows. Both
jobs ended and the scoped sleep guard released. Execution/ETA statements below
preserve the earlier declaration or morning snapshot, not current live status.

Declared 2026-10-30. [Loss diagnosis](SEED1_SEARCH7_DIAGNOSIS_2026-10-30.md) ·
[Experiment contract](EXPERIMENT.md) · [Campaign declaration](../configs/campaigns/objective-alignment-2026-10-30.json)

**User intent:** change search so that it helps the actual final driving outcome,
while keeping the paper scientifically truthful. This authorizes development of
the selector, not a promise of improvement or a rewrite of old negative results.
The main model remains programmed simulation.

**Assistant first intervention:** isolate one plausible cause before changing
geometry, dynamics, candidates, horizons and objectives together. Scale all old
nonprogress penalties while preserving the progress term. Numerical values, two
actors, twenty-track budget and scheduling below are assistant choices. They are
not user-prescribed parameters or claimed prior-work innovations.

## Exact change and limits

The separate `policy-objective` method uses

`internal_cost = alpha * original_nonprogress_penalties - original_terminal_arc_progress`.

| Setting | Meaning |
|---|---|
| alpha=1 | Original score, with complete saved-trace parity required |
| alpha=0.25 | Reduce departure, center-distance, unknown-area and lateral-speed penalties together |
| alpha=0 | Progress-only extreme ablation; no explicit versions of those penalties |

`PolicySearchConfig` and old method defaults are unchanged. Only the explicitly
named `ObjectiveSearchConfig` carries the new scale; the CLI requires an explicit
scale for this new method and rejects it for other methods. At alpha1 the shared
rollout keeps the old arithmetic/order without a new multiplication.

**This is still a proxy.** It does not estimate exact official distinct-tile reward,
add a terminal value, reconstruct visited tile IDs, remove hidden-road uncertainty,
or guarantee that simulated ranking matches actual return. No global progress,
true future road or wheel state is provided. Native reward/physics/termination
are unchanged. Alpha0 may encourage model exploitation or poor road retention;
it is an informative extreme, not a proposed default asserted to work.

The [diagnosis](SEED1_SEARCH7_DIAGNOSIS_2026-10-30.md) found that lower coverage,
including stalls and off-road failures, explains the observed return loss. It did
not identify penalty weights as the cause. The trial asks whether changing those
weights improves actual driving. A uniform time penalty across equal-length
rollouts would cancel in ranking, so it is not added as a purported repair.

## Fixed components, panels and costs

Use the two existing useful local actors, training seeds0/1, selected previously
by standalone driving. Neither is a converged SOTA baseline. Seed2's failed
learner remains in the full current matrix; omitting it from this explicitly
useful-proposer diagnostic does not permit omitting it from that report or from
future robustness claims. Do not choose a different checkpoint because it benefits
more from a new score.

Keep the same local30m road/ego/history inputs, repeat4, frozen recurrent proposer,
estimated state/contact, tangent continuation, sparse renderer, seven candidates,
three residual knots/scales, 20-decision horizon, planning stream0 and execute-one-
decision/replan loop. The candidate family and random stream are fixed; realized
states and subsequent proposals naturally diverge after different actions.

For each actor, rerun alpha1, alpha0.25 and alpha0 on the **first20 development
tracks**. Six new search panels, all retained. Before them, two direct-policy
episodes per actor must reproduce saved data; alpha1 must reproduce all twenty
saved search7 episodes before proceeding to its modified settings. Reuse the
saved direct first20 table after its parity check. The source, immutable weights,
inputs, controller and analysis are frozen before any new driving.

New caps: **124,000 real native frames and 16,800,000 internal native physics
ticks**, zero policy/model training. They include four direct parity episodes and
120 search episodes at worst-case episode length. Actual native frames, actor
forwards, internal transitions and wall time remain separate run costs. Failed
arms stay recorded; no incomplete matrix is presented as a complete comparison.

## Analysis, selection and paper integrity

- **Primary:** alpha0.25 minus alpha1 official mean return, separately for each
  actor. Secondary: alpha0 versus alpha1 and both settings versus direct.
- Co-report completion, distinct-tile coverage, off-road frames and a frame-weighted
  estimate of time with sampled decision-boundary speed below2m/s. This is not a
  newly sampled native-frame speed sensor. Count control, validation and failed
  trial costs. Do not switch the primary metric after seeing completion changes.
- Use paired track differences and descriptive 5,000-resample intervals with
  seed20261030. Two actor instances and reused development tracks do not support
  headline independent-training or final-test uncertainty.
- A setting is merely screened for **separately declared broader validation** if
  its point return exceeds both direct and alpha1 for both actors. A small positive
  difference is not proof of practical/statistical improvement. Publish every arm
  regardless; no automatic default change, budget expansion or final-test access.

These tracks and actors have already informed development and the failure diagnosis.
The pilot is **adaptive tuning/diagnosis**, not untouched confirmation. Any selected
configuration needs a subsequent frozen broader study, stronger comparators,
independent actor seeds and final evaluation before a strong paper claim. Keep
the original negative result and count this search among development trials.

An improvement would show that this penalty balance matters conditional on our
other components. A loss would not prove that every objective repair is useless.
Neither outcome declares perception/prediction solved, validates a universal
decision bottleneck, or establishes novelty. Search failure despite accurate models
is already close prior work: [MRS.Q, ICML2026](https://proceedings.mlr.press/v306/chang26a.html).
Optimizer sampling efficiency is another established axis, e.g.
[iCEM](https://proceedings.mlr.press/v155/pinneri21a.html); this trial changes score
weights rather than claiming a new optimizer.

## Execution and next decisions

The current three-policy matrix stays unchanged and finishes first. The pilot
controller snapshots the new source and waits for that recovery's successful
completion. Dependency failure/staleness, parity failure, changed input, resource
cap or less than2GB free output space stops the pilot. All outputs, temp files
and caches use the mounted secondary volume. Runtime allowance is approximately
**one to one-and-a-half hours after the current matrix**, with uncertain episode
lengths; this is not a promised improvement or complete-paper ETA.

```bash
uv run python scripts/run_objective_alignment.py --manifest configs/campaigns/objective-alignment-2026-10-30.json --check
uv run python -m cars.cli research evaluate --help
```

The new method requires `--method policy-objective --penalty-scale VALUE`; manual
driving runs should not compete with the queued measurements. Full project tests
and exact legacy-output/score-only isolation checks precede arming the pilot.
CPU software-check overlap with the existing matrix is disclosed in the check
record; inference-efficiency claims still require dedicated clean measurement.

Full CPU checks passed **140 tests with one CUDA-only skip**. The controller was
armed at **2026-10-30 18:50:37 UTC**, with zero pilot driving frames and status
`waiting_for_frozen_matrix`. Its live status is
`/media/sasha/72AECF44AECEFF99/cars_project_recovery_20261030/objective_alignment_20261030/queue/status.json`.
The [launch record](../reports/checks/objective_alignment_launch_2026-10-30.json)
and [software-check record](../reports/checks/objective_alignment_software_2026-10-30.json)
separate preparation from actual driving results. All 73 frozen production-file
hashes, controller/reporter hashes and 94 required inputs are checked before staging.

If these score variants are not useful, the next separately declared candidates
are locally observable distinct-progress credit and continuation/horizon treatment,
with model/road errors isolated rather than assumed away. A learned terminal value
is an optional comparison, not an activated replacement for programmed simulation.
Custom/GPU models and agentic evolution remain future options. No private ranking
or oracle-return stream may be used to tune this or any later variant.

Confidence: high that the controlled comparison is interpretable; low that any
tested alpha will improve both actors before results exist. The requirement is
truthful evidence, including the possibility that no setting qualifies.
