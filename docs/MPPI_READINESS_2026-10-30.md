# Same-engine MPPI: source and comparison readiness

Reviewed 2026-10-30. [Comparator review](COMPARATOR_REVIEW_2026-10-30.md) ·
[Source record](../reports/references/mppi_source_audit_2026-10-30.json)

**Status:** reference sources pinned and inspected, not integrated or executed.
No current campaign, simulator, score or policy is changed. The user's main idea
is programmed-simulation behavior selection; the assistant adds an established
planner control to test whether its benefit survives a credible alternative.
MPPI and physics-simulator planning are prior work, not project inventions.

## Sources and implementation choice

[Physics-simulation MPPI](https://arxiv.org/abs/2307.09105) is close methodological
prior work. Its authors' [mppi_torch repository](https://github.com/tud-amr/mppi_torch/tree/75e17e87fd42a803d3227258d14c4231cbcbc0ec)
was pinned at `75e17e87fd42a803d3227258d14c4231cbcbc0ec`; five selected files were
cached. Its inspected Git tree has no license file and its package metadata does
not declare one. Retain it as a reference and do not copy it into project source.

The credited [UM-ARM-Lab reference](https://github.com/UM-ARM-Lab/pytorch_mppi/tree/e04a569cd4215e705f9013145c496fc59cb25ed6)
is separately pinned at `e04a569cd4215e705f9013145c496fc59cb25ed6`, with an MIT license.
Its README, license, package metadata and controller were cached. **Assistant
implementation choice:** use this as the algorithm reference, preserve attribution,
and declare our NumPy/Box2D adapter separately. No package was installed or imported.
Nine cached hashes and three Python AST parses passed; runtime parity is untested.

The references are not interchangeable configurations. The former includes
simple and Halton-spline modes, prior slots, temperature adaptation and optional
filtering. The MIT base MPPI bounds perturbations, recomputes effective noise,
includes the control/noise term, updates a nominal sequence by weighted noise,
and shifts it between decisions. A weighted average over our existing costs is
not automatically a faithful implementation of all those choices.

## Separate the questions

| Comparison | What it can identify |
|---|---|
| Full action-sequence shooting versus base MPPI, same coded world and objective | Optimizer effects in a shared action representation |
| Feedback-policy residual shooting versus policy-guided MPPI adaptation | Optimizer effects around the same frozen proposer; explicitly an adaptation |
| Existing three-knot search versus either full-sequence method | Combined representation/optimizer difference |

Choose one bounded implementation study first, not a Cartesian sweep. Reuse the
existing shared rollout/state initialization and retain the old method unchanged.
An actor-free MPPI reference needs its own open-loop rollout adapter; the current
`evaluate_feedback` routine assumes future actor observations and memory. Do not
silently relabel that feedback controller as standalone open-loop MPPI.

## Synthesized controls require their own model transition

MPPI normally executes a weighted nominal control rather than the sampled
minimum-cost trajectory. Averaging actions, residuals or controller parameters
does not average vehicle/contact states. The resulting first action may not have
been simulated at all. Our estimator must commit its own first-transition snapshot,
not the best sample's snapshot or a mixture of wheel states.

For a **proposed 21-rollout-equivalent** feedback adaptation, one possible complete
budget is twenty population rollouts plus one synthesized-sequence rollout, each
twenty decisions at repeat four: **1,680 internal native physics ticks** per real
decision. This matches historical search21's physics work while disclosing fewer
population samples. The synthesized rollout supplies the committed state; its
cost must not be used for an undeclared best-sample fallback. A first-action-only
commit rollout is an alternative with different exact costs, not a free operation.

These are design options, not frozen run settings or a declaration that twenty
samples constitute a strong MPPI configuration. A low-budget loss cannot refute
MPPI generally. Larger-population performance/latency curves need a separate
declaration and resource allowance. The final comparison must show both driving
and measured compute, including any incumbent, mean or shifted-plan reevaluation.

## Required gates before driving

- Implement bounded-action, noise-cost, weighted-update and shift/reset parity
  against the pinned reference on synthetic dynamics. Check normalization at
  extreme costs and reject nonpositive temperatures.
- Verify all batches start from the same observation-derived road/vehicle state.
  For policy-guided variants, copy imagined memory and keep real actor history
  unchanged. Test that synthesized-action commit matches that action's rollout.
- Keep allowed information, state/contact estimate, road continuation, horizon,
  official environment, objective terms and action cadence fixed within optimizer
  comparisons. Record any necessary control/noise regularization separately.
- Declare temperature, covariance, initialization, filtering, sampling and total
  work before a bounded development pilot. Give competitors comparable tuning
  allowances and charge unsuccessful trials. No present private ranks or regrets
  may guide these choices.
- Run full project tests and train-side synthetic/replay checks first, then a
  separately declared small development profile after the frozen queue finishes.
  Only subsequently schedule a paired driving comparison and independent planning
  streams. Do not compete with current latency measurements.

Confidence is high in the need for these accounting and algorithm-identity
controls. The useful temperature, population and parameterization are unknown.
No MPPI performance, efficiency or SOTA result is established by this audit.
