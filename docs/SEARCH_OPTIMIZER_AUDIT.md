# Established search optimizer: source audit and proposed comparison

Reviewed 2026-10-28. [Index](../README.md) · [Next moves](NEXT_MOVES.md) ·
[References](REFERENCES.md). **Status: source audited; adaptation and driving runs
not implemented or launched.** The broad campaign and subsequent geometry diagnostics
are complete. Following the [20-track reference replication](REFERENCE_REPLICATION_2026-10-29.md),
this comparison is again the next implementation priority; it requires its own
bounded declaration and tests before new racing.

## Why this comparison

The user's main hypothesis concerns choosing behavior through programmed
simulation. Our current selector evaluates one population of residuals around
the frozen recurrent policy. An established optimizer is needed to determine
whether adaptive allocation improves that same system, and whether our simple
search offers anything beyond existing planning methods. This is an algorithmic
comparison, not a proposed new optimizer contribution.

iCEM targets more efficient trajectory optimization using correlated proposals
and memory. Its published gains concern its own tasks/settings, not our CarRacing
system. [Pinneri et al., CoRL 2020 / PMLR 2021](https://proceedings.mlr.press/v155/pinneri21a.html).

## Pinned implementation

Official repository revision `98c1c1fe2bfc94a87e4658b6c793f4e81bc30203`, MIT license.
The controller, parent MPC implementation, defaults and task examples were
downloaded without executing their legacy environment. File hashes and source
URLs are in [the audit record](../reports/references/icem_source_audit_2026-10-28.json).

The [pinned controller](https://github.com/martius-lab/iCEM/blob/98c1c1fe2bfc94a87e4658b6c793f4e81bc30203/icem/controllers/icem.py)
samples temporally correlated bounded action sequences, reduces populations,
fits elite mean/deviation, and shifts plans between decisions. It can inject
the distribution mean into the final evaluated population. Execution uses an
evaluated sequence. Within one decision, retained elite rollouts are cached;
shifted elites from the previous decision are simulated again. Consequently,
population size is not total work, and cached reuse differs from reevaluation.

The [default configuration](https://github.com/martius-lab/iCEM/blob/98c1c1fe2bfc94a87e4658b6c793f4e81bc30203/icem/settings/defaults/i-cem-blitz.json)
uses three iterations, ten elites, reuse fraction 0.3, smoothing 0.1 and population
decay 1.25. Its temporal-noise exponent is 3, whereas the
[HalfCheetah configuration](https://github.com/martius-lab/iCEM/blob/98c1c1fe2bfc94a87e4658b6c793f4e81bc30203/icem/settings/halfcheetah_running/i-cem-blitz.json)
uses 0.25. The source does not supply a validated CarRacing setting.

## Our proposed adaptation, with alternatives and confidence

**High confidence:** keep the original frozen actor, local input/history contract,
coded dynamics, observer, road reconstruction, horizon, utility and sparse renderer.
Only real observations advance real recurrent memory. Each optimization batch
starts from the same root memory and vehicle estimate. Keep the first-transition
wheel estimate belonging to the actually executed candidate, even if that
candidate was evaluated in an earlier batch. Reusing the last batch's estimate
would silently change the next decision's model initialization.

**Moderate confidence:** optimize full 20-step residual sequences around the
feedback policy. This supplies a temporal axis for colored noise and exact
one-step plan shifting. The final actions still come from the policy plus clipped
residuals. It is a policy-guided iCEM adaptation, not the original open-loop
action-space method. The alternative is CEM over our three knots; that isolates
the nine-dimensional parameterization but should not be mislabeled a full iCEM
reproduction. A later actor-free control requires a separate comparison.

**Proposed starting choices, low confidence:** exponent 1, the existing marginal
residual scales (0.10 steering, 0.15 gas/brake), four elites, smoothing 0.1 and reuse
fraction 0.3. Reset deviation each real decision; shift the mean and one retained
elite. Include the unchanged-policy trajectory as an evaluated incumbent and
execute the best evaluated candidate across the current decision. Maintaining
an incumbent across all batches is an explicit adaptation. Clip actual actions;
declare residual bounds separately rather than accidentally applying action
bounds to residuals. Fix the exact residual-bound and spectral-normalization
rules before an implementation smoke test; no driving sweep is authorized here.

For **21 total evaluations**, propose one base trajectory plus batches of 12 and
8. A shifted prior elite occupies a slot in the first batch, and the distribution
mean occupies a slot in the last. Within-decision cached reuse adds no simulation.
Two iterations are a budget adaptation. Four elites retain one reused candidate:
copying a 0.3 fraction with only two or three elites would round to zero and
silently disable memory. The unchanged base trajectory may enter the elite fit.
All actual evaluated candidates count, including repeated/duplicate sequences.

If the small-budget experiment is inconclusive, a separately declared **63-total**
comparison can use one base trajectory plus 26, 20 and 16 evaluations. This
matches the default decay structure without claiming those populations are
optimal. Seven total evaluations are too restrictive for this proposed four-elite
design; do not force a nominal iCEM label onto a different degenerate algorithm.

## Attribution controls and budget gates

The primary optimizer contrast needs **colored-noise shooting versus adaptive
colored-noise search**, both in the same full-sequence residual space. Retain
the existing three-knot shooting condition as a representation reference. A gain
over knots alone confounds parameterization and optimization. Match actual total
physics steps, horizon, actor-forward examples where possible, and information;
measure wall time separately because different batch sizes change throughput.

Start with deterministic cached training observations, synthetic optimizer tests,
then a two-track development smoke check. Tests must verify exact budget accounting,
unchanged real actor memory across imagined iterations, correct selected first-state
commit, repeatability, reset, bounded controls, and reevaluation of shifted plans.
Test cached cost reuse only while root state/world/utility are unchanged.

Only after those checks, declare a small development campaign: first 20 tracks,
new full-sequence shooting and adapted iCEM at 21 evaluations each. Maximum
40,000 real frames and 16,800,000 internal physics steps, excluding separately
declared smoke/replay costs. Reuse the already completed knot reference for return
comparisons only; its historical latency is not an interleaved timing comparison.
Keep seed zero initially; independent planning streams follow a useful, correct
pilot. Do not choose a winner from a single lucky trajectory or silently increase
the optimizer's total work.

**Uncertain outcomes:** improved internal objective can worsen real driving if
optimization exploits model/contact/geometry error. Public returns, completion,
off-road traces, model predictions and compute may guide interpretation. The
user-only privileged candidate ranks/regrets remain unavailable. If extra search
does not help, investigate the implicated component rather than describing the
result as proof that established optimizers fail. This audit does not authorize
final-test access, probabilistic-road claims or a SOTA conclusion.
