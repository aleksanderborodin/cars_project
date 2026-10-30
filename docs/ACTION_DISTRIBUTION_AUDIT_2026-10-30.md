# Offline action-distribution and update audit

Pre-execution declaration, 2026-10-30. [Latest local trial](LOCAL_DISCOUNT_TRIAL_2026-10-29.md)

**Origin:** user requests continuing the paper work; assistant proposes this
diagnostic after native/local training regressions and unsuccessful repair probes.
Purpose: distinguish the logged learned gSDE noise scale from effective control
variance and clipping. High confidence in the measurement value, unknown cause
of observed regression. No cause, performance gain or new training recipe is assumed.

Freeze all four checkpoints from each of three completed families: original local,
local discount, and native. For every checkpoint replay the first five recorded
development episodes twice: once from that checkpoint's own evaluation and once
from a fixed common history within its protocol. Common local histories come from
the original selected local 1M policy; common native histories from native 1M.
Five tracks are a fixed diagnostic panel, not an independent performance estimate.
Replay actor memory from zero through complete histories. Only permitted actor
observations enter inference. Common histories are off-policy conditional inputs:
they do not predict what the inspected actor would actually observe while driving.

Measurements per steering/gas/brake channel:

- Raw Gaussian mean and state-dependent marginal standard deviation, verified
  against the gSDE latent-feature/noise-scale formula in installed SB3.
- Deterministic action clipping frequency and analytic sampled clipping probability.
- Exact clipped-Gaussian expectation/variance and the gap between clipped mode
  and expected clipped action. These are single-history marginal statistics,
  not a new controller or Monte Carlo driving scores.
- Probability of gas and brake both exceeding 0.05 at one history, and overlap
  of deterministic controls. The threshold is a declared diagnostic convention.
- Last 20 logged PPO update summaries near each checkpoint: approximate KL,
  clipped fraction, logged learned noise scale and explained variance. Association
  with actor statistics does not establish a learning mechanism.

Require deterministic clipped means to reproduce every recorded own-history action
within 1e-6; compare the first 16 steps per episode with the standard model.predict
API exactly. Verify analytical marginal moments against independent numerical
quadrature in tests. Replay once per panel, save results and hashes under new IDs.
Policies and their normalizers remain fixed; observation normalization is disabled
in these protocols. No training, scenario selection or environment instantiation.

Budget: 12 checkpoints × two panels × five episodes; at most 60,000 offline
decision replays and 61,920 actor forward examples including API parity checks.
**Zero new environment steps, internal physics steps or optimizer updates.**
No private candidate ranks/oracle regrets, calibration or final-test identities
are read. The [manifest](../configs/campaigns/action-distribution-audit-2026-10-30.json)
pins model and permitted-history artifacts. Do not expand panels or choose a
different checkpoint from these measurements.

gSDE shares sampled noise across time; per-history Gaussian formulas do not
reproduce that temporal correlation or sampled-policy driving. Small learned
noise-scale parameters alone cannot establish exploration collapse. The audit
may motivate one separately declared replay/learning control or an independent-
seed baseline replication. It does not authorize an open-ended repair sweep.

Status: declared; implementation/tests precede execution. All existing checkpoint
and method-selection decisions remain as recorded in the completed campaigns.
