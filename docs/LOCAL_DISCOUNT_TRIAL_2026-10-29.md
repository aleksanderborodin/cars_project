# Bounded local-policy discount trial

[Baseline audit](BASELINE_AUDIT.md) · [Native results](NATIVE_BASELINE_RESULTS_2026-10-29.md) ·
[Native deployment diagnostic](NATIVE_DEVICE_AUDIT_2026-10-29.md)

**Origin:** the user wants a stronger frozen proposer for the main programmed-
simulator comparison. The assistant proposes one targeted local training trial
after the native campaign and deployment control. This is an engineering hypothesis,
not a user-prescribed hyperparameter or a claimed solution to native regression.

## Change, rationale and controls

The original local driver uses gamma=0.99 per decision, with each decision held
for up to four native frames. Four native-frame discounts of 0.99 instead multiply
to **0.99⁴ = 0.96059601**. Test that PPO gamma under the existing local interface.
This lowers the effective planning horizon of the learner. High confidence in
the discount arithmetic; low confidence that the change improves driving.

Keep reward-normalization gamma explicitly at **0.99**. The old trainer tied it
to PPO gamma, so an optional separately configured normalization discount is
implemented before this trial. Its default preserves existing behavior. This
avoids changing a second configured normalization timescale along with PPO gamma;
the learned return statistics will still differ when trajectories differ.

Keep local-v1 inputs (30 m road/visibility masks, ego and previous action), repeat
four, encoder/architecture, seed 0, CPU/four subprocess workers, n_steps=512,
batch=128, ten epochs, learning rate, clipping, GAE=.95, SDE resampling every four
decisions and reward clipping unchanged. No new observations, expert labels,
scenario forecasts, learned scorer or global information enter the actor.

This does **not** reproduce the native recipe: the within-repeat reward sum is
undiscounted, GAE and exploration cadences remain local, observations differ,
and early termination can shorten a repeat. It is a single configured PPO-gamma
probe, not full time-discretization equivalence or a causal component decomposition.

## Frozen budget and selection

Train fresh to **2M native frames**, completing the final rollout (cap 2,008,191),
under the same global 8M-native-frame linear learning-rate schedule as the original
local trial. This is roughly 0.5M learner decisions, not 4M decisions. Save/evaluate
checkpoints at requested 0.2/0.5/1/2M frames on the same first 50 development tracks.
Cap evaluation at 200,000 native frames, plus eight train-only reload-check frames
and separately counted reset ticks. No automatic training beyond the 2M review.

Select by mean official return, then completion, then earlier checkpoint. Report
all four checkpoints. The original selected local checkpoint is the historical
reference: 858.11 return, 30% completion on this same development panel. Retain
the original proposer/search result unless a new frozen comparison justifies
changing it. A higher selected score in one adaptive trial is not independent
confirmation or SOTA; independent seeds and fuller competence checks remain open.

Use the existing driver-block runner: one training process and one queued
checkpoint evaluator. This matches the earlier local execution layout; their
contention is part of measured wall time. Provisional duration 25–35 minutes from
the earlier CPU4 2M block, not a guarantee. Preserve failed trials and stop on
technical errors. No calibration, final-test or user-only rank access occurs.

Before launch, pass the full test suite and replay the first two episodes of the
old selected local policy exactly under current code (timing excluded), capped
at 2,000 extra native frames. The [manifest](../configs/campaigns/local-discount-2026-10-29.json)
pins config and control artifact hashes. The historical trainer differs in
metadata, diagnostics and optional disabled KL stopping; its original encoder,
physics and local evaluation behavior are retained. Reuse is a historical
reference, not a new simultaneously trained randomized control.

## Alternatives and interpretation

Alternatives: increase decision budget without changing gamma; study GAE or SDE
cadence separately; change the encoder; run independent seeds; compare uninterrupted
and segmented training. These are not bundled into this probe. If the trial fails,
record it and reassess the next comparator; do not keep tuning individual tracks.
If it improves, freeze the standalone-selected checkpoint before testing the same
original simulation search and matched cheap/random controls. This serves the
user's understanding-versus-deciding question without presupposing its answer.

Status: declared; test and historical replay gates precede launch.
