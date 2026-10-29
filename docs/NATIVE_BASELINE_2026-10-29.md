# Native baseline campaign: one seed, fixed schedule and selection

[Profile and interface](NATIVE_RECIPE_2026-10-29.md) · [Baseline audit](BASELINE_AUDIT.md)

**Origin:** the user's strong-comparator goal; the assistant's next bounded
experiment after the native interface and workstation profile. This is a native
pixel baseline under our recorded versions/partitions. It is not the local frozen
proposer and does not itself test simulation-search improvement or establish SOTA.

## Pre-execution choices

Start a fresh seed-0 run, using the successful execution setup chosen by the
profile's cost rule. Do not reuse profile-trained weights or select by profile
driving returns. Keep the audited CNN-LSTM architecture, eight workers, 512-step
rollouts, ten epochs and all recipe parameters. Linear learning-rate schedule:
1e-4 to zero over **4M aggregate one-frame decisions**. Freeze resolved settings
and selected profile provenance in the campaign manifest before launch.

Run four sequential stages ending at requested 0.5M, 1M, 2M and 4M decisions.
Expected post-update counts: 503,808; 1,003,520; 2,002,944; 4,001,792. Each stage
ends after a complete rollout/optimizer update. Resume restores weights, optimizer,
normalization, RNG and train streams. It starts fresh episodes and memory, so
this segmented execution is explicitly not identical to uninterrupted training.
Save all learning curves and checkpoints, including regressions or failed stages.

After each stage evaluate deterministic actions on the **same first 50 development
tracks** with CPU inference, frozen normalization, the native image history and
unaltered official reward/termination. The repeated panel is adaptive development
data, not four independent confirmations. Choose the checkpoint by mean official
return, then completion fraction, then earlier training step. Report the full
four-point curve. Do not stop solely because an early policy drives poorly and do
not tune on individual track outcomes during this campaign.

Native inputs include the official camera zoom and HUD, with no numeric ego or
previous-action field. Do not pool these results with the local 30 m mask/repeat-four
table or use a native-versus-local score difference as a causal component claim.
CPU/GPU action equality is not assumed; CPU evaluation is an explicit fixed choice.

## Budget, failures and next gate

- Training cap: 4,001,792 aggregate native frames after rollout rounding, 977 full
  rollout updates, one fresh training seed. Profile and correctness costs are extra.
- Checkpoint reload checks: eight train-only native frames, plus counted reset ticks.
- Development: four × 50 episodes, at most 200,000 native frames, plus reset ticks.
- No calibration, final-test, legacy held-out or private ranking data is accessed.
- Stop and preserve artifacts on nonfinite updates, interface/reload failure,
  subprocess failure or the declared free-disk floor. This is a technical stop,
  not a score-dependent checkpoint choice. No automatic hyperparameter retry.
- All stages run serially. Disk availability is checked before each stage; no
  existing data is deleted. No paid cloud or additional hardware is used.

Short-profile timing gives a provisional training ETA; learning sequence lengths,
episode lengths, evaluation and checkpoint I/O can change it. Report a range and
update it from the running campaign. The first 0.5M checkpoint is a health/data
milestone, not an asserted adequate baseline.

After the full curve, decide whether native competence is established for this
one seed and protocol. Then design a stronger **matched local** proposer trial
using the cadence/budget audit, retain independent-seed requirements, and freeze
proposers before direct-versus-original-simulation-search comparison. The trained
V/L/F decomposition and modern world-model comparators remain later gates.

## Execution status

The completed profile selects CUDA with eight subprocess environments. Measured
projection is 2.05 training hours; allow provisionally 2–3 hours including the
declared 200 development episodes and checkpoint overhead. The first 0.5M data
milestone is approximately 15–25 minutes after launch. These are short-profile
estimates, not guarantees. The [frozen manifest](../configs/campaigns/native-baseline-2026-10-29.json)
records all settings and selection provenance. Launch follows the full-episode
evaluator/replay readiness check.

Readiness passed before launch: both complete two-track evaluation passes produced
identical actions, rewards, stacked images, terminal observations and episode
metrics. Normalization remained frozen; no extra track was consumed at final
VecEnv reset. Cost: 4,000 development frames and four reset ticks, separate from
the main campaign cap. The [verification artifact](../reports/native_profile_2026-10-29/evaluation_readiness.json)
records exact compared fields and immutable artifact hashes. No training setting
or checkpoint choice was changed from these smoke outcomes.

### Launch record

Observed running at `2026-10-29T11:06:08.023930+00:00`. Campaign: `runs/campaigns/native_baseline_20261029_s0`; runner PID 2720951. Training source/launch commit: `925a4be`. First stage:
`runs/native_training/native_main_20261029_s0_to500k`; early optimizer updates are
finite. The runner proceeds serially through the four declared training/evaluation
stages unless a technical failure or disk floor stops it. This document records
launch, not completion. Read `status.json` and the active training `learning.jsonl`
for live progress; later status must not be inferred from this launch snapshot.
