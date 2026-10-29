# Native recurrent PPO: interface and workstation profile

Declaration date: 2026-10-29. [Index](../README.md) · [Audit](BASELINE_AUDIT.md)

**Origin:** the user wants competent learned comparators and improvement from
simulating a frozen proposer's alternatives. The assistant proposes this bounded
recipe implementation/profile to close the audited native-interface gap before
committing workstation time to a larger baseline run. This is a separate native
pixel comparator; it does not replace the local programmed-simulator study.

## Frozen interface and recipe

Source: RL Zoo commit `61956d90d9f8ac126043ed4024efa12a462d97e1`,
[CarRacing-v3 recipe](https://github.com/DLR-RM/rl-baselines3-zoo/blob/61956d90d9f8ac126043ed4024efa12a462d97e1/hyperparams/ppo_lstm.yml).
The cached source and factory hashes are indexed in [BASELINE_AUDIT.md](BASELINE_AUDIT.md).
Use official native RGB, including the camera zoom and HUD, resize to 64×64,
grayscale with singleton channel, reward-only VecNormalize, two-frame stacking,
then channel-first conversion. The actor receives only uint8 `(2,64,64)` pixels,
scaled by the standard CNN preprocessing. One action advances one native frame.
No ego fields, previous-action field, track seed or geometry enters the policy.
The initial older image is zero; autorestarts clear frame and recurrent histories.

Use CnnLstmPolicy with the standard NatureCNN (512 features), empty post-LSTM
MLP, 128-unit actor LSTM, separate linear critic projection, GELU, no orthogonal
initialization, log_std=-2, state-dependent exploration resampled every four
decisions. Eight environments, 512 steps per worker, batch 128, ten epochs,
gamma=.99, GAE=.95, clip=.2, entropy=0, value coefficient=.5, gradient clip=.5.
Linear learning rate starts at 1e-4 over **4,000,000 aggregate decisions**. Reward
normalization uses gamma=.99 and clip=10; evaluation uses original rewards and
frozen statistics. Actual full rollouts may exceed the requested stop milestone.

Adaptations: our disjoint partition-enforced train streams, recorded installed
dependencies, deterministic execution settings, post-update checkpoints and
cumulative schedule across interrupted segments. Resume restores model, optimizer,
normalizer, RNG and train streams but starts fresh episodes/histories. It is not
bitwise uninterrupted training. This is recipe alignment under recorded versions,
not an already successful reproduction of a published score or SOTA result.

## Bounded execution declaration

After correctness tests, run four **serial** profiles: CPU/dummy, CPU/subprocess,
CUDA/dummy, CUDA/subprocess. Each has eight environments, training seed 0 and
**three complete 4096-transition rollouts** (12,288 native frames). Keep the full
4M schedule and all learning settings identical. One Torch/OpenMP/BLAS/OpenCV
thread per process. First-rollout startup/warmup and later timings are separate.
Total cap: **49,152 training frames**, plus at most two train-only checkpoint
reload actions per case (8 frames), reset physics ticks and separately recorded
correctness tests. No development or final-test performance selection is permitted
in this profile. No large training run is implicitly launched by it.

Choose execution backend by finite updates, successful reload and steady rollout
plus update time, with memory and total wall time reported. Do not choose by
training return. Record initialization hashes, rollout/optimizer counts, all
attempts/failures, source/lock/config hashes, hardware, reset counts and GPU peak
allocation. Small-case timings give a provisional 4M runtime estimate, not a
promise: render cost, episode lengths and recurrent minibatch padding can change.

Scheduled future checkpoints: 0.5/1/2/4M decisions after full optimizer updates.
Before longer training, freeze its campaign, development selection rule and total
evaluation budget. Native results stay separate from the existing repeat-four
local proposer/search table. A stronger matched local proposer and multi-seed
comparison are subsequent work, not outcomes of these short profiles.

## Confidence and remaining questions

High confidence in the wrapper/budget correction, subject to parity tests. Medium
confidence in profiling transfer to long-run throughput. Low confidence that this
recipe alone resolves learning instability or establishes a strong comparator.
Alternative: profile fewer workers or a smaller network if memory fails; such a
change requires a new declared adaptation, not silent substitution in these cases.
The larger V/L/F decomposition, custom simulator and agentic evolution remain
separate future studies. User-only ranking diagnostics remain unread.

## Execution status

Implemented and profiled. All 106 tests pass. Original CPU cases and corrected
GPU cases completed; original GPU verification failures remain preserved.
See the measured results below.

## Technical follow-up declared after original profile

All four original cases completed 12,288 training frames. Both CPU runs passed;
both CUDA runs failed only their post-training GPU-versus-CPU numerical assertions:
reported action difference 3.92e-6 in the dummy case; recurrent-state difference
2.21e-5 among violating coordinates in the subprocess case. These are failed
original runs, retained unchanged. They do not demonstrate corrupted checkpoints.

**Assistant correction:** require exact saved/reloaded action and recurrent-state
agreement on the training device, and exact repeatability of two independent CPU
loads. Report CPU/GPU numerical differences separately, without implying identical
cross-device driving. Training settings and all wrapper semantics stay unchanged.
Add a CUDA integration test. Then rerun only the two CUDA profile cases, three
rollouts each, under new immutable IDs: additional 24,576 training frames and four
reload-check frames plus resets. Do not rerun valid CPU cases merely for a cleaner
report. The [follow-up manifest](../configs/campaigns/native-profile-reload-fix-2026-10-29.json)
records this technical correction before execution. CPU/GPU validation changes are
not return-based method tuning. Retain original failed training costs and report
that their interrupted reload checks consumed between zero and one step each;
the original failure logger did not save that exact counter.

## Completed profile and next decision

The [public results and provenance](../reports/native_profile_2026-10-29/summary.md)
verify 408 frozen source-file hashes across all six attempts and identical initial
policy parameters. Only native_training.py changed between original and retry
snapshots: verification, failure-cost persistence and metadata; no learning-rule,
architecture, preprocessing or budget change. No development score was used.

| Setup | Later update mean | Training frames/s | Projected 4M training |
|---|---:|---:|---:|
| CPU, serial vector | 28.19 s | 145.3 | 7.65 h |
| CPU, subprocess | 17.25 s | 237.5 | 4.68 h |
| CUDA, serial vector, corrected reload check | 18.09 s | 226.5 | 4.91 h |
| CUDA, subprocess, corrected reload check | 7.54 s | 543.4 | 2.05 h |

**Decision:** CUDA/subprocess, eight environments. Peak PyTorch GPU allocation
was about 0.13 GiB; this excludes driver/other-process memory. Host RSS is measured
only for the parent process, not an aggregate worker-memory claim. The two-hour
projection excludes validation/checkpoint overhead and later rollout-length changes.
All attempts cost 73,728 training frames, 8–10 reload-check frames, 102 reset physics
ticks and 18 full optimizer rollouts; campaign wall time 5.41 minutes. Correctness
unit/integration work is separate. The range is an explicit logging limit in the
original failed reload checks, not an estimated exact count.

The [next one-seed native campaign](NATIVE_BASELINE_2026-10-29.md) freezes 4M
aggregate decisions, four scheduled checkpoints and the first 50 development tracks.
Before launching, run two full development episodes twice with the selected
profile checkpoint, solely to validate evaluation/replay. This separate
[readiness declaration](../configs/campaigns/native-evaluation-readiness-2026-10-29.json)
caps new development work at 4,000 frames plus four resets; it cannot select the
checkpoint or alter training parameters. It is not a competence result.

The declared evaluator readiness pass is complete: all two-episode replay fields
and episode metrics match exactly, normalization is frozen, and the pass consumes
exactly two tracks each time. Added cost: 4,000 development frames and four resets.
See the [verification record](../reports/native_profile_2026-10-29/evaluation_readiness.json).
