# Native inference-device diagnostic

Pre-execution declaration, 2026-10-29. [Baseline campaign](NATIVE_BASELINE_2026-10-29.md)

**Origin and purpose:** the user requests continued rigorous experiments. The
assistant proposes this bounded technical control because the completed native
baseline trains on CUDA but evaluates on CPU, and reload diagnostics documented
small numerical differences. Chaotic driving can amplify small action changes.
This possibility must be tested before attributing the entire later regression
to learning. It is not evidence that either device is better or a new training
recipe. High confidence in the value of the control; unknown expected effect size.

Freeze the existing 1M, 2M and 4M checkpoints. Reuse the first ten episodes of
their original 50-track CPU evaluations. Add explicit CUDA inference to the
native evaluator, leaving its CPU default and all policy/environment inputs intact.
Use deterministic actions, the same normalizer, frame history, one-action/one-frame
cadence and exact same ten development tracks. Record precision/determinism flags.
No training, checkpoint selection, private ranks or final-test access occurs.

For each checkpoint:

1. Replay the original first two CPU episodes exactly, excluding timings. Stop
   if the default CPU path does not match the original actions/images/rewards.
2. Evaluate CUDA on the first ten development tracks.
3. Repeat CUDA on the first two tracks; require exact same-device episode replay.

New work cap: six CPU replay episodes + 30 CUDA comparison episodes + six CUDA
replay episodes = **42,000 native frames**, plus 42 reset ticks. The original
CPU panel is reused and its old cost is not counted again. Cases run serially.
Do not select a device, change weights, select another checkpoint or expand the
panel from this diagnostic's driving scores. The frozen main selection remains
CPU-evaluated 1M. GPU driving is a separate diagnostic table even if it scores higher.

Report paired CUDA-minus-CPU return and completion changes for every checkpoint,
and the 2M-minus-1M / 4M-minus-1M checkpoint contrasts on each device. Ten tracks
give descriptive evidence only; no universal hardware-equivalence claim or
significance gate is appropriate. Small-device effects support retaining CPU
evaluation; large effects call for a separately declared precision/deployment study.
Neither result establishes the cause of optimization regression or RL insufficiency.

Possible future questions: uninterrupted versus segmented training, independent
training seeds, exploration scale and PPO update size. Logged high approximate KL
and a falling learned SDE scale are correlates, not proven causes. A stronger
matched-local proposer is still the main comparison prerequisite; native results
must not be pooled with local repeat-four/search scores.

Status: complete. All prior native runs and the original 1M checkpoint-selection
decision remain immutable.

## Completed results

All six historical CPU replay episodes and six same-GPU replay episodes match
exactly, excluding timings. The [public summary](../reports/native_device_audit_2026-10-29/summary.json)
verifies 612 frozen source-file hashes; all normalization checks pass.

| Checkpoint | CPU mean, first 10 tracks | CUDA mean, same 10 | CUDA minus CPU | CPU completed | CUDA completed |
|---|---:|---:|---:|---:|---:|
| 1M | 836.51 | 818.25 | −18.26 | 6/10 | 4/10 |
| 2M | 480.55 | 479.76 | −0.79 | 0/10 | 0/10 |
| 4M | 549.89 | 616.92 | +67.03 | 0/10 | 0/10 |

The 2M-minus-1M contrast is −355.96 on CPU and −338.49 on CUDA; 4M-minus-1M
is −286.62 on CPU and −201.33 on CUDA. Later regression therefore persists on
both devices in this panel. Device differences can still matter (particularly
4M's +67.03), so do not claim device equivalence or dismiss precision sensitivity.
These ten-track diagnostics do not replace the original 50-track selection table.
Descriptive paired bootstrap intervals and provenance are included in the public
report; they are not confirmatory tests or training-seed uncertainty.

Cost: **39,214 new native evaluation frames, 42 reset physics ticks, 203.59 seconds**,
zero training/search steps. All nine runs completed; no jobs from this batch remain.
The next [bounded local-policy discount trial](LOCAL_DISCOUNT_TRIAL_2026-10-29.md)
returns to the matched local proposer. Keep native CPU evaluation and the selected
1M policy as frozen evidence; no score-driven device reselection or native sweep.
