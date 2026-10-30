# Local baseline replication: two fresh training seeds

Pre-execution declaration, 2026-10-30. [Action audit](ACTION_DISTRIBUTION_AUDIT_2026-10-30.md)
· [Next moves](NEXT_MOVES.md)

**Origin:** the user wants a credible frozen-proposer/simulated-selection study.
The assistant chooses independent-seed replication after unsuccessful local
training variants and the completed offline audit. Reproducing the retained
recipe quantifies an important uncertainty that another parameter change would
leave unresolved. High confidence that replication is needed; low confidence
that a new seed alone produces a stronger driver. The purpose is evidence about
reproducibility, not selecting a lucky seed.

Freeze the original local recipe. Train **seeds 1 and 2** fresh, serially, on the
existing disjoint train/development namespace. Compared with canonical seed 0,
only the training seed changes: local-v1 30 m mask/ego/previous-action inputs,
repeat four, same encoder, PPO gamma 0.99, reward normalization gamma 0.99, CPU/four
workers, batch/epochs/GAE/SDE and global 8M-native-frame learning-rate schedule.
Optional normalization_gamma=None and target_kl=None preserve the retained recipe.
The original seed 0 result is reused as historical exploratory evidence; it was
used for method development and does not become an independent confirmatory test.

Each seed stops at the **2M-native-frame review**, with complete-rollout cap
2,008,191. Save and evaluate 0.2/0.5/1/2M checkpoints on the same first 50 development
tracks. Select each seed's standalone checkpoint by mean return, then completion,
then earlier step. Publish both full curves regardless of score. Do not drop a
weak seed, change settings between seeds, or extend a successful seed implicitly.
The global 8M schedule does not imply authorization for 8M continuation here.

Total new caps: **4,016,382 training native frames, 400,000 development frames,
16 train-only reload-check frames**, plus recorded reset ticks. No expert/true
future information, calibration, final test or private candidate rankings are used.
Production source, block runner, reporter, manifests and both configs are copied
before either seed begins. Each seed has one queued checkpoint evaluator, matching
the prior execution layout. Separate seeds run serially; failures stop the queue.

Provisional runtime: 25–35 minutes per seed, 50–70 minutes overall, based on prior
local 2M measurements. Preserve artifacts on nonfinite/runtime/reload failures or
free storage below 1.5 GB. No paid compute, new dependencies or hardware is required.

After completion, compare training stability, selected scores and learning curves
across seeds 0/1/2. Any candidate used next must be frozen by standalone selection
before direct-versus-original-programmed-search evaluation, with matched random
and cheap controls. This is an exploratory three-seed stage under 2M review caps;
it is not the proposed full-budget/headline five-seed study, proof of convergence
or a strong-comparator/SOTA result. Stronger actor families and native benchmark
reproductions remain separate requirements.

The [parent manifest](../configs/campaigns/local-replications-2026-10-30.json) pins
source/config hashes and both child declarations. `run_local_replications.py
--check` validates the declared recipe differences without creating a run.
Status: declared; full checks and frozen launch precede training.

Launch observed at `2026-10-30T13:09:49.782403+00:00`. Parent campaign:
`runs/campaigns/local_replications_20261030`; source/launch commit `9611849`.
Both arms use copied source, runner, reporter and configs. Seed 1 starts first;
seed 2 is queued without adapting its recipe from seed 1. This is a launch
record; live status is in the campaign JSON, not inferred from this snapshot.
