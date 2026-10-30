# Development guide: research and legacy execution paths

## Separate objective-alignment pilot

The [declared pilot](OBJECTIVE_ALIGNMENT_2026-10-30.md) adds an explicit
`policy-objective` method. Original methods/default configs remain unchanged;
`--penalty-scale` is required for the new method and rejected elsewhere.

```bash
uv run python -m cars.cli research evaluate --help
uv run python scripts/run_objective_alignment.py --manifest configs/campaigns/objective-alignment-2026-10-30.json --check
```

The queued controller uses copied source and waits for the current matrix. It
then checks saved direct/legacy-search traces and runs all declared score settings
serially. Do not start competing manual runs. Its report requires every actor/arm,
retains negative effects and cannot promote a new default automatically. The
score remains an approximate local proxy; official reward/termination are unchanged.

## Frozen search recovery

The [recovery guide](CAMPAIGN_RECOVERY_2026-10-30.md) records the disk-full
interruption, retained results and new mounted-volume workspace. Validate its
existing frozen declaration without running an environment:

```bash
uv run python scripts/recover_paper_search.py --campaign /media/sasha/72AECF44AECEFF99/cars_project_recovery_20261030/queue --check
```

Do not start a second controller over an active recovery. The new live status is
in that workspace; the original queue status is historical. New runs have recovery
suffixes, exact prefix gates and explicit failed-attempt cost bounds. No completed
run is overwritten. Temporary/cache outputs also use the secondary volume.

## External algorithm source staging

The [external interface and synthetic verification](EXTERNAL_INTERFACE_2026-10-30.md)
provides the restricted local vector/action boundary, explicit source patches and
RNG isolation. This is feedforward local preflight, not a native-pixel or driving
reproduction. The stager verifies selected cached official bytes against the public
source audit, retains licenses, refuses overwrites and imports no external code.

```bash
uv run python scripts/stage_external_baseline.py --help
uv run python scripts/stage_external_baseline.py --algorithm MRQ --output runs/external_sources/NEW_MRQ_STAGE
uv run python scripts/stage_external_baseline.py --algorithm MRSQ --output runs/external_sources/NEW_MRSQ_STAGE
```

Use fresh output directories. This command does not start training. Small CPU
synthetic learner/reload checks passed after a declared MR.Q priority-initialization
fix; full training/evaluation, replay continuation/ring-wrap and GPU checks remain.

## Separate native recipe commands

The [native interface/profile](NATIVE_RECIPE_2026-10-29.md) uses official images,
resize64/grayscale/stack2, repeat one and standard CnnLstmPolicy. This is distinct
from the older `--view native` RGB/previous-action dictionary adapter.

```bash
uv run python scripts/profile_native_training.py --manifest configs/campaigns/native-profile-2026-10-29.json
uv run python scripts/run_native_baseline.py --manifest configs/campaigns/native-baseline-2026-10-29.json
uv run python -m cars.cli research train-native --help
uv run python -m cars.cli research evaluate-native --help
```

`train-native --config CONFIG --stop-at N --run-id ID [--resume CHECKPOINT]`
uses aggregate one-frame decisions and completes the last PPO update. Full resolved
configuration, source snapshot, optimizer, RNG, stream and normalization state are
saved. `evaluate-native --model CHECKPOINT --episodes N --run-id ID` uses only the
ordered development manifest, original reward and frozen normalization. The final
VecEnv autoreset is a non-actionable placeholder so no extra reserved track is
consumed. Neither command opens final-test data. Never overwrite run IDs.

[Index](../README.md) · [Specification](EXPERIMENT.md) · [Status](STATUS.md)

The restricted research path is now implemented. The original privileged path
is retained separately below. Do not use held-out splits during development.

## Restricted research commands

```bash
uv run python -m cars.cli research --help
uv run python -m cars.cli research reserve
uv run python -m cars.cli research collect --episodes 2 --run-id demo_train
uv run python -m cars.cli research collect --partition development --episodes 2 --run-id demo_dev
uv run python -m cars.cli research fit --training runs/collection/demo_train --development runs/collection/demo_dev --run-id demo_model
uv run python -m cars.cli research evaluate --method search --model runs/dynamics/demo_model --episodes 4 --run-id demo_search
uv run python -m cars.cli research evaluate --method local --episodes 4 --run-id demo_local
uv run python -m cars.cli research evaluate --method fixed --speed 60 --gain 1.2 --episodes 4 --run-id demo_fixed
uv run python -m cars.cli research evaluate --method physics --speed-grid 20 25 30 35 40 45 50 --episodes 4 --run-id demo_physics
uv run python -m cars.cli research evaluate --method physics --speed-grid 30 40 50 60 70 80 90 --gain-grid 1.2 --episodes 4 --run-id demo_physics7
uv run python -m cars.cli research evaluate --method physics-global --gain-grid 1.2 --episodes 4 --run-id demo_global_physics7
```

Each command creates an immutable run with config, source snapshot, dependency
versions/hashes, costs and relevant datasets/checkpoints/traces. Use fresh IDs.
`collect --collector mixed` cycles local and fixed-speed collection policies.
`evaluate --speed-grid ... --gain-grid ...` changes the search/physics candidate
set; `--horizon` changes its
decision horizon. Fixed controllers accept `--speed` and `--gain`. These
development choices are recorded with each result. Search configurations now
record `road_query=shared_road_query_v2`; old query results retain their original
interpretation. New trajectory files contain post-episode evaluator-only
`evaluation_offroad_frame_indices`, never an agent observation.

`--method physics` uses programmed vehicle equations and Box2D in independent
observation-initialized worlds; it needs no fitted model artifact. Current
prototype contact/state approximations are declared in its config. It records
`internal_physics_frames` separately from native evaluation frames and candidate
decision transitions. `--method search --model ...` remains the fitted-dynamics
baseline. The private candidate monitor currently supports that older fitted
model path only; do not bypass its restrictions for a new engine.

`physics-global` uses a separate `global` protocol ID. It supplies full ordered
road geometry with the same ego sensors and estimated wheel state; it changes
path reconstruction and belongs in a privileged diagnostic table. This protocol
is restricted to development evaluation and cannot collect training data.

The new configured trainer is separate from the legacy engineering trainer below:

```bash
uv run python -m cars.cli research train-driver --help
uv run python -m cars.cli research train-driver --config configs/training/recurrent-local-v2-s0.json --stop-at 8192 --run-id demo_configured_driver
uv run python -m cars.cli research evaluate --method policy-search --model runs/driver_training/demo_configured_driver --candidates 7 --horizon 20 --episodes 4 --run-id demo_policy_search
uv run python -m cars.cli research evaluate --method policy-random --model runs/driver_training/demo_configured_driver --candidates 7 --planning-seed 0 --episodes 4 --run-id demo_policy_random
uv run python -m cars.cli research evaluate --method policy-search --model runs/driver_training/demo_configured_driver --candidates 7 --observation-renderer sparse --episodes 4 --run-id demo_sparse_search
uv run python -m cars.cli research evaluate --method policy-bias --model runs/driver_training/demo_configured_driver --action-bias 0 0.03 0 --episodes 4 --run-id demo_fixed_bias
```

`--stop-at` is a cumulative native-frame boundary, rounded up to a completed PPO
rollout/update. The config's `total_native_frames` controls the global schedule.
Use `--resume` with an immutable completed run/checkpoint and identical resolved
configuration. Reward normalization is learning-only; actor observations use
fixed scales. Intermediate checkpoints include model/optimizer/normalizer/RNG/
sampler state and the original source snapshot. A resume begins fresh episodes.
The optional config field `target_kl` enables upstream minibatch early stopping;
missing/`null` means the earlier behavior. Changing it requires a separately
declared trial, not a silent resume. Trainer-v3 logs initial parameter fingerprints,
actual epochs entered and cumulative optimizer steps. The tested 0.02 trial was
worse and is not a recommended replacement for the original actor.

Policy search freezes the loaded actor and branches its recurrent memory only in
imagined rollouts. Candidate zero is the original actor; three-knot residuals
vary its controls. Random selection uses the same proposal rule without physics.
Neither path supports the private oracle monitor. New cost records distinguish
actor forward examples, internal physics frames and actual environment frames.
`policy-intervention` is a separate privileged development method around that
same local actor. `--reference oracle_visible` replaces only the objective path
with the clipped visible directed centreline; `--continuation oracle` replaces
only hidden-road occupancy/margins. Both default to the original reconstructed
reference/prior, which must replay baseline episodes exactly. The adapter removes
map fields before any real or imagined actor call. The outside-footprint penalty
and observation-based vehicle observer stay fixed. See the
[controlled intervention protocol](INTERVENTION_STUDY.md) for the declared
four-cell pilot, costs, caveats and staged V/L/F policy study.
`--reference oracle_continued` adds the separately declared true-road reference
tail, preserving the visible samples and the same 100 m/one-metre extension.
It remains a privileged development method; see the
[follow-up protocol](REFERENCE_CONTINUATION_2026-10-29.md).
The optional sparse renderer skips unnecessary occupancy work; reference remains
the default. Its parity evidence and latency limits are in the latest report.
`policy-bias` adds a declared constant offset to each real actor action without
simulation; declare any development fitting of that offset as additional tuning.
The [execution report](GEOMETRY_AND_DRIVER_2026-10-28.md) indexes declarations,
completed diagnostics and current campaign status. The first 2M block has stopped
at its health review; an 8M schedule declaration does not mean 8M frames ran.
`scripts/run_frozen_driver.py` executes the declared six-condition comparison and
automatically builds its public report after all conditions finish. Its status
file is `runs/campaigns/frozen_driver_20261028/status.json`. Do not rerun the runner
against an existing campaign ID; inspect status and preserve partial/failed runs.
That campaign and its subsequent fixed-bias/KL/renderer checks are now complete;
see [FROZEN_POLICY_RESULTS_2026-10-28.md](FROZEN_POLICY_RESULTS_2026-10-28.md).
`run_driver_block.py --manifest PATH` supports separately declared fresh campaigns
with unique training/evaluation IDs. Completed source snapshots remain authoritative.

```bash
uv run python -m cars.cli research train-recurrent --frames 2048 --workers 2 --run-id demo_recurrent
uv run python -m cars.cli research train-recurrent --frames 2048 --workers 2 --resume runs/recurrent_training/demo_recurrent --run-id demo_recurrent_continued
uv run python -m cars.cli research evaluate --method recurrent --model runs/recurrent_training/demo_recurrent_continued --episodes 4 --run-id demo_recurrent_eval
```

Frame budgets are native-frame targets; PPO completes whole rollouts and records
actual overshoot. Continuation preserves weights/optimizer/RNG/sampler state but
starts fresh episodes; it is not exact mid-episode Box2D continuation. Worker count,
seed and protocol must match. Older checkpoints lacking RNG snapshots are rejected
for continuation. `--view native` selects official RGB plus previous actions only.

`evaluate --user-monitor` exports frozen search candidates to a separate process
on a fixed sparse development schedule. Its real outputs under `private_diagnostics/`
are user-only: never open or feed them into agent/training/tuning/selection analysis.
Only public technical status/costs under `runs/user_monitor/` may be inspected.
The monitor uses extra simulator queries and may contend for CPU; keep its timing
separate from ordinary inference profiling.

The first campaign and public artifact rebuild command are in
[PILOT_2026-10-28.md](PILOT_2026-10-28.md). The completed follow-up's manifest,
runner and public analysis commands are in
[CONTROL_AUDIT_2026-10-28.md](CONTROL_AUDIT_2026-10-28.md). Campaign runners create
fresh immutable directories and reject overwrites. The completed coded-physics
campaigns and public rebuild commands are in
[PROGRAMMED_SIMULATOR_RESULTS.md](PROGRAMMED_SIMULATOR_RESULTS.md). No final-test command is exposed by the
research CLI. The rest of this guide describes the **legacy privileged path**.

## Environment and checks

Use Python 3.12 and the committed lockfile:

```bash
uv sync --locked
uv run pytest -q
uv run python -m cars.cli --help
uv run python -m cars.cli train-rl --help
uv run python -m cars.eval.report --help
```

Use `uv add` to add packages, and record any upstream baseline revision and its
adapter changes. Baselines requiring incompatible dependencies should have
separate locked environments. Do not silently update all dependencies.

## Current observation and policy APIs

`cars.env.make.make_env(split_or_seed, obs="state")` returns StateObs v1, a
74-dimensional privileged vector. `obs="raw"` returns raw state. The environment
is state-only: neither mode is a pixel or local sensor interface.

`Policy.reset(track: TrackInfo)` receives the full track. `Policy.act(state, obs)`
receives raw `CarState` plus the vector observation. The expert uses this privileged
geometry; learned policies use the state vector. Both differ from the implemented
restricted local-input protocol. Ground-truth extraction remains useful for
declared training labels and evaluator diagnostics behind the separate interface.

The standard factory uses `CarRacing-v3` behavior, `lap_complete_percent=0.95`, a
1000-frame time limit, and 50 Hz physics. Action-repeat and the optional two-axis
throttle adapter sit outside the per-frame environment. The default official
controls are steering, gas, and brake. Changes to action frequency or action
parameterization belong in separately matched comparison conditions.

## Development smoke runs

These examples are small infrastructure checks, not paper training budgets or
claims of convergence. Choose fresh run IDs; completed artifacts are immutable.

```bash
uv run python -m cars.cli eval --method base --split seen-1 --n 1 --out runs/base/dev_smoke
uv run python -m cars.cli train-bc --split seen-1 --seed 0 --episodes 1 --epochs 1 --run-id dev_smoke
uv run python -m cars.cli eval --method il_bc --model-path runs/il_bc/dev_smoke --split seen-1 --n 1 --out runs/il_bc/dev_smoke
```

`train-dagger` and `train-rl --algo ppo|sac` are also implemented. Consult their
`--help` for settings. RL `--steps` is an SB3 decision budget; native physics frames
can differ with action repeat. Use recorded costs for actual native-frame counts.
RL `--episodes` specifies a nominal steps budget, not an exact episode count.

Do not launch new paper training with `seen-inf` until partition exclusions are
implemented. Existing training evaluation is seen-side monitoring, not the new
independent development/calibration scheme.

For a live training run:

```bash
uv run python scripts/progress.py runs/rl_ppo/dev_example
```

The utility sums completed-episode monitor lengths and prints recent training
returns. It is not a held-out evaluator, and incomplete episodes are absent.

## Artifacts and reporting

An evaluation writes `config.json`, `eval/<split>/metrics.csv`, and compressed
per-episode logs. A training run includes its config, method-specific model and
normalization, and cost records. Evaluation inside its training directory adds
an evaluation-specific config without replacing the training config.

Logs contain true track geometry and state for analysis. They must not become
hidden input channels to future local policies. Preserve failed episodes and
record dependencies, source hashes, seeds, and configuration with each run.

```bash
uv run python -m cars.eval.report runs/base/dev_smoke/eval/seen-1/metrics.csv --out runs/base/dev_smoke/report.csv
```

The current reporter groups by method/split. Use only homogeneous protocols and
settings; it cannot yet enforce the new grouping requirements. With multiple
training seeds it bootstraps seed-level metric means; its IQM is then over those
means. A single run uses episode-level uncertainty instead. This is not a
multi-task benchmark IQM or a substitute for training-seed uncertainty.

Metric interpretation:

- `return`: unchanged official reward accumulated over agent steps.
- `success`: lap finished strictly before frame 1000 and no playfield exit.
- `lap_time_s`: finish frame divided by 50, missing when no lap finishes; a finish
  at frame 1000 has a lap time but does not satisfy the current success definition.
- `off_track_events`: transitions from some wheel contact to no wheel contact.
- `off_track_steps`: frames with no wheel on road.
- `crash`: playfield exit, not a physical collision count.
- `progress`: fraction of visited road tiles, evaluator-only truth for local protocols.

Always report completion alongside lap time. Comparing only completed laps can
favor a method that fails on difficult tracks.

## Validation and maintenance

The test suite covers environment equivalence, deterministic replay, metrics/log
round trips, multiprocessing evaluation, seed handling, learning smoke runs, and
RL action adapters, restricted local/native interfaces, and shared road-query
semantics. It does not establish the new method's performance. Upstream Box2D/SWIG deprecation warnings are
currently expected.
Default discovery targets `tests/` only. Saved experiment snapshots and private
diagnostic directories are excluded; do not restore recursive root discovery.

After code changes run the full suite; after documentation/CLI changes check the
referenced commands and local Markdown links. Update [STATUS.md](STATUS.md) with
observed outcomes. Historical source and its original tests are recoverable via
[ARCHIVE.md](ARCHIVE.md).

## Frozen broad-development campaign

The declared comparison and queued lower-LR probe are complete. See the
[completed review](BROAD_RESULTS_2026-10-29.md). Commands below document reproduction
and orchestration; their original run IDs remain immutable.

```bash
uv run python scripts/run_broad_driver.py
uv run python scripts/summarize_broad_driver.py --campaign runs/campaigns/broad_driver_20261028 --output reports/broad_driver_2026-10-28
```

The declared campaign has already been launched; its fixed IDs refuse overwriting.
Read `runs/campaigns/broad_driver_20261028/status.json` for technical progress.
The runner copies production source and the reporter before starting child runs;
children import that verified copy through `PYTHONPATH`. Every condition must
replay the original first 20 episodes exactly. The reporter refuses incomplete
or incompatible data. See [the declaration](BROAD_DEVELOPMENT_2026-10-28.md).

The independent baseline probe is already queued with:

```bash
uv run python scripts/run_queued_driver.py --manifest configs/campaigns/driver-lr3e5-block-2026-10-28.json
```

Do not launch it twice: queue/run paths refuse overwriting. The queue copies
source, tests, configuration, runner and reporter, then waits for the completed
broad campaign. It serializes regression tests and training after that campaign.
Public queue status is `runs/queues/driver_lr3e5_block_20261028/status.json`.
`run_driver_block.py --reporter PATH` supports a copied reporter; its default
behavior is unchanged. A failed/stale predecessor stops the queue for review.

## Local discount probe and explicit inference device

`research evaluate-native --device cpu|cuda` records the execution device and
precision/determinism settings. Default CPU inference remains unchanged. Hardware
comparisons use the [declared diagnostic](NATIVE_DEVICE_AUDIT_2026-10-29.md), not
score-based device selection.

`DriverTrainingConfig.normalization_gamma=None` preserves the existing rule that
reward-normalization gamma follows PPO gamma. Set it explicitly to keep that
normalization timescale fixed while testing another policy discount. The
[local probe](LOCAL_DISCOUNT_TRIAL_2026-10-29.md) uses PPO 0.96059601 and
normalization 0.99. Launch its immutable declaration with:

```bash
uv run python scripts/run_driver_block.py --manifest configs/campaigns/local-discount-2026-10-29.json
```

Run IDs must be new; the existing campaign is complete and must not be overwritten. Its status and
checkpoint results live in `runs/campaigns/local_discount_20261029_s0/`.

## Offline action audit and independent-seed replication

The [offline audit](ACTION_DISTRIBUTION_AUDIT_2026-10-30.md) has completed. Its
runner accepts only the declared recorded actor-input fields, checks own-history
controls and API parity, and never creates a simulator. Analytic clipping moments
are per-history marginals; gSDE temporal correlation requires a separate study.

The active [local replication](LOCAL_REPLICATIONS_2026-10-30.md) can be validated
without launching or creating runs:

```bash
uv run python scripts/run_local_replications.py --manifest configs/campaigns/local-replications-2026-10-30.json --check
```

The same command without `--check` launches a new immutable campaign. The current
campaign already exists and must not be overwritten. Both seeds' source/configs
are frozen before either starts. Status: `runs/campaigns/local_replications_20261030/status.json`;
child checkpoint curves live in each seed's child campaign and report directory.

## Queued paper development comparison

The [paper execution ledger](PAPER_CAMPAIGN_2026-10-30.md) records required gates,
optional branches, budgets and timeline estimates. Its corrected queue is already
prepared and waiting for the seed replications. Validate its immutable declaration:

```bash
uv run python scripts/run_paper_search_queue.py --manifest configs/campaigns/paper-search-sparse-queue-2026-10-30.json --check
```

Without `--check` the command prepares a new immutable queue; the current ID
already exists. Production code, reporter and configs are copied before execution.
The initial reference-renderer preparation was stopped before any new episodes;
its records are retained. Use the corrected sparse declaration above. Do not run
other project training concurrently with the queued measurement phase.
