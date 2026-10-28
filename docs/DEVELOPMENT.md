# Development guide: research and legacy execution paths

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

After code changes run the full suite; after documentation/CLI changes check the
referenced commands and local Markdown links. Update [STATUS.md](STATUS.md) with
observed outcomes. Historical source and its original tests are recoverable via
[ARCHIVE.md](ARCHIVE.md).
