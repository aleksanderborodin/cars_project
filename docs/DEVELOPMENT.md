# Development guide: current executable foundation

[Index](../README.md) · [Specification](EXPERIMENT.md) · [Status](STATUS.md)

This guide describes existing code, not the proposed local-geometry system.
All examples use permitted seen-side data. Do not use the held-out splits during
method development, even though the current CLI exposes them for eventual evaluation.

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
geometry; learned policies use the state vector. Both differ from the proposed
local-input protocol. Ground-truth extraction remains useful for training labels
and evaluator diagnostics once a separate agent-facing interface exists.

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
RL action adapters. It does not establish the new method's performance or test
an unimplemented local interface. Upstream Box2D/SWIG deprecation warnings are
currently expected.

After code changes run the full suite; after documentation/CLI changes check the
referenced commands and local Markdown links. Update [STATUS.md](STATUS.md) with
observed outcomes. Historical source and its original tests are recoverable via
[ARCHIVE.md](ARCHIVE.md).
