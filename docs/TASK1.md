# Task 1: state-only driving and evaluation

Use Python 3.12 and `uv run`. IL, RL, and evolution packages are empty placeholders.

```bash
uv run pytest -q
uv run python -m cars.cli eval --method base --split seen-10 --n 10 --workers 2 --out runs/base/example
uv run python -m cars.cli build-rust rust/carctl
uv run python -m cars.cli eval --method rust --rust-path /path/to/libcarctl.so --split seen-10 --n 10 --out runs/rust/example
uv run python -m cars.eval.report runs/base/example/eval/seen-10/metrics.csv --out runs/base/example/report.csv
```

The output argument is the **run root**, `runs/<method>/<run_id>`. Each run has
`config.json`, `eval/<split>/metrics.csv`, and one gzipped JSON log per episode.
Rust runs also preserve `candidate.so`. Existing run configs cannot be overwritten.
The config contains the policy parameters, episode seeds, environment settings,
observation schemas, package versions, and Python source hashes. Rust configs
include the candidate hash. Preserve the checkout and uv.lock with results.

To reproduce a baseline run with a saved config:

```python
import json
from cars.eval.runner import evaluate
from cars.policies.baseline import BaselineConfig, BaselinePolicy

config = json.load(open("runs/base/example/config.json"))
evaluate(BaselinePolicy(BaselineConfig(**config["policy"])),
         config["split"], config["n_episodes"], "runs/base/reproduction",
         workers=config["workers"], evaluation_seed=config["evaluation_seed"])
```

## Environment and state contracts

`make_env(seed_or_split, obs="state"|"raw", hard=None, log=None)` returns a
Gymnasium environment with the official 1000-step limit and episode metrics in
terminal `info["episode_metrics"]`. Split resets cycle finite seeds; seen-inf
uses a reproducible sequence of distinct seeds >= 1,000,000. Explicit reset seeds
are supported. The evaluation harness records every resolved seed.

`StateCarRacing` inherits upstream `reset` and `step` and overrides only their
internal `_render("state_pixels")` hook to extract state. It does no pixel work.
Tests compare exact geometry, physical state, rewards, termination, and truncation
with the official environment, including a complete successful lap.

Hard track creation temporarily multiplies TRACK_TURN_RATE by 1.5 and TRACK_WIDTH
by 0.75, restoring globals even on failure. Hard road tiles retain road_friction
0.75, which reduces wheel grip through the official friction calculation; grass
friction stays upstream's 0.6. Track creation through this subclass is locked;
concurrent upstream track creation in the same process during a hard reset is
unsupported. Evaluation workers are separate processes. No held-out track needs
to be generated to test the constant restoration mechanism.

`CarState` uses world metres and seconds, heading in radians from world +x,
positive counterclockwise. Car-frame coordinates are forward/left. A raw
observation is the version-1, 20-float layout `STATE_FIELDS`. Rust receives exactly
this vector, not StateObs. `TrackInfo.as_array()` is contiguous float32 with tile
rows `(x, y, tangent_heading, half_width)`. The same layouts are documented at
the top of `rust/carctl/src/policy.rs`.

`StateObs` version 1 has 74 float32 entries, named by `OBS_SCHEMA`. It includes
20 centre-line points spaced two tiles apart, including signed curvature. Scales
are encoded in the names; values are normalised by fixed physical scales but not
clipped. Heading error is wrapped to [-pi, pi). Lateral offset is positive left
of the nearest centre point. The final three entries are the preceding action
(the action that produced the current observation), zero after reset.

Metrics count only agent steps, excluding upstream reset's initial physics tick.
Success requires official lap completion at step < 1000 without leaving the
playfield. A finish at step 1000 has a lap time but is not a success. A crash is
identified by the official playfield bounds, rather than guessing from reward.
Progress is visited-tile fraction; nearest tile index is purely geometric.

Logs describe post-action state; `t` is elapsed agent time (step / 50). The initial
raw state is stored separately. Event step indices start at 1. Gzip timestamps
are fixed, so deterministic log data produces identical compressed bytes.

## Evaluation and reporting

The policy protocol is `reset(track)` followed by `act(CarState, StateObs)`.
Spawn workers isolate Rust candidate instances, and worker count does not change
episode seeds or outcomes. Candidate libraries are not sandboxed; ABI glue handles
null/short inputs and catches Rust panics. A candidate causing a native fault can
still kill its worker. Build helpers give each build its own target directory.

The report gives episode means and sample standard deviations. With several
training seeds (a `training_seed` CSV column), it computes per-training-seed means
and bootstraps their IQM using rliable. Otherwise multiple run_ids are the units.
With one baseline run it reports an explicitly labelled episode bootstrap;
this cannot estimate uncertainty across training seeds. Missing lap times are
excluded, so lap-time statistics are conditional on finishing. rliable 1.2 uses
global NumPy sampling internally; report generation saves/seeds/restores that
state to keep CIs reproducible. pandas is pinned below 3 because rliable's
arch<8 dependency is incompatible with pandas 3's deprecation API.

## Seen-only baseline tuning

The expert uses pure pursuit with speed-adaptive lookahead, a curvature-limited
speed profile propagated backwards for braking, and rear-wheel spin/lateral-slip
throttle reduction. All tuning used seen-10, seeds 0 through 9.

Saved exploratory runs in `runs/base/tune_accel_*` tested lateral-acceleration
budgets 60, 90, 120, 150, 180, 220. The selected budget 180, speed cap 100, and
braking budget 50 completed all 10 tracks; 220 failed one. These budgets are
controller planning parameters in environment units, not changes to physics.
The final measured numbers are in `runs/base/task1_seen10/eval/seen-10/metrics.csv`.
Neither unseen nor hard was evaluated or used for tuning.
