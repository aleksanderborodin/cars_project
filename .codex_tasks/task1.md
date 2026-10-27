You are implementing task 1 of a research codebase. Read AGENTS.md and docs/EXPERIMENT.md first; they are the spec. The uv project already exists (pyproject.toml, .venv with gymnasium[box2d] 1.4, torch, stable-baselines3, rliable). Rust toolchain: ~/.cargo/bin/cargo.

Task 1 scope (do NOT implement IL, RL training, or the evolution loop yet; leave empty packages for them):

1. src/cars/env/
   - carracing.py: subclass of gymnasium's CarRacing that skips pixel rendering in step()/reset() (setting A, state only) while keeping the official reward, termination and physics exactly. Add a `hard` flag that generates harder tracks (higher TRACK_TURN_RATE, narrower TRACK_WIDTH, lower wheel friction) by temporarily overriding the module constants during track creation; standard mode must be bit-identical to the original env for the same seed (test this: same track, same rewards for the same action sequence vs gym.make("CarRacing-v3")).
   - state.py: TrackInfo (centre xy per tile, tangent angle, half-width, from env.unwrapped.track), CarState (world-frame pose, velocities, yaw rate, per-wheel omega/steer/on-road flag, nearest tile index, progress), and StateObs: an engineered car-frame float32 vector with a named, versioned schema (speed, body-frame velocities, yaw rate, heading error, lateral offset / half-width, wheels on road, ~20 lookahead centre-line points in car frame with curvature, progress, previous action). Expose `OBS_SCHEMA` (list of names) and a Gymnasium wrapper producing it.
   - metrics.py: wrapper computing success, lap_time_s, off_track_events, off_track_steps, crash, return, progress exactly as defined in docs/EXPERIMENT.md.
   - splits.py: seed sets seen-1/10/100, seen-inf, unseen (10000..10099), hard (20000..20099, hard generator).
   - make.py: make_env(split_or_seed, obs="state"|"raw", hard=..., log=...) factory.
2. src/cars/eval/
   - episode_log.py: per-episode gzipped JSON: meta (method, run_id, split, seed, hard), track (centre, left/right edges, half-width), per-step arrays (t, x, y, heading, speed, slip angle, steer, gas, brake, reward, wheels_on_road, tile_idx), events (off_track steps, crash, finish step), summary metrics. Reader too.
   - runner.py: evaluate(policy, split, n_episodes, out_dir, workers) -> DataFrame of metrics; multiprocessing over episodes; writes metrics.csv and logs.
   - report.py: aggregate metrics.csv files into a mean±std table and IQM with bootstrap CIs (rliable).
3. src/cars/policies/
   - base.py: Policy protocol: reset(track: TrackInfo), act(state: CarState, obs: np.ndarray) -> np.ndarray[3].
   - baseline.py: strong hand-written expert using full information: pure pursuit on the centre line (or a smoothed racing line) with speed-adaptive lookahead, a curvature-based speed profile with braking ahead of corners, and slip-aware throttle. Tune it ONLY on seen-10 seeds. Target: mean return >= 900 and success >= 0.95 on seen-10. Report what you reach honestly.
   - rust.py: ctypes loader for a Rust candidate .so (see 4), with build helper (cargo build --release with its own target dir).
4. rust/carctl/: Rust crate, crate-type cdylib, no external dependencies, stable C ABI:
   - `carctl_abi_version() -> u32`, `carctl_new() -> *mut c_void`, `carctl_free(p)`,
   - `carctl_reset(p, track: *const f32, n_tiles: u32, stride: u32)` (per tile: x, y, tangent angle, half-width),
   - `carctl_act(p, state: *const f32, n: u32, out: *mut f32)` (state = flat CarState with a documented field order; out = steer, gas, brake).
   - src/lib.rs holds the ABI glue; src/policy.rs holds a deliberately minimal controller (steer toward the next centre point, constant moderate gas). Only policy.rs will be edited by the evolution loop later, so keep the ABI and the state layout documented at the top of policy.rs.
5. src/cars/cli.py: `uv run python -m cars.cli eval --method base|rust --rust-path ... --split seen-10 --n 10 --out runs/...`, and `build-rust <crate_dir>`.
6. tests/: determinism and bit-identity vs the official env, off-track event counting, log round-trip, Rust template builds/loads/runs one episode, baseline runs one episode.

When done: run `uv run pytest -q`, then evaluate the baseline on seen-10 (10 episodes) and print the metrics table. Do not evaluate on unseen or hard. Do not git commit. End with a short report: files created, test results, baseline numbers on seen-10, and any deviations from the spec.
