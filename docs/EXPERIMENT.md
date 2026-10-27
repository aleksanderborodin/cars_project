# Experiment design

Compare four ways to get a driving controller for Gymnasium `CarRacing-v3`, with full information about the car and the track (setting A), on seen and unseen tracks.

| Key | Method | Language | Learns from |
|---|---|---|---|
| `base` | Pure pursuit + curvature speed profile (hand-written) | Python | nothing; it is also the IL expert |
| `il_bc`, `il_dagger` | Behaviour cloning, DAgger (MLP) | Python, torch | expert demonstrations on seen tracks |
| `rl_ppo`, `rl_sac` | PPO, SAC (Stable-Baselines3, MLP) | Python | reward on seen tracks |
| `evo` | Codex (gpt-6.1-sol) designs a controller in a loop from eval logs | **Rust** | eval logs on seen tracks |

## Environment

- Base: `gymnasium.make("CarRacing-v3")`, continuous actions `[steer ∈ -1..1, gas ∈ 0..1, brake ∈ 0..1]`, 50 FPS, 1000-step limit, `lap_complete_percent=0.95`. Reward and termination are the official ones, untouched.
- Setting A only: no pixels. A subclass skips pixel rendering in `step` (speed), and wrappers expose the full state.
- Two views of the same state:
  - `CarState` (raw, world frame): car pose and velocities, per-wheel data, current tile index, lap progress. With `TrackInfo` (centre line, tangent angle, half-width per tile), this is what the Rust controllers receive.
  - `StateObs` (engineered, car frame, normalised float32 vector with a named, versioned schema): what the neural policies receive.

## Track splits

| Split | Seeds | Generator |
|---|---|---|
| `seen-K` | `0 … K-1`, K ∈ {1, 10, 100}; `seen-inf` = fresh random seed ≥ 1 000 000 every episode | standard |
| `unseen` | `10000 … 10099` (100 tracks, the standard 100-episode protocol) | standard |
| `hard` | `20000 … 20099` | harder: sharper turns, narrower road, lower grip |

Training and method design may only touch `seen-*`. `unseen` and `hard` are evaluated once per final model.

## Metrics (per episode)

| Metric | Definition |
|---|---|
| `success` | lap completed (env's `lap_complete_percent`) before the 1000-step limit and without leaving the playfield |
| `lap_time_s` | steps to finish / 50; empty if not finished |
| `off_track_events` | "collisions": number of transitions from ≥1 wheel on road to all 4 wheels off road |
| `off_track_steps` | steps with all 4 wheels off road |
| `crash` | left the playfield (official −100) |
| `return` | official episode return; directly comparable with papers |
| `progress` | fraction of tiles visited |

Report: mean ± std over episodes, and IQM with 95% bootstrap CIs over training seeds (rliable). 5 training seeds per learned method.

## Experiments

1. Main table: methods × {seen, unseen, hard} × metrics.
2. Generalisation curve: unseen return vs K (number of seen tracks).
3. Robustness: observation noise, lower friction, one-step actuation delay.
4. Cost: env steps (RL), expert labels (IL), Codex iterations and tokens (evo).

## Codex evolution protocol (`evo`)

- One iteration = Codex edits `rust/carctl` in a fresh copy of the best (or a chosen parent) candidate; the harness builds it and evaluates it on `seen-K` only; the result (metrics + compact log digest of worst episodes) goes into the archive and into the next prompt.
- Codex can run a seen-only evaluation tool inside its iteration. It never receives unseen or hard seeds or results; transcripts are kept for audit.
- Fixed budget of iterations per run (reported as cost). The start candidate is a minimal template, not a port of `base`, so the design is Codex's own.
- Candidates are C-ABI `cdylib`s loaded with `ctypes`, so many candidates coexist and each is evaluated in a subprocess.

## Outputs

`runs/<method>/<run_id>/` holds `config.json`, the model or candidate, and `eval/<split>/` with one gzipped JSON log per episode plus `metrics.csv`. Episode logs carry the track geometry and per-step car trajectory, actions, rewards and events: enough to replay the run in the Kiln game view and to draw all charts.
