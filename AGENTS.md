# Agent rules for this repo

Read `docs/EXPERIMENT.md` first: it is the spec.

## Layout

```
src/cars/
  env/        make.py (factory), carracing.py (no-render subclass, hard tracks), state.py (CarState, TrackInfo, StateObs schema),
              metrics.py (episode metrics wrapper), splits.py (seed sets)
  policies/   base.py (Policy protocol), baseline.py (pure pursuit expert), rust.py (ctypes loader for Rust candidates), torch_mlp.py
  il/         bc.py, dagger.py
  rl/         train.py (SB3 PPO / SAC)
  eval/       runner.py (evaluate policy on a split), episode_log.py (log writer/reader), report.py (tables, rliable)
  evolve/     loop.py (Codex loop), digest.py (log digest for prompts)
  cli.py      `python -m cars.cli <command>`
rust/carctl/  Rust controller template (C ABI cdylib). src/policy.rs is the only file the evo loop edits.
tests/        pytest
runs/         outputs, gitignored
```

## Rules

- Python 3.12, run everything with `uv run`. Add dependencies with `uv add`.
- Never change the official reward, termination or physics of standard tracks. The hard generator only changes track and grip constants.
- Never read `unseen` or `hard` seeds or results while training or designing a method.
- Every run writes its config and seed; results must be reproducible from `runs/<method>/<run_id>/config.json`.
- Determinism: the same seed gives the same track and, for a deterministic policy, the same episode.
- Keep modules small and typed; numpy for maths; no pixel rendering in setting A.
- Tests: `uv run pytest -q` must pass before you finish.
