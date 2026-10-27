# Agent rules for this repository

Read `docs/EXPERIMENT.md` first: it is the active experiment spec. Start navigation
at `README.md`; use `docs/STATUS.md` to distinguish implemented code from proposals.

## Scope

- Gymnasium CarRacing only, simulation only, current workstation first.
- Main project: modular prediction, internal simulation, online behavior search,
  action execution, and replanning. Read `docs/USER_INTENT.md` for user decisions.
- Predictor-space/hyperparameter selection uses downstream validation driving
  after normal predictive training. Special decision-focused predictor losses are
  not the default. Hidden-road prediction is one component, not the whole project.
- Retain expert/known-track diagnostics as separate privileged experiments.
  Agentic evolution (possibly Rust) is a future option, not the current work path.
- No established novelty, SOTA, safety, or sim-to-real result. Claims require evidence.
- Previous plans and evolution tooling live on the archive branch documented in
  `docs/ARCHIVE.md`. Do not restore them as current instructions or paper claims.

## Implemented layout

```text
src/cars/
  env/        state-only environment, state schema, metrics, existing split helpers
  policies/   privileged Policy interface, pure pursuit, MLP and SB3 policy loaders
  il/         behavior cloning, DAgger, demonstration collection
  rl/         SB3 PPO/SAC and action wrappers
  eval/       privileged evaluator, episode logs, aggregate reports
  training.py training seeds, metadata, run creation
  cli.py      uv run python -m cars.cli <command>
scripts/      training progress utility
tests/        pytest
docs/         active specification, research, references, roadmap, status, development
runs/         outputs and historical artifacts, gitignored
```

The local sensor interface, geometry-belief model, MPC, visual model, and modern
baseline adapters are planned; do not describe them as implemented.

## Rules

- Python 3.12. Run Python with `uv run`; add dependencies with `uv add`.
- Never change official reward, termination, or physics for standard tracks.
  Diagnostic generators and action/observation wrappers must be explicitly named.
- Never read `unseen` or `hard` seeds/results while training or designing methods.
  Do not launch final held-out evaluation before freezing the study.
- Build disjoint train/dev/calibration/test manifests before new paper training;
  current `seen-inf` sampling does not implement those exclusions.
- Local methods must not receive full tracks, raw truth, seeds, global progress,
  hidden-map simulator clones, or future scenario identities. Current APIs are
  privileged and must not be relabeled local.
- Match and record information, history, supervision, action frequency, native
  frames, model rollouts, and compute within each comparison.
- Every run writes config and seeds and must be reproducible from
  `runs/<method>/<run_id>/config.json`. Never overwrite completed runs.
- Same seed and deterministic policy must reproduce the same track and episode
  within the recorded software/hardware configuration.
- Keep modules small and typed; use NumPy for maths. No rendering in state-only
  setting A; visual protocols must enable and document it separately.
- Keep `docs/STATUS.md` and the roadmap consistent with actual code. Treat research
  hypotheses, planned budgets, published findings, and our measurements separately.
- Run `uv run pytest -q` before finishing code changes. Check relevant CLI entry
  points and local documentation links when changing navigation or commands.
