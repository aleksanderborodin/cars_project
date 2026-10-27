Implemented MLP/SB3 policy adapters, BC/DAgger/PPO/SAC trainers, shared training utilities, CLI commands, eval-runner integration, and learning tests.

`uv run pytest -q`: **36 passed**, with three upstream SWIG warnings.

Seen-10 smoke results:

| Method | Labels / training steps | Mean return | Finished |
|---|---:|---:|---:|
| BC | 23,996 labels | 764.55 | 7/10 |
| DAgger | 9,422 labels | 920.06 | 10/10 |
| PPO | 52,224 steps | −49.03 | 0/10 |
| SAC | 2,000 steps | −54.38 | 0/10 |

Artifacts are under `runs/<method>/task2_smoke/`. Commands, implementation details, and recommended full-training budgets are in [docs/TASK2.md](/home/sasha/Desktop/cars_project/docs/TASK2.md).

No scope deviations. PPO rounded 50k requested steps to a complete rollout. Environment/metrics semantics remain unchanged. No full training, held-out evaluation, or git commit.