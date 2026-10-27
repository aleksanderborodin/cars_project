Implemented task 1: environment/state/metrics, logging/evaluation/reporting, baseline and Rust policies, Rust ABI crate, CLI, tests, and [usage documentation](/home/sasha/Desktop/cars_project/docs/TASK1.md). IL, RL, and evolution packages remain empty.

`uv run pytest -q`: **20 passed**, three dependency deprecation warnings.

Seen-10, 10 episodes:

| Metric | Mean ± std |
|---|---:|
| Return | **921.18 ± 9.26** |
| Success | **100%** |
| Lap time | 15.764 ± 1.853 s |
| Off-track events / steps | 0 / 0 |
| Crashes | 0 |
| Progress | 100% |

Return IQM: **921.52**, 95% episode-bootstrap CI **[914.72, 927.58]**.

Saved [metrics](/home/sasha/Desktop/cars_project/runs/base/task1_seen10/eval/seen-10/metrics.csv), [config](/home/sasha/Desktop/cars_project/runs/base/task1_seen10/config.json), logs, and [report](/home/sasha/Desktop/cars_project/runs/base/task1_seen10/report.csv).

Compatibility adjustment: pinned pandas below 3 for `rliable`/`arch`. Hard grip uses upstream tile friction. Unseen and hard were not evaluated. No commit made.