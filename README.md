# CarRacing: prediction, simulation, and control by search

Research project on a modular system that observes, predicts, simulates candidate
behaviors, chooses a control, and replans. Scope: **Gymnasium CarRacing only, simulation only,
current workstation first**.

The broad perception–world-model–planning pipeline is established prior work.
Our goal is competitive actual driving, supported by component-level evidence.
Hidden-road forecasts are one possible component; predictor-space selection by
validation driving is an optional study. There is **no demonstrated new-method
advantage or SOTA result**.

## Documentation index

| Document | Read it for |
|---|---|
| [User intent](docs/USER_INTENT.md) | User decisions, main system, paper goal, expert experiment, and future evolution option |
| [Experiment specification](docs/EXPERIMENT.md) | Authoritative scope, information contracts, fairness, evaluation rules |
| [Geometry research](docs/RESEARCH_PLAN.md) | Literature, feasibility, and detailed design for the geometry/uncertainty branch |
| [Predictor selection research](docs/PREDICTION_SELECTION.md) | Optional prediction spaces, normal training, downstream validation selection, and critiques |
| [Design decisions](docs/DESIGN_DECISIONS.md) | Concrete starting settings, confidence, alternatives, and decision gates |
| [References](docs/REFERENCES.md) | Primary papers, official code, comparison roles and reproduction status |
| [Implementation status](docs/STATUS.md) | What exists, what does not, and evidence boundaries |
| [Roadmap](docs/ROADMAP.md) | Ordered milestones, deliverables, and proceed/stop gates |
| [Development guide](docs/DEVELOPMENT.md) | Current executable APIs, safe development commands, artifact semantics |
| [Archive](docs/ARCHIVE.md) | Previous project snapshot and how to inspect or restore it |
| [Agent instructions](AGENTS.md) | Repository workflow and research-integrity rules |

The specification governs experimental decisions; status describes actual code;
the roadmap orders proposed work. All three must agree before a paper run starts.

## What runs today

The retained foundation implements a state-only CarRacing environment, a
privileged pure-pursuit expert, BC/DAgger, feedforward PPO/SAC, reproducible episode
logs, and evaluation reports. Policies currently receive privileged information.
Local sensors, road forecasting, MPC, visual reconstruction, and modern baseline
adapters have not been implemented.

```bash
uv sync --locked
uv run pytest -q
uv run python -m cars.cli --help
```

See the development guide for seen-side smoke runs. Existing run artifacts are
historical infrastructure results, not evidence for the proposed paper.

## Next milestone

Build the information/data/evaluation foundation and a working modular prediction
and online-search loop. Include an expert/full-map diagnostic and an early
native-input pilot. Improve predictors or add evolution only after this basic
system is working; the latter remains future work.

The previous project is preserved on
`archive/pre-belief-research-2026-10-27`. Its obsolete plans and tooling are absent
from the active tree; see the archive guide for recovery. No physical hardware
or additional simulator is required by the current plan.
