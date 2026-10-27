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
| [Research log](docs/RESEARCH_LOG.md) | Attributed ideas/questions, intent, context, probes and the user-only ranking-monitor restriction |
| [Experiment specification](docs/EXPERIMENT.md) | Authoritative scope, information contracts, fairness, evaluation rules |
| [Geometry research](docs/RESEARCH_PLAN.md) | Literature, feasibility, and detailed design for the geometry/uncertainty branch |
| [Predictor selection research](docs/PREDICTION_SELECTION.md) | Optional prediction spaces, normal training, downstream validation selection, and critiques |
| [Design decisions](docs/DESIGN_DECISIONS.md) | Concrete starting settings, confidence, alternatives, and decision gates |
| [References](docs/REFERENCES.md) | Primary papers, official code, comparison roles and reproduction status |
| [Implementation status](docs/STATUS.md) | What exists, what does not, and evidence boundaries |
| [Roadmap](docs/ROADMAP.md) | Ordered milestones, deliverables, and proceed/stop gates |
| [Before main runs](docs/RUN_READINESS.md) | What to freeze first, partitions and random streams, readiness gates, and rerun consequences |
| [First development experiment](docs/PILOT_2026-10-28.md) | Implemented system, actual results, limitations and public artifacts |
| [Development guide](docs/DEVELOPMENT.md) | Current executable APIs, safe development commands, artifact semantics |
| [Archive](docs/ARCHIVE.md) | Previous project snapshot and how to inspect or restore it |
| [Agent instructions](AGENTS.md) | Repository workflow and research-integrity rules |

The specification governs experimental decisions; status describes actual code;
the roadmap orders proposed work. All three must agree before a paper run starts.

## What runs today

The research path now implements disjoint seed namespaces, restricted local
geometry and native RGB protocols, fitted ego dynamics, finite behavior search,
local references, recurrent PPO, reproducible artifacts and development reporting.
The original privileged expert/BC/DAgger/PPO/SAC path remains separate.
The first development experiment is complete; the revised search has not
established an advantage over its same-family fixed controller.
Learned road distributions, scenario MPC, CEM, local visual reconstruction and
modern world-model reproductions remain open.

```bash
uv sync --locked
uv run pytest -q
uv run python -m cars.cli --help
uv run python -m cars.cli research --help
```

See the development guide for seen-side smoke runs. Existing run artifacts are
historical infrastructure results, not evidence for the proposed paper.

## Next milestone

Improve the working loop's robustness and establish a competitive learned
baseline before the main paper campaign. The initial local/native interfaces,
expert diagnostic and native-input integration pilot have run. See the pilot
report for evidence and open gates. Bayesian search and evolution remain future
options. Privileged candidate-ranking values are user-only and excluded from
all agent/training/selection feedback.

The previous project is preserved on
`archive/pre-belief-research-2026-10-27`. Its obsolete plans and tooling are absent
from the active tree; see the archive guide for recovery. No physical hardware
or additional simulator is required by the current plan.
