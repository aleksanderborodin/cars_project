# CarRacing: planning with uncertain road geometry

Research project on when an explicit belief about unseen road geometry helps a
car choose better actions. Scope: **Gymnasium CarRacing only, simulation only,
current workstation first**.

The broad perception–world-model–planning pipeline is established prior work.
Our candidate contribution concerns decision-relevant geometric uncertainty and
planning that respects when future road segments become visible. It remains a
hypothesis: there is **no demonstrated new-method advantage or SOTA result**.

## Documentation index

| Document | Read it for |
|---|---|
| [Experiment specification](docs/EXPERIMENT.md) | Authoritative scope, information contracts, fairness, evaluation rules |
| [Research plan](docs/RESEARCH_PLAN.md) | Honest novelty/feasibility assessment, closest work, proposed method and ablations |
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

Build an information-safe local-geometry interface and a development-only pilot
comparing a competent local controller, the same controller with full geometry,
and a simple statistical road prior. Learn whether hidden geometry materially
changes control before investing in a learned forecaster.

The previous project is preserved on
`archive/pre-belief-research-2026-10-27`. Its obsolete plans and tooling are absent
from the active tree; see the archive guide for recovery. No physical hardware
or additional simulator is required by the current plan.
