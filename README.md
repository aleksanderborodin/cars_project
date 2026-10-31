# CarRacing: prediction, simulation, and control by search

Research project on a modular system that observes, predicts, simulates candidate
behaviors, chooses a control, and replans. Scope: **Gymnasium CarRacing only, simulation only,
current workstation first**.

The active architecture uses a **programmed internal simulator** and selects
behaviors by simulated outcomes. Learned dynamics are retained as pilot baselines.
The key comparison is a frozen behavior proposer with and without simulation
selection; no newly learned behavior scorer is required by the main approach.

The scientific goal separates **understanding the situation** from **choosing
what to do**: identify which components limit driving and test whether simulated
selection among a frozen proposer's sampled behaviors improves its performance.
A future task-useful simulator may be approximate or abstract; realism is not
required if the resulting decisions improve actual driving.

The broad perception–world-model–planning pipeline is established prior work.
Our goal is competitive actual driving, supported by component-level evidence.
Hidden-road forecasts are one possible component; predictor-space selection by
validation driving is an optional study. A frozen-policy development comparison now shows a promising search gain.
There is **no established general advantage, novelty or SOTA result**.

Start with the [paper blueprint](docs/PAPER_BLUEPRINT.md): the consolidated idea
register separates user ideas, assistant refinements, prior work and measured evidence.

## Documentation index

| Document | Read it for |
|---|---|
| [Paper blueprint](docs/PAPER_BLUEPRINT.md) | Central three-stage study, all project ideas with origin/status, validity checks, novelty and execution gates |
| [Native recipe interface and profile](docs/NATIVE_RECIPE_2026-10-29.md) | Separate audited pixel CNN-LSTM interface, parity checks, completed CPU/GPU profile, verification correction and costs |
| [Native baseline campaign](docs/NATIVE_BASELINE_2026-10-29.md) | One-seed 4M decision budget, fixed checkpoint/development selection, completed execution and declared selection |
| [Native baseline results](docs/NATIVE_BASELINE_RESULTS_2026-10-29.md) | Completed 4M curve, selected 1M driver, later regression, 200-trace audit and full costs |
| [Native inference-device audit](docs/NATIVE_DEVICE_AUDIT_2026-10-29.md) | Completed CPU/CUDA control: later regression persists on both devices; 12 exact episode replays |
| [Local discount trial](docs/LOCAL_DISCOUNT_TRIAL_2026-10-29.md) | Completed negative probe: best 691.74/2%, original local retained; all checkpoints, costs and next diagnostic |
| [Offline action-distribution audit](docs/ACTION_DISTRIBUTION_AUDIT_2026-10-30.md) | Completed twelve-checkpoint audit: effective local variation can rise as learned scale falls; exact replay, clipping and update summaries |
| [Local independent-seed replications](docs/LOCAL_REPLICATIONS_2026-10-30.md) | Fixed retained recipe, seeds 1/2, serial frozen execution, 2M review per seed and all checkpoint curves |
| [Paper execution ledger](docs/PAPER_CAMPAIGN_2026-10-30.md) | Required versus optional studies, fixed 3-policy/200-track queue, costs, remaining claim gates and realistic timeline |
| [October31 progress and ETA](docs/PROGRESS_2026-10-31.md) | Verified overnight system sleep, scoped keep-awake guard, current queues and remaining paper work |
| [Storage recovery and revised ETA](docs/CAMPAIGN_RECOVERY_2026-10-30.md) | Disk-full interruption, verified retained panels, unchanged-method restart on the secondary volume and recovery cost accounting |
| [Seed1 search7 loss diagnosis](docs/SEED1_SEARCH7_DIAGNOSIS_2026-10-30.md) | Exact reward accounting, poorer unfinished runs and observed stalls; separates measured effects from unproven causes |
| [Objective-alignment pilot](docs/OBJECTIVE_ALIGNMENT_2026-10-30.md) | Separate score-only method, three penalty settings, two actors, actual-return selection and a dependency-gated twenty-track pilot |
| [Comparator and novelty gap review](docs/COMPARATOR_REVIEW_2026-10-30.md) | Candid current baseline coverage, newly identified MRS.Q/APC-RL prior art, prioritized missing competitors and publication gates |
| [MR.Q/MRS.Q source readiness](docs/MRQ_MRSQ_AUDIT_2026-10-30.md) | Pinned official sources, action/history/replay and stochastic-evaluation constraints; subsequent synthetic checks linked below |
| [External interface and synthetic checks](docs/EXTERNAL_INTERFACE_2026-10-30.md) | Implemented local-array boundary, staged source patches, preserved MR.Q replay failure and passing CPU learner/reload checks; no driving result |
| [MPPI comparison readiness](docs/MPPI_READINESS_2026-10-30.md) | Pinned reference, algorithm identity, synthesized-action state commit and full work accounting before integration |
| [Controlled intervention study](docs/INTERVENTION_STUDY.md) | Executable four-cell geometry diagnostic, isolation tests, pilot budget and staged V/L/F training design |
| [Intervention results](docs/INTERVENTION_RESULTS_2026-10-29.md) | Completed negative driving pilot, executed-trajectory audit, better prediction with oracle road, and next validity gate |
| [20-track reference replication](docs/REFERENCE_REPLICATION_2026-10-29.md) | Completed 20-track paired result: +286.25 versus tangent but −71.86 versus local search; failures, costs and replay evidence |
| [Reference continuation](docs/REFERENCE_CONTINUATION_2026-10-29.md) | Frozen follow-up: preserve the visible reference and change only its future continuation; execution and results |
| [Completed broad results](docs/BROAD_RESULTS_2026-10-29.md) | Full 200-track comparison and failed lower-learning-rate probe; both campaigns finished |
| [User intent](docs/USER_INTENT.md) | User decisions, main system, paper goal, expert experiment, and future evolution option |
| [Task-useful simulator](docs/DECISION_USEFUL_SIMULATOR.md) | Future independent/approximate simulator, downstream adaptation, GPU and agentic options; realism optional |
| [Information and control study](docs/INFORMATION_AND_CONTROL_STUDY.md) | Future local/full-map learned/search matrix, three-module decomposition, user motivation and claim limits |
| [Research log](docs/RESEARCH_LOG.md) | Attributed ideas/questions, intent, context, probes and the user-only ranking-monitor restriction |
| [Experiment specification](docs/EXPERIMENT.md) | Authoritative scope, information contracts, fairness, evaluation rules |
| [Programmed simulator and search](docs/SIMULATOR_SEARCH_PLAN.md) | Active architecture, same-proposer comparisons, execution stages and uncertain choices |
| [Programmed simulator results](docs/PROGRAMMED_SIMULATOR_RESULTS.md) | Completed physics comparisons, reduced-cost search, verification, limitations and next diagnostic |
| [Next moves](docs/NEXT_MOVES.md) | Active execution order: bounded diagnostic, competent frozen driver, paired search, broader comparisons and final freeze |
| [Geometry and stronger driver](docs/GEOMETRY_AND_DRIVER_2026-10-28.md) | Completed full-map diagnostic, 2M training review, selected checkpoint and preceding search implementation |
| [Broad development campaign](docs/BROAD_DEVELOPMENT_2026-10-28.md) | Frozen 200-track declaration, exact replay gates, analysis and historical interim snapshots |
| [Frozen-policy search results](docs/FROZEN_POLICY_RESULTS_2026-10-28.md) | Completed search gain, cheap controls, renderer verification, failed KL repair and next decision |
| [Geometry research](docs/RESEARCH_PLAN.md) | Literature, feasibility, and detailed design for the geometry/uncertainty branch |
| [Predictor selection research](docs/PREDICTION_SELECTION.md) | Optional prediction spaces, normal training, downstream validation selection, and critiques |
| [Design decisions](docs/DESIGN_DECISIONS.md) | Concrete starting settings, confidence, alternatives, and decision gates |
| [Sequence-search comparison](docs/SEQUENCE_SEARCH_2026-10-29.md) | Matched colored shooting/iCEM implementation, declared choices, replay gates, budgets and measurements |
| [Search optimizer audit](docs/SEARCH_OPTIMIZER_AUDIT.md) | Pinned iCEM source, adaptation choices, budget accounting and attribution controls; resolved adaptation linked |
| [References](docs/REFERENCES.md) | Primary papers, official code, comparison roles and reproduction status |
| [Implementation status](docs/STATUS.md) | What exists, what does not, and evidence boundaries |
| [Roadmap](docs/ROADMAP.md) | Ordered milestones, deliverables, and proceed/stop gates |
| [Earlier next-experiment plan](docs/NEXT_EXPERIMENTS.md) | Completed control-audit plan and historical conditional follow-ups; architecture priority superseded |
| [Recurrent baseline audit](docs/BASELINE_AUDIT.md) | Pinned source, current adapter gaps, and prerequisites for credible longer training |
| [Valdi source audit](docs/VALDI_AUDIT.md) | Pinned actor-free planning comparator, pixel-derived sensors, evaluation differences and reproduction prerequisites |
| [Before main runs](docs/RUN_READINESS.md) | What to freeze first, partitions and random streams, readiness gates, and rerun consequences |
| [First development experiment](docs/PILOT_2026-10-28.md) | Implemented system, actual results, limitations and public artifacts |
| [Control audit](docs/CONTROL_AUDIT_2026-10-28.md) | Corrected world query, all 21 fixed candidates, failure patterns and next decision |
| [Development guide](docs/DEVELOPMENT.md) | Current executable APIs, safe development commands, artifact semantics |
| [Archive](docs/ARCHIVE.md) | Previous project snapshot and how to inspect or restore it |
| [Agent instructions](AGENTS.md) | Repository workflow and research-integrity rules |

The specification governs experimental decisions; status describes actual code;
the roadmap orders proposed work. All three must agree before a paper run starts.

## What runs today

The research path implements disjoint seed namespaces, restricted local geometry
and native RGB interfaces, observation-initialized programmed vehicle physics,
recurrent PPO, frozen-policy behavior search and reproducible development reports.
Local search sees a 30 m road mask, ego sensors and permitted history. Hidden road
uses a tangent prior; learned road distributions and visual reconstruction remain
unimplemented. Current evaluation pauses the environment while planning.

The completed 200-track comparison scored **884.48 return / 72% completion** with
21-candidate search versus **862.42 / 35.5%** for the same direct policy. Its p95
latency was 362.11 ms, above the 80 ms action interval. This is one selected actor
on development tracks. The [baseline audit](docs/BASELINE_AUDIT.md) confirms that
its training is an early-stopped local adaptation, not a converged reference or
an original native-pixel reproduction.

The [matched sequence-search experiment](docs/SEQUENCE_SEARCH_2026-10-29.md) is
complete. On twenty development tracks, colored shooting scored **884.86 / 70%**
and the policy-guided iCEM adaptation **634.13 / 15%**, versus historical three-knot
search **905.94 / 85%**. The adaptive-minus-shooting difference is −250.72. Neither
new variant is promoted. All five exact replays and 100 tests passed for that completed batch.

```bash
uv sync --locked
uv run pytest -q
uv run python -m cars.cli --help
uv run python -m cars.cli research --help
```

See the development guide for seen-side smoke runs. New research artifacts are
exploratory development evidence; legacy runs retain their historical protocol.

## Next milestone

Keep the original local three-knot search; the bounded optimizer comparison is
closed with its negative results retained. The next priority is competent baseline
reproduction/adaptation, following the [next-moves plan](docs/NEXT_MOVES.md).
The [paper blueprint](docs/PAPER_BLUEPRINT.md) keeps the user's three-stage
perception/forecasting/action-selection study distinct from what has been measured.
Competent baselines, independent training seeds, trained visual/full-map policies
and final held-out evaluation remain necessary.

The [20-track reference-tail diagnostic](docs/REFERENCE_REPLICATION_2026-10-29.md)
is complete: correcting the scoring continuation improved the oracle-road variant
by 286.25 mean return versus its tangent control, but it remained 71.86 below
historical local search. The original local method remains the working reference.
The [Valdi audit](docs/VALDI_AUDIT.md) records external protocol differences before
reproduction. No modern external baseline has yet been reproduced here.

Bayesian search, independent/GPU simulators and agentic evolution remain future
options. Privileged candidate-ranking values are user-only and excluded from all
agent, training and selection feedback. Final-test data remains unopened.

The previous project is preserved on
`archive/pre-belief-research-2026-10-27`. Its obsolete plans and tooling are absent
from the active tree; see the archive guide for recovery. No physical hardware
or additional simulator is required by the current plan.
