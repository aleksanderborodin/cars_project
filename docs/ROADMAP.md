# Roadmap and decision gates

**Current baseline gate:** The [native interface/profile](NATIVE_RECIPE_2026-10-29.md) and exact evaluator replay are complete. The [declared one-seed 4M native baseline](NATIVE_BASELINE_2026-10-29.md) is now running as `native_baseline_20261029_s0`, initially toward 0.5M decisions and the fixed first-50-development evaluation. Four checkpoints and the global schedule are frozen. Strong-baseline and independent-seed conclusions await data; a stronger matched-local proposer and its original simulation-search comparison remain subsequent work.

Updated 2026-10-29. [Index](../README.md) · [User intent](USER_INTENT.md) ·
[Status](STATUS.md) · [Proposed numerical settings](DESIGN_DECISIONS.md)

The main project is a working modular prediction → internal simulation/search →
action → observation loop. **Active architecture:** the world model is a programmed
simulator, as clarified by the user. Follow [SIMULATOR_SEARCH_PLAN.md](SIMULATOR_SEARCH_PLAN.md)
for the current P0–P3 architecture and uncertain choices. The
[next-moves plan](NEXT_MOVES.md) specifies the execution order, including baseline
readiness, paired frozen-policy search and broader evaluation. The A–G milestones
below remain the broad scientific roadmap; the former learned-dynamics engine
is retained as a comparison. Predictor-space selection, expert variants, and future
agentic evolution must not silently replace this goal. Initial A/B implementation
and development pilots have run; see the
[latest report](PROGRAMMED_SIMULATOR_RESULTS.md).
Use permitted seen-side development data; keep unseen/hard
identities and results sealed until the frozen final campaign.
Use [RUN_READINESS.md](RUN_READINESS.md) for the two freezes, data/provenance
requirements, pilot gates and consequences of later protocol changes. No new
paper collection or expensive training starts before the infrastructure gates.

**Current execution:** the user requested the intervention study. The first
[four-cell module pilot](INTERVENTION_STUDY.md) now separates visible-reference
reconstruction and hidden-road information with a fixed actor/model/budget.
It precedes new oracle-informed policy training and does not satisfy the strong
baseline, established-optimizer or final-test gates below.
The [completed pilot](INTERVENTION_RESULTS_2026-10-29.md) found worse oracle-cell
driving despite better oracle-road prediction on recorded controls. This motivated
an objective/reference compatibility check before expanding the diagnostic or training
V/L/F policies; do not interpret these substitutions as component adequacy.
The immediate [reference-continuation pilot](REFERENCE_CONTINUATION_2026-10-29.md)
is complete: after exact control replay, changing only the tail raised return
594.21→876.29 and completion 0/5→4/5. The original local mean is still 906.99.
The [20-track paired replication](REFERENCE_REPLICATION_2026-10-29.md) is complete:
true tail 813.85/60%, tangent 527.60/0%, historical local 885.70/70%. The additional15
retain +287.64 versus tangent but one counterexample and a −85.57 local deficit.
Ten exact replays passed; no jobs remain. Close this bounded diagnostic and return
to the audited equal-work optimizer comparison and baseline competence gates.
No automatic 200-track expansion, targeted failure tuning or global training.
The visible prefix, physics and all other planner settings stayed fixed.

The [sequence-search adaptation](SEQUENCE_SEARCH_2026-10-29.md) completed its
matched twenty-track comparison: colored shooting 884.86/70%, iCEM 634.13/15%,
historical knot search 905.94/85%. Five replays and 100 tests passed. Keep the
original search; no automatic larger sweep follows. The next priority is baseline
competence, with [factory/budget](BASELINE_AUDIT.md) and [Valdi source](VALDI_AUDIT.md)
audits now completed. Strong-baseline, independent-seed and trained V/L/F gates
remain open; no experiment jobs remain active from that completed batch.

| Order | Deliverable | Evidence / gate |
|---|---|---|
| A1 | Declared global/local/native information interfaces | Agent receives only permitted inputs; no hidden-state or seed leakage |
| A2 | Disjoint partition manifests and run metadata | Reproducible data, whole-track splitting, and historical training overlap audit |
| A3 | Evaluation and cost accounting | Incompatible information/supervision settings cannot silently pool |
| B1 | Working internal model and online candidate simulation/search | Competent development driving; reproducible rollouts; measured latency |
| B2 | Expert/full-map and local-reference diagnostics | Identify whether failures arise from information, prediction, or control |
| B3 | Early native-input pilot and baseline smoke runs | Understand input adaptation and runtime before assuming transfer |
| C | Complete-system comparison | Strong learned policies and established planners; matched information and budgets |
| D | Focused component ablations | Fixed predictor with different controllers; fixed planner with different predictors |
| E | Optional predictor-space/configuration selection | Normal predictive training; freeze predictors; select downstream systems using validation racing |
| F | Full visual/native comparison and identified failure fixes | Supervision and adaptation documented; actual task performance supports claims |
| G | Freeze, final evaluation, release, paper | No test-driven redesign; independent training runs, paired tracks, uncertainty and failures reported |

## First implementation sprint: A–B

Status on 2026-10-28: disjoint namespaces, local/native interfaces, programmed
vehicle physics, fitted ego dynamics, deterministic behavior search, local/expert references and a recurrent
adapter are implemented and exercised. Basic-loop and reproducibility checks pass.
Competitive control and strong-baseline gates do not yet pass. Multi-world road
beliefs, modern reproductions and the final study are not completed.

1. Implement the agent/evaluator boundary and versioned information contracts.
2. Implement disjoint manifests before new paper training. Add provenance and costs.
3. Verify official behavior, deterministic replay, input isolation, and logging.
4. Fit/adapt a compact internal model and connect a bounded candidate optimizer.
   Existing predictors and planners may be reused; no new predictor theory is needed.
5. Compare with the retained expert and a full-map diagnostic on development data.
   Run a small native-input pilot and profile the entire decision pipeline.
6. Save the working-system report, limitations, and next component priorities.

The [first report](PILOT_2026-10-28.md) supplies that record. The original A–B
sequence above records the implementation order; completed infrastructure should
not be rebuilt. Remaining work follows the budgeted
[next-experiment plan](NEXT_EXPERIMENTS.md):

1. Correct inconsistent unknown-space grip assumptions and rerun the three frozen
   models on the same 20 development tracks.
2. Evaluate fixed controllers from the exact 21-candidate grid and diagnose
   completion failures, including pace and recovery.
3. Run a controlled same-planner full-map diagnostic if needed; improve the
   component implicated by ordinary public driving evidence.
4. Establish a credible recurrent baseline with audited upstream settings and
   explicit local-adaptation versus original-protocol reproduction.
5. Broaden to 200 development tracks and independent trained models, then select
   the focused paper study and required modern comparisons.

Steps 1–2 are complete: 473,737 evaluation frames, no new model training. The
[control audit](CONTROL_AUDIT_2026-10-28.md) reports no established search advantage
and many on-road timeouts. Its 60k-frame pace/full-map proposal is deferred while
we prioritize the user's coded-simulator architecture. The coded-physics comparison
and reduced-cost follow-up are now complete: see
[PROGRAMMED_SIMULATOR_RESULTS.md](PROGRAMMED_SIMULATOR_RESULTS.md). No mean-return
advantage is established. The subsequent full-map diagnostic completed all 20
tracks at both budgets, using the same estimated vehicle state. It also changes
centerline reconstruction, so the information effect is not fully isolated. The
configured driver block completed its 2M review with regression; the selected
1M checkpoint now has a completed positive exploratory policy/search comparison.
Cheap constant corrections did not reproduce the full gain, and a KL-stop training
trial was not better. The [200-track frozen campaign](BROAD_DEVELOPMENT_2026-10-28.md)
is complete; the lower-LR repair also failed. Established optimizers, baseline
verification and the [paper blueprint](PAPER_BLUEPRINT.md) guide the next stage. See [the execution report](FROZEN_POLICY_RESULTS_2026-10-28.md).
User-only ranking diagnostics cannot supply optimization feedback.
A geometry information-gap diagnostic informs one component; it is not a
universal gate for all model-based control.

## Optional experiments

- **Known-track expert:** additional privileged reference and possible teacher.
  Keep its information advantage explicit. See USER_INTENT.md for motivation.
- **Information/control matrix:** future local/full-map learned/search contrasts,
  including local search versus a full-map learned driver; three-module diagnosis
  and precise claim limits are in [INFORMATION_AND_CONTROL_STUDY.md](INFORMATION_AND_CONTROL_STUDY.md).
- **Prediction space/model selection:** after normal training, use validation
  racing to select the predictor plus downstream system. Record all tuning and
  controller-training costs. Special task-loss training is not the default.
- **Future agentic evolution:** possibly evolve controller/planner code, including
  Rust, if it fits the system and improves results. The archived implementation
  remains archived. This is not part of A–B and is not currently running.

- **Task-useful simulator:** future separately implemented/approximate or abstract
  programmed model, potentially GPU-executed and adapted by driving outcomes.
  Realism is optional; compare model selection, implementation speed and agentic
  code adaptation separately. See [intent and design](DECISION_USEFUL_SIMULATOR.md).

## Claim gates

- Gains must survive information, supervision, training and inference-cost controls.
- Better reconstruction alone is not better driving. More search may exploit errors.
- A simple model matching a complex one retires the unnecessary component claim.
- A full-map expert is not an equal-information competitor or proven upper bound.
- Strong baselines winning must be reported and investigated, not removed.
- Native-input failure narrows the claim; a local diagnostic cannot inherit a
  standard-CarRacing SOTA claim.
- Existing MPC/world-model methods already search without a learned actor. The
  paper needs a substantive measured result or method distinction beyond that idea.

Write the problem and related-work draft while developing, leaving unmeasured
result claims empty. Use the working title **Predict, Simulate, Drive: Control
Through World-Model Search** until the evidence supports a more specific title.
