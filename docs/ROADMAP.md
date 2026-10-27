# Roadmap and decision gates

Updated 2026-10-27. [Index](../README.md) · [User intent](USER_INTENT.md) ·
[Status](STATUS.md) · [Proposed numerical settings](DESIGN_DECISIONS.md)

The main project is a working modular prediction → internal simulation/search →
action → observation loop. Predictor-space selection, expert variants, and future
agentic evolution must not silently replace this goal. All implementation stages
below remain open. Use permitted seen-side development data; keep unseen/hard
identities and results sealed until the frozen final campaign.

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

1. Implement the agent/evaluator boundary and versioned information contracts.
2. Implement disjoint manifests before new paper training. Add provenance and costs.
3. Verify official behavior, deterministic replay, input isolation, and logging.
4. Fit/adapt a compact internal model and connect a bounded candidate optimizer.
   Existing predictors and planners may be reused; no new predictor theory is needed.
5. Compare with the retained expert and a full-map diagnostic on development data.
   Run a small native-input pilot and profile the entire decision pipeline.
6. Save the working-system report, limitations, and next component priorities.

A–B is roughly a one-to-two-week engineering target, not a completion promise.
Profile actual throughput before fixing the full experimental budget. A geometry
information-gap diagnostic determines whether hidden-road forecasting is useful;
it is not a universal gate for all model-based control.

## Optional experiments

- **Known-track expert:** additional privileged reference and possible teacher.
  Keep its information advantage explicit. See USER_INTENT.md for motivation.
- **Prediction space/model selection:** after normal training, use validation
  racing to select the predictor plus downstream system. Record all tuning and
  controller-training costs. Special task-loss training is not the default.
- **Future agentic evolution:** possibly evolve controller/planner code, including
  Rust, if it fits the system and improves results. The archived implementation
  remains archived. This is not part of A–B and is not currently running.

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
