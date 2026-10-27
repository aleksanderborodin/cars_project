# Roadmap and decision gates

Updated 2026-10-27. [Index](../README.md) · [Status](STATUS.md) ·
[Detailed design](RESEARCH_PLAN.md)

The research direction and documentation are established. All implementation
milestones below remain open. Run design/tuning only on permitted seen-side data.
Do not open the existing `unseen`/`hard` seeds or results during these phases.

| Order | Deliverable | Acceptance evidence / decision gate |
|---|---|---|
| A1 | Versioned local geometry observation and policy interfaces | Agent receives only declared sensors/history; hidden geometry changes cannot change identical visible input or immediate action |
| A2 | Train/dev/calibration manifests and data provenance | Disjoint complete tracks; exclusions for fresh-seed sampling; historical training overlap checked without opening held-out results |
| A3 | Protocol and cost metadata; report separation | Runs record sensors, supervision, action repeat, native frames, rollouts, labels, inference/training cost; incompatible protocols cannot silently pool |
| B1 | Fitted short-horizon dynamics and competent local/global deterministic MPC | Rollout errors and latency measured; control works on development tracks |
| B2 | Local/global/simple-prior comparison | Establish a meaningful hidden-information gap and whether a cheap curvature prior closes it |
| C | Compact conditional road-distribution model | Beats deterministic and simple-prior variants in decisions at comparable cost; forecast quality alone is insufficient |
| D | Calibration and future-observation-aware planning | Isolate each contribution with matched data, dynamics, solver and budget; eliminate future-information leakage |
| E | Strong competitor reproductions/adaptations | Recurrent model-free, TD-MPC2, visual DreamerV3, direct Valdi audit; document faithful replication versus adaptation |
| F | Local visual perception and module swaps | Advantage survives estimated geometry and equal-label controls; no initial global-view leak |
| G | Freeze, evaluate, release, write | Frozen endpoints/manifests/models; five training seeds for headline learners; paired final tracks; complete failures/costs/intervals |

## First implementation sprint: A–B

1. Specify local sensor geometry, coordinate frame, range, ego channels, memory,
   reset, action timing, and reward access. Keep evaluator truth outside the
   agent interface. Native pixels remain a separate protocol.
2. Implement manifests and development/calibration selection before collecting
   new training data. Split by track, not by adjacent frames.
3. Test observation isolation, official environment equivalence, deterministic
   replay, and cost accounting. Include seed/reset and reward side channels.
4. Fit a small vehicle model and establish competent local and full-map control
   with the same dynamics/optimizer. Add an empirical road prior.
5. Measure performance against visibility/braking distance, model rollout error,
   and end-to-end latency. Record unsuccessful cases as well as successful laps.
6. Decide whether a learned road belief is warranted. Save the development-only
   pilot report and update the status ledger before expanding experiments.

A–B is roughly a one-to-two-week engineering target, conditional on integration
and control difficulties; it is not a completion promise. GPU memory and actual
throughput must be profiled before setting the final study budget.

## Claim gates

- **No hidden-information gap:** change the visibility/task regime on development
  data or narrow the question; do not build a large forecaster automatically.
- **Simple prior matches learning:** retain the simple baseline and reconsider
  the learned-model contribution.
- **Better forecasts, worse decisions:** investigate calibration, model mismatch,
  and planner exploitation; do not claim success from reconstruction metrics.
- **Gains disappear at matched information/labels/compute:** the method claim fails.
- **Strong learned baselines match or beat us:** report the result and revise the
  mechanism or pursue a carefully scoped explanatory study.
- **Only one artificial diagnostic shows gains:** insufficient evidence for a
  general CarRacing advantage.

## Publication plan

Write a focused problem/related-work/method draft during development, with result
claims left empty. The main evidence should comprise matched task performance
and cost, a mechanism ablation, visibility-dependent behavior, calibration, and
failure cases. One simulator constrains external validity; multiple conditions
within it do not prove real-world transfer.

A standard-CarRacing SOTA claim additionally requires the untouched native-input
protocol and independently audited comparator settings. A custom local-visibility
benchmark can support a strong scientific paper without inheriting that claim.
See [REFERENCES.md](REFERENCES.md) before fixing the comparator roster.
