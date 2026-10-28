# Before expensive runs

Updated 2026-10-28. [Index](../README.md) · [Specification](EXPERIMENT.md) ·
[Status](STATUS.md) · [Design defaults](DESIGN_DECISIONS.md)

The user's priority is to make experiments reusable and avoid preventable reruns.
This plan was written before implementation. A first implementation/development
experiment has since run; see [PILOT_2026-10-28.md](PILOT_2026-10-28.md) for which
gates have evidence. Final-test and legacy unseen/hard data remain sealed.

## Two freezes

1. **Before data collection:** freeze the versioned environment, information and
   action contracts, partition rules/membership, metrics, logging and accounting.
2. **After development pilots:** freeze the method configurations, selected
   checkpoints, comparator roster, budgets, sample sizes, analysis and complete
   condition list before opening any final test results.

A seed reproduces a track only with the recorded generator, environment arguments
and software versions. Train/validation/test use different tracks from the same
standard generator, not three different simulators. New-track performance does
not establish transfer to different physics or generators.

Prediction space, forecast distance, model size, search horizon and risk settings
remain development choices. An infrastructure change that alters the scientific
task creates a new protocol version; do not silently replace earlier results.

## Partition and randomization plan

| Partition | Proposed starting size/rule | Purpose |
|---|---|---|
| Training | Reproducible fresh-track stream restricted to training partition | Fit models, normalization and policies; collect labels/demonstrations |
| Development / validation | 200 fixed standard tracks | Debugging, checkpoint/configuration selection, baseline tuning, budget and precision pilots |
| Calibration reserve | 200 other fixed standard tracks | Fit a declared uncertainty-calibration procedure, if used |
| Final standard test | Target 500 fixed tracks per model/primary condition | Frozen final campaign only |
| Legacy unseen/hard | Existing sealed sets, separate from new partitions | Separate frozen evaluation if included; no method design access |

The train stream mechanism and 200-track development/calibration manifests are now
implemented; the final test commitment remains sealed. Counts are not statistical
guarantees and final counts remain subject to the study freeze. If development
variance justifies a larger test, extend it by a
predeclared random rule before the study freeze, without screening test tracks or
scores. Calibration is optional: reserving tracks does not require running that
experiment. Do not create several additional holdouts without a specific purpose.

Required implementation:

- Split by **whole track/family**, never adjacent frames or laps. Sensor views,
  demonstrations, counterfactual branches and derived labels inherit their source
  track's partition. Related custom diagnostic variants stay together.
- Train, development and calibration are disjoint even though they all belong
  to the broadly permitted development side. Use independent development episodes
  to select checkpoints, not episodes used to fit the learner.
- Audit historical training/development provenance only. If old collection seeds
  cannot be reconstructed, label their exclusion status unverified and do not
  certify old artifacts as clean paper data. A fresh seed namespace helps only
  when its disjointness from historical sampling can actually be established.
- Enforce exclusions in RL, demonstrations, supervised map generation, dynamics
  collection, diagnostic branching and evaluation callbacks. A low probability
  of collision or a documentation warning is insufficient.
- A partition tool may check sealed identities internally and report counts,
  hashes and overlap failures; it must not expose held-out identities, render
  tracks or run policies during design. Test the machinery with synthetic lists.
  Do not screen held-out tracks for difficulty or quality when creating a split.
- Hash/version manifests and the generation recipe; log realized training IDs
  and sampler state. Keep these infrastructure records out of agent observations.
- Use common final tracks across methods and compatible protocols. Freeze the
  entire cross-protocol campaign before viewing any of its test results.

Separate random streams for track generation, model initialization/training,
collection actions and evaluation-time planning. Derive and record them from a
run seed so an extra search/logging call cannot silently change future tracks.
Reset per-episode memory and disable test-time weight/normalization updates;
online adaptation would require a separately declared experiment.

Start with one training seed for engineering, three for exploratory comparisons,
and five independently trained runs for headline learners. More evaluation laps
from one model do not create more independent trained models. A fixed expert has
no training-seed variation. For stochastic evaluation, predeclare a repeat schedule
and retain the model/track/repeat hierarchy.

Training need not use a permanently small finite track list. A fixed partition
and stream recipe permits fresh tracks reproducibly. Common streams do not mean
identical experience: policies generate different trajectories/episode lengths.

## Environment and information contract

An immutable protocol ID must cover:

- Gymnasium/Box2D versions, generator/source hash, environment arguments,
  reward, termination/truncation and the 1000-native-frame limit.
- Observation fields/units/reference frame, crop/rendering/HUD, footprint,
  history/memory, normalization and previous-action/reward inputs.
- Three-channel controls, clipping/scaling, action repeat and observation cadence.
  Count actual native frames when an episode ends inside a repeated action.
- Allowed training labels, teacher actions, pretrained data/weights, internal
  model simulations and external simulator queries, including replayed prefixes.
- Metric definitions, selection rules and inference-time adaptation rules.

Actor input rules stay the same at training, validation and test. Auxiliary
training labels are separately declared. Full-map expert, local geometry, fixed
local vision and native pixels remain separate conditions. Official CarRacing
uses 96x96 RGB observations and steering/gas/brake controls; pin the installed
implementation rather than relying on a floating documentation page alone
([official specification](https://gymnasium.farama.org/environments/box2d/car_racing/)).

Local geometry is an inexpensive mechanism pilot; run a native-input pilot early
to catch adaptation failures before committing the full budget. Select primary
conditions on development evidence before main training. Do not launch every
combination of sensor range, predictor and horizon. Repeat four and 30 m local
radius are the implemented local pilot defaults, not scientifically optimal settings.

Leakage checks cover reset/step `info`, full-map handles, raw truth, global
progress, logs, filenames/IDs, caches and cloning of the actual evaluation world.
The legacy `Policy.reset(track)` / `act(state, obs)` is privileged. Local methods
now use the implemented restricted observation interface. Future simulated actions may depend on permitted
simulated observations, never an otherwise hidden scenario identity.

## Artifacts and evaluation worth retaining

Every run should save or reference by immutable hash:

- Source commit plus any patch, dependency lock, hardware/runtime settings,
  protocol/partition hashes, complete configuration and all random streams.
- Data schema/provenance, observation/action/reward/termination sequences,
  collection policy and label provenance. Separate evaluator truth and restricted
  training labels from agent-facing data.
- Selected and scheduled checkpoints with their normalization; optimizer,
  scheduler/scaler, replay data and sampler/RNG state as needed for continuation.
- Costs, selection history, failed trials and per-episode metrics. Count actual
  environment frames, decisions, labels, teacher/query access, model transitions,
  train/tune/evaluation time and latency quantiles.

Measure storage/compression on a small collection first. A seed alone does not
reconstruct a stochastic collection without its actions and collection state.
Data from one observation/supervision regime are not automatically reusable in
another. Shared offline data support module comparisons; on-policy algorithms
still need their own interactions. Report unique collected data and downstream
training work separately when sharing artifacts.

Verify checkpoint reload **and continuation**. Exact mid-episode Box2D/vector-worker
resume is not assumed; support checked episode-boundary resume first. Record any
reset, partial-episode loss and extra frames. Do not describe a restart as an
identical continuation without evidence.

Primary endpoint: mean official return across all scheduled episodes. Co-report
completion, departures, progress and latency; conditional lap time includes its
denominator. Preserve or explicitly version the current strict-before-frame-1000
success metric, testing that boundary. Distinguish playfield exit from collision
and `terminated` from `truncated`.

Store rows by model/track/repeat/condition. Report per-trained-model outcomes and
method differences on common tracks, with uncertainty respecting model and track
variation. Pair track draws across methods; equal initialization-seed integers
alone do not justify treating trained-model draws as paired. State whether the
inference concerns the fixed suite or the track distribution. Use development
variance to decide whether more training runs or more evaluation tracks are useful.
Five models and 500 tracks do not guarantee sufficient precision. Specify the IQM
estimand and interval procedure; current method/split grouping needs extension.
The methodological rationale is supported by
[Agarwal et al.](https://agarwl.github.io/rliable/) and
[Henderson et al.](https://arxiv.org/abs/1709.06560).

Freeze checkpoint selection, primary contrasts and multiplicity treatment before
testing. Do not choose the primary condition or comparator from test wins.
Calibration fits only declared parameters on its partition; architecture,
prediction-space and controller selection use development racing. Repeatedly
revising a calibrator from calibration outcomes makes those data development
data, not an untouched calibration audit.

Invalid actions, NaNs and method timeouts are failures under a frozen handling
rule, not omitted episodes. Infrastructure interruptions may be retried with the
identical specification, preserving the failed attempt and reason. Profile the
whole observation→prediction→search→action pipeline. Offline simulation waits for
planning: high return alone establishes no real-time capability. Injected delays
are a separate condition.

## Gates before the main budget

The namespace, restricted-interface and basic-loop gates now have implementation
and development evidence. Competitive control/baseline and final-study gates remain
**open**; tests do not certify scientific superiority.

| Order | Deliverable | Required evidence |
|---|---|---|
| 1 | Versioned environment and information/action contracts | Official standard behavior preserved; correct reset/episode boundaries; no forbidden policy inputs |
| 2 | Partitions integrated across collectors/trainers | Reproducible streams, exclusion checks, whole-track provenance and historical audit |
| 3 | Run schema, evaluation and checkpoint pipeline | Small run reloads; continuation semantics verified; costs correct; incompatible protocols cannot pool |
| 4 | Small system and baseline pilots | Working prediction→simulation/search→action loop; local/expert diagnostic; recurrent-policy smoke run; native-input and runtime feasibility |
| 5 | Exploratory comparison and claim plan | Strong comparator actually works; learning curves and variance justify budgets; essential component swaps and primary contrasts selected |
| 6 | Main training, then final study freeze | Fixed protocol/tuning rules; all selected models, conditions, analysis and failure handling frozen before test |

Use a small engineering cap first (the existing proposal is up to 100k native
frames per run). Preserve pilots as pilots; reuse artifacts only if provenance,
protocol and access match. They do not silently become independent confirmatory
replications.

Audit competitor conventions and pin upstream code early. A working install is
not a reproduced result. Keep original-protocol replication separate from our
adaptation. Investigate broken or undertrained baselines before comparisons.
Declare tuning budgets and extension rules; equal frames do not equal compute.

Prioritize the main system and comparisons that separate model quality from
search: a strong recurrent policy, established planning, a controller using the
same representation, and fixed-predictor/fixed-planner module swaps as applicable.
Direct candidate scoring is needed if claiming explicit multi-step simulation
adds value. Optional predictor-space selection and future evolution do not block
the basic system.

## What changes require reruns?

| Later change | Expected consequence |
|---|---|
| Plot or aggregation fix with correct complete episode records | Recompute analysis; no training |
| New diagnostic recoverable from saved trajectories | Recompute diagnostic; no training or driving |
| Missing diagnostic trajectories or evaluator-only bug | Re-evaluate frozen checkpoints; retraining only if learning/selection was affected too |
| Inference-only planner horizon/budget change | New validation and driving runs; reuse predictor if its contract still fits |
| Predictor target/architecture/loss change | Retrain affected models and possibly downstream control; recollect only if needed data are missing |
| Sensor/history/action, physics/reward or training-preprocessing change | New protocol and affected retraining; preserve old result interpretation |
| Train/test overlap, forbidden information or test-driven design | Affected comparison invalid; assess contamination, repair and use an untouched test partition |

If final testing exposes a genuine bug, preserve the campaign, document the fix
and rerun affected comparisons. Once test feedback influences method design, it
becomes development feedback; a fresh confirmatory campaign needs untouched data.
Do not quietly replace losing or buggy rows.

Research cannot be guaranteed free of reruns. These gates target preventable
losses from missing provenance, changing interfaces, leakage and unfair
comparisons, while keeping the method open to useful development evidence.
