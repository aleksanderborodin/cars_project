# MR.Q/MRS.Q: pinned source and adaptation readiness

Reviewed 2026-10-30. [Comparator review](COMPARATOR_REVIEW_2026-10-30.md) ·
[Experiment contract](EXPERIMENT.md) · [Provenance](../reports/references/mrq_mrsq_source_audit_2026-10-30.json)

**Original source-audit status:** selected official files pinned, cached,
hash-checked and inspected without importing their algorithms. The subsequent
[interface and synthetic verification](EXTERNAL_INTERFACE_2026-10-30.md) implements
the local boundary and passes tiny CPU learner/reload checks after a declared MR.Q
priority fix. No external driving run or competence result exists. The frozen
paper queue continues unchanged; the reproduction gate remains open.

**Origin:** the user requests serious contemporary comparisons and autonomous
progress. The assistant chooses this pair because MR.Q supplies an independent
actor family and MRS.Q directly challenges search-related claims. Their algorithms
are prior work, not project inventions. Adapter choices below are assistant
proposals, with uncertain driving quality.

## Exact sources and roles

| Source | Pinned revision | Intended role |
|---|---|---|
| [Official MR.Q](https://github.com/facebookresearch/MRQ/tree/073adcd6b30f84694c241e00bd684db50242a215) | `073adcd6b30f84694c241e00bd684db50242a215` | Independent actor with learned representations |
| [Official MRS.Q](https://github.com/facebookresearch/MRSQ/tree/b12abaf309833f28619dfd87f4251074abed3ae3) | `b12abaf309833f28619dfd87f4251074abed3ae3` | Modern learned-model planner, separate from our programmed main model |

Both roots carry CC-BY-NC-4.0 licenses. Sixteen selected files include agents,
networks, replay, preprocessing, runners, utilities, READMEs and licenses. Every
cached byte hash and all eleven Python AST parses passed. This is source inspection,
not runtime verification, a complete repository audit or a benchmark reproduction.
No upstream results directory was downloaded or consulted.

[MRS.Q's publisher paper](https://proceedings.mlr.press/v306/chang26a.html) is close
prior art. Its specific value-overestimation explanation must not be assumed to
explain failures of our engineered utility and observation-initialized physics.
MR.Q and MRS.Q also differ in model losses, value ensembles and training details;
their score difference alone cannot isolate the effect of adding search.

## Concrete source findings

1. **Environment access:** both Gym wrappers use generic Gym construction and
   declare nonpixel input with history one. They do not enforce our namespaces.
   Calling the supplied runners on CarRacing is not a valid local reproduction.
   Use a repository-owned harness around `DrivingEnv`; never hand the actor its
   environment, seeds or evaluator information.
2. **Actions:** agents produce symmetric normalized controls multiplied by one
   scalar maximum. The outer adapter must map `u` to
   `[u[0], (u[1]+1)/2, (u[2]+1)/2]`. Replay stores the original normalized control;
   the permitted previous-action sensor stores the actual physical control.
   Test endpoints, inverse mapping, random warm-up and exploration consistently.
3. **Storage:** MR.Q's nonpixel memory estimate evaluates to scalar four because
   of conditional-expression precedence. MRS.Q's helper calls a CUDA memory query
   even for a CPU configuration. Neither should decide our allocation unchecked.
   Make storage explicit and record any helper patch independently of the agent.
4. **Image shape:** both CNN branches hard-code 1,568 flattened features, matching
   84×84 inputs. A 61×61 mask produces 512 features through those convolutions.
   Road masks plus floating ego sensors are not a drop-in upstream pixel input.
   MRS.Q additionally initializes `ln_zsa` only in its nonpixel constructor branch,
   while `model_all` accesses it in both branches. This is a static pixel-path
   concern requiring a synthetic runtime check; no crash was observed here.
5. **Evaluation randomness:** MRS.Q's evaluation still uses Gaussian proposals
   and Gumbel elite selection. Zero exploration noise does not make it deterministic.
   Reset warm starts per episode, isolate the planning RNG, and replay with that
   recorded stream. A deterministic final argmax would be a separate adaptation.
6. **Effective horizons:** MRS.Q declares `q_horizon=3`, but its pinned `train`
   samples `enc_horizon=5` for the RL target. Preserve actual source behavior in
   the first adaptation and log declared/effective values. Do not silently repair
   this while claiming unchanged algorithm behavior.
7. **Training work:** both agents perform 250 encoder updates at synchronization
   boundaries in addition to their per-call RL work. A profile must cross the
   10,000-slot warm-up threshold and a complete subsequent burst. Replay size
   includes final-observation slots and is not a real-decision counter.
8. **Isolation and continuation:** evaluation consumes global random streams in
   the supplied runners. Save/restore learner RNG around evaluation. Record native
   frames, training tracks, replay and normalization state, optimizer/RNG states,
   and fresh-episode restart semantics in our own harness. Test terminal, timeout,
   ring-wrap and history behavior before any performance run.

## Representation, memory and history decision

A lossless preflight option is the flattened 61×61 road plus six numeric values
(ego velocity/yaw and previous physical action): **3,727 floats**. The fixed visible
mask can be reconstructed exactly from the unchanged protocol. Removing its
repeated storage is not removal of an observation-dependent signal. This option
keeps the source MLP path and makes algorithm smoke checks possible without an
unreviewed image/ego fusion architecture. It is **not selected as a competitive
final encoder**; an MLP over a grid may learn poorly. A shared CNN/ego encoder for
both external agents is an alternative requiring an explicitly labeled adaptation.

At one million replay slots, source-style float observation storage alone is
**14.908 GB** for road plus six values, or **29.792 GB** if both masks are flattened.
These buffers store observations once and index successor observations; do not
multiply by two as though they allocate a separate full next-observation array.
Lossless uint8-road/float-numeric storage would use 3.745 GB, or 0.490 GB with a
bit-packed road. Those compressed forms are **not implemented** and require replay
sample parity checks. These are decimal byte calculations, not measured RAM peaks;
indices, actions, priorities, networks, optimizers, staging and saves add memory.

Gym history one is not equivalent to our PPO's recurrent observation history.
Any finite stack must be declared, reset without crossing episodes and paired
with appropriate history controls. A first feedforward preflight cannot establish
superiority over a recurrence-matched strong actor. Same-checkpoint actor/search
contrasts remain useful, with planner state and known-physics access disclosed.

## Planning work and valid comparisons

Default MRS.Q uses 512 samples, six iterations, three explicit decisions and 24
policy proposals. Counting repeated evaluations gives **9,264 latent transition
examples per plan**: 9,216 population transitions plus 48 proposal transitions.
Terminal state-action encoding/value computation adds 3,072 examples, with ten
value heads; policy calls add 3,144 examples. These are source-derived counts,
not measured latency. Learned transitions and Box2D native ticks are different
units, so do not equate them when matching compute. Report wall time, peak memory,
data, training work and each model's own rollout counters.

Keep three questions separate:

- MR.Q versus MRS.Q: an external algorithm comparison with all differences stated.
- A frozen MRS.Q checkpoint's actor versus its ordinary planner: a deployment
  diagnostic conditional on planner-trained data, not training without search.
- A frozen competent actor versus that actor plus our coded selection: the user's
  primary mechanism comparison. Supporting a new actor requires a proposal adapter;
  the current search loader is tied to SB3 recurrent checkpoints.

Retain default stochastic MRS.Q planning in the first faithful adaptation. Any
smaller population, modified value ensemble, alternate final selection, new encoder
or reduced replay is a recorded variant. The main project still uses programmed
simulation; the external planner does not replace it.

## Next executable gates, after the current measurement queue

1. Implement the restricted harness, explicit storage/action mapping and isolated
   RNGs. Keep upstream files immutable and patches/diffs separate. Add meaningful
   synthetic boundary, replay and reload checks; run the full project suite.
2. Perform a bounded serial readiness profile, proposed ceiling **12,000 real
   decisions / 48,000 native frames per algorithm**, plus separately declared
   reload/evaluation checks. This crosses warm-up and update bursts. Freeze the
   exact declaration before execution; no profile is launched by this document.
3. Resolve representation, measured memory/device/throughput and effective
   horizons. Do not assert a runtime from PPO measurements. Then declare a full
   learning budget and checkpoint/development-selection rule before performance
   training. The upstream Gym default is 1M decisions; at repeat four that is up
   to 4M native frames, not 1M native frames. Our old 2M-native review is smaller.
4. Show complete learning curves and competence before scheduling a headline
   five-seed comparison. Keep failed seeds and all trial costs. Strong SAC and
   same-engine MPPI remain independent controls, not closed by this source audit.

Confidence is high in these source-derived constraints and low in a performance
advantage or a best representation before execution. Native RGB/DreamerV3/Valdi
comparison decisions remain separate. Calibration, final-test data and private
candidate rankings remain untouched.
