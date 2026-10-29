# Valdi comparison audit

Reviewed 2026-10-29. [Index](../README.md) · [References](REFERENCES.md) ·
[Paper blueprint](PAPER_BLUEPRINT.md)

**Status:** source audit only. No third-party code was executed, dependencies
installed, model trained or score reproduced. The user's intent is comparison
with credible existing systems; the audit and choices below are the assistant's.
The method and code belong to the Valdi authors.

Valdi is closely related actor-free learned-model planning in CarRacing. Its
[paper](https://arxiv.org/html/2607.00917v1) describes preliminary experiments with
two training runs, a diffusion-versus-MLP comparison and a learned terminal value.
It is relevant prior work, not an established CarRacing SOTA ranking. Our programmed
engine and frozen learned proposer are different resources and must be disclosed.

The reviewed [source revision](https://github.com/Kit115/ValueDiffusionWorldModels/tree/c5a203d933166dd55fb012323f928fb11b66e789)
is `c5a203d933166dd55fb012323f928fb11b66e789` (2026-08-06), later than paper v1.
The [public provenance record](../reports/references/valdi_source_audit_2026-10-29.json)
pins 17 downloaded files, dependency versions and source locations. We have not
established which revision produced the paper's figures.

## Comparison contract

| Item | Verified source behavior | Consequence for our comparison |
|---|---|---|
| Observations | RGB with its dashboard strip zeroed, plus seven values extracted from the original dashboard pixels | These numeric channels do not read hidden simulator state; they still differ from our exact local geometry/ego interface |
| Planning | 512 candidates ×10 iterations; five chunks of three distinct actions | 5,120 evaluated trajectories per replan, versus our 21; three-action execution is not a repeated single action |
| Objective | Learned discounted chunk rewards and two terminal-value heads | Different model, horizon and objective from our 1.6-second coded rollouts; no policy prior |
| Evaluation | Fixed 600-step loop, human rendering; termination flags ignored | Cannot insert that score into our official 1,000-frame terminating-task table |
| Software | Lockfile has Gymnasium 1.2.3, Torch 2.9.1, NumPy 2.3.5 | Use a separate environment for replication; our installed stack differs |

Sources: [observation wrapper](https://github.com/Kit115/ValueDiffusionWorldModels/blob/c5a203d933166dd55fb012323f928fb11b66e789/environment.py),
[planner and trainer](https://github.com/Kit115/ValueDiffusionWorldModels/blob/c5a203d933166dd55fb012323f928fb11b66e789/train.py),
[evaluation](https://github.com/Kit115/ValueDiffusionWorldModels/blob/c5a203d933166dd55fb012323f928fb11b66e789/evals/performance_evaluation.py),
[lockfile](https://github.com/Kit115/ValueDiffusionWorldModels/blob/c5a203d933166dd55fb012323f928fb11b66e789/uv.lock).

The evaluation-script finding is not proof that the published scores used that
exact script. Training collection does stop on termination. Both differences need
resolution before a reproduction claim; silently correcting evaluation would be
an adaptation whose scores need a separate label.

## Work required before execution

The default [configuration](https://github.com/Kit115/ValueDiffusionWorldModels/blob/c5a203d933166dd55fb012323f928fb11b66e789/configs/default_diffusion.yaml)
implies at most 3,006,000 collected frames including initial trajectories and
300,000 gradient updates. These are derived caps, not measured costs. The declared
[buffer arrays](https://github.com/Kit115/ValueDiffusionWorldModels/blob/c5a203d933166dd55fb012323f928fb11b66e789/buffer.py)
occupy approximately 16.65 GB before model/batch overhead. Inference and memory
must be profiled before promising workstation feasibility or an ETA.

The source also leaves integration questions: all control channels use `tanh`,
including gas/brake; training resets lack our partition exclusions; full RNG state
is not checkpointed; buffer save/load omits its terminal array; the value target
does not consume terminal masks. These are static findings, not measured causes
of poor performance. Each repair needs an explicit original-versus-adapted record.
No LICENSE/COPYING file occurs in the pinned repository tree, so source reuse
terms remain unresolved before copying its implementation into our active code.

**Decision:** keep Valdi on the comparator roster and preserve this source audit.
Prioritize a competent matched local proposer and verified native PPO recipe before
committing to this larger reproduction. A later Valdi experiment needs its own
pinned environment, action/termination decisions, disjoint partitions, checkpoint
rule, execution profile and budget. Do not import upstream evaluation seeds into
our final test or relabel a patched run as an unchanged reproduction.

Confidence is high in the documented source differences, moderate in adaptation
feasibility, and currently unknown in runtime or reproduced performance. The audit
does not replace an execution test or establish that our method is better.
