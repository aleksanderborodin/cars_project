# Comparator and novelty gap review

Reviewed 2026-10-30. [Paper ledger](PAPER_CAMPAIGN_2026-10-30.md) · [References](REFERENCES.md)

**User question:** are we comparing against the best methods and doing the work
needed for an actual SOTA-level scientific paper? **Assistant answer:** not yet.
The executed experiments provide reproducible mechanism evidence, but strong
external-comparator coverage and a defensible novel contribution remain open.
The current queue is useful development work, not a completed SOTA comparison.

## Actual status and required comparison roles

| Role | Executed here | Remaining gap / priority |
|---|---|---|
| Frozen actor, random proposals, fixed correction, search budgets | Local recurrent PPO; broad seed 0 data; seed 1/2 comparison pipeline | Establish consistency and show all seeds; this alone is insufficient |
| Established same-world planner | Colored shooting and a policy-guided iCEM adaptation, with negative results | Add a carefully audited MPPI/CEM comparison with the same coded model, state estimate, objective and resource envelope; do not treat an adaptation's failure as refuting iCEM |
| Strong independent direct actor | Native/local PPO; legacy SAC exists but is not a valid restricted-protocol reproduction | Prioritize a competent matched local SAC baseline and an official MR.Q adaptation audit; retain full budgets/curves and history disclosures |
| Modern learned-model planning | No execution | Prioritize MRS.Q versus MR.Q; TD-MPC2 is another credible candidate depending on protocol and feasibility |
| Native visual world-model control | No reproduced external score | DreamerV3 and CarRacing-specific Valdi need original-protocol and adapted-protocol decisions, profiling and verified runs |
| Behavioral-prior composition | No execution | APC-RL is relevant recent CarRacing work; its information, demonstrations and fixed-track task need explicit matching |

The named external methods are credible algorithmic comparisons, not verified
leaders on our exact CarRacing-v3 continuous-control contract. This review does
not establish an authoritative current CarRacing ranking. Search results, model
cards and third-party scores were not promoted to that status. Adding every name
is less useful than reproducing the methods that directly challenge our claim.

## Newly identified close work

**MRS.Q, ICML 2026:** [Chang et al., publisher paper](https://proceedings.mlr.press/v306/chang26a.html)
examines search failures despite accurate models and addresses value overestimation.
[Author code](https://github.com/facebookresearch/MRSQ) is available. This directly
overlaps our accuracy-versus-decision question. Our coded mechanics and engineered
utility differ from learned dynamics/terminal values, so their specific mechanism
must not be assumed to explain our failures. Its supplied [task list](https://github.com/facebookresearch/MRSQ/blob/main/MRSQ/utils.py)
does not include CarRacing: adaptation and validation remain necessary. No MRS.Q
code was executed or paper score reproduced here. The earlier roster missed this
important work and is corrected by this review.

**APC-RL, ICLR 2026:** [Rietz et al., proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cb22704ebc9bc4e0ef875d1959651012-Abstract-Conference.html)
studies composing demonstration-derived priors with online control. Its author
paper Appendix B.3 uses a fixed track and simulator-derived 16-dimensional features,
including future centerline points; demonstration costs also matter. Its published
scores cannot be inserted into our changing-track/local-mask table. The
[official implementation](https://github.com/frietz58/apc-rl) includes SAC controls.
This is a relevant protocol audit target, not a verified matched SOTA competitor.

## Existing work constrains the broad idea

[POPLIN](https://arxiv.org/abs/1906.08649) already combines policy proposals with
online planning. [Pezzato et al.](https://arxiv.org/abs/2307.09105) use parallel
physics simulations for MPPI. Thus neither searching around learned policies nor
using a programmed simulator is a sufficient novelty claim. The user's idea and
intent remain attributed to the user; these sources establish independent prior
work, not authorship of the user's conversation.

[Official MR.Q](https://github.com/facebookresearch/MRQ) is a modern direct-policy
candidate; its default Gym preprocessing is not a CarRacing/local-mask adapter.
[TD-MPC2](https://github.com/nicklashansen/tdmpc2) supports state/pixel tasks and an
explicit episodic configuration, but its listed original benchmark tasks are not
our local CarRacing study. [DreamerV3](https://github.com/danijar/dreamerv3) is an
author-hosted reimplementation with a different software stack; its name alone
does not establish a CarRacing reproduction. [Valdi](https://arxiv.org/abs/2607.00917)
has preliminary CarRacing evidence, not an established championship ranking; the
[prior pinned source audit](VALDI_AUDIT.md) retains its evaluation/input caveats.
[TD-M(PC)^2](https://arxiv.org/abs/2502.03550) is additional policy/planner-mismatch
prior work. Do not stop a literature update at an older method just because it
is convenient to install.

## Protocol work before external executions

Use immutable official revisions and separate environments; validate licenses,
dependency locks and native/normalized action mapping. Gas/brake bounds differ
from the symmetric action convention used by several control implementations.
Match allowed observations/history and keep original native-pixel, restricted
local and privileged/fixed-track results in separate tables.

The local 61×61 masks plus ego/history require an explicit representation/replay
adapter. Blindly flattening masks into an upstream million-transition float buffer
can exceed workstation memory; preprocessing, storage and history need a profile.
Preserve the official reward and termination, partition-enforced fresh training,
frozen evaluation statistics and recorded native-frame/learner-decision clocks.
Any wrapper/architecture deviation is an adaptation, not an unchanged reproduction.

Report known-physics access as prior knowledge, training/tuning costs, real data,
latent/model/physics rollouts, wall time and inference latency. The current
search21 p95 of roughly 362 ms exceeds its 80 ms action interval; offline score gains
do not establish deadline-respecting control. Same-model planners and separately
declared latency-budget controls are essential to the eventual efficiency claim.

## What would justify a strong paper

1. A precise contribution beyond established modular MPC: a validated component
   diagnosis or useful new decision/model tradeoff that survives close prior art.
2. Competent independent policy families and strong planner/world-model controls,
   with fair information, supervision, history and compute accounting.
3. Controlled perception/prediction/action interventions with sanity checks;
   oracle improvements, retraining effects and prediction errors are not by
   themselves causal proof of a single component bottleneck.
4. Robustness across training seeds and tracks, all negative trials, frozen
   standalone selection, and adequate uncertainty reporting. Current three-actor
   development work is exploratory; a headline multi-seed study is still pending.
5. A frozen final protocol and untouched final evaluation, after method selection.

The existing partition/replay/cost/failure safeguards are meaningful strengths.
They cannot substitute for novelty or strong competitors. A single Gym simulator
can support a rigorous paper, but a narrow expensive improvement over weak PPO
would be a weak basis for the user's desired high-level claim. We must be willing
to narrow the claim or report a negative result if stronger comparisons defeat it.
No acceptance probability or guaranteed SOTA outcome is justified.

## Execution decision and review provenance

Let the already frozen local mechanism queue run unchanged. During its measurement
phase, external work is limited to source/protocol/design review; no competing
training/profiling is launched. Next prioritize the MR.Q/MRS.Q pair and same-engine
MPPI comparison readiness, with SAC as a practical independent actor control.
Choose the visual comparison from DreamerV3/Valdi after feasibility and protocol
audits. This is a concrete comparison priority, not a declaration that any of
those unimplemented methods is already running or a known CarRacing champion.

The [review record](../reports/references/comparator_gap_review_2026-10-30.json)
lists primary URLs and execution status. Official browser sources were reviewed;
an attempted GitHub API commit-pinning fetch timed out. New repository revisions
are therefore not claimed pinned or fully audited. The author APC PDF was cached
and text-extracted locally; its SHA256 is recorded. No third-party code, dependency
installation, benchmark episode, private rank or held-out data was used in this review.

**Subsequent readiness work:** the fetch problem is resolved in the separate
[MR.Q/MRS.Q pinned audit](MRQ_MRSQ_AUDIT_2026-10-30.md) and
[MPPI source/readiness audit](MPPI_READINESS_2026-10-30.md). Those records include
immutable revisions, cached hashes and concrete adaptation constraints. The original
review record above preserves its earlier unsuccessful fetch honestly. Source
inspection is now complete for the selected files; external integration, runtime
validation and driving reproduction remain open.

### Replication evidence observed during this review

Both fresh local training seeds completed with finite diagnostics and all scheduled
checkpoints. Seed 1's selected 1M checkpoint scores 869.23/32%; seed 2's selected 1M
checkpoint scores −17.89/0% on the same first 50 development tracks. This is an
urgent competence/robustness gap in the current PPO recipe. The latter seed remains
in the fixed comparison; it must not be omitted or advertised as a strong actor.
Finite optimization and passing tests do not establish learned task competence.
The current 2M-native-frame review is also roughly 0.5M learner decisions, not a
full 4M-decision native training budget. No general RL limitation follows from it.
