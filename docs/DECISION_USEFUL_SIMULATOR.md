# Future option: a programmed simulator selected for useful decisions

2026-10-28. [User intent](USER_INTENT.md) · [Main experiment](EXPERIMENT.md) ·
[Programmed simulator](SIMULATOR_SEARCH_PLAN.md) · [Research log](RESEARCH_LOG.md)

**Status: user-proposed future experiment. No implementation, GPU port, program
evolution or new campaign is launched.** The main goal remains separating world
understanding from decision-making and testing improvement from simulated
selection among a frozen proposer's candidate behaviors.

## User intent and corrections

The user proposes our own simulator, potentially much faster through GPU execution,
and possibly adapted through an agentic code-rewriting/evolution loop. Assess its
usefulness through the actions it helps select and their actual driving outcomes.
The user explicitly clarifies: **the simulator need not be realistic; it needs to
be useful.** A physically inspired approximation is one starting option, not a
requirement. Simplified, distorted or abstract action-conditioned rollouts are
allowed hypotheses. Prediction accuracy is a diagnostic, not an acceptance gate.

There are two motivations: determine whether useful search needs the target
environment's implementation, and whether it needs a faithful model of reality.
The user also hopes this would strengthen the rationale for eventual real-world
use. Current scope remains Gymnasium CarRacing on the workstation; simulated
evidence alone cannot establish physical-world transfer.

This optional extension does not replace the current programmed engine, the
understanding-versus-decision question, or the earlier normal-training/downstream-
selection protocol for learned predictors. It introduces a distinct future way
to construct or select a programmed world model. It does not require differentiating
racing reward through a neural predictor.

## Define usefulness outside the internal simulator

The inner loop stays: observe, construct an internal model, simulate candidate
behaviors, select one, execute its first action in the official environment, replan.
The proposed outer loop changes simulator parameters or code and measures the
resulting **actual environment** returns, completion, departures and computation.
Higher internal predicted utility alone is not evidence of improvement.

Initially freeze the proposer, road representation, candidate-generation rule,
planning horizon, selector and scoring rule while varying the simulator. If a
later study changes those too, report joint system optimization separately.
Identical residual draws specify matched feedback behaviors; their imagined
actions can differ as the model changes future observations. Do not claim
identical action trajectories merely because random seeds match.

Choose the primary driving objective and compute constraints before outer-loop
optimization. Official return remains the current primary score; completion,
off-road exposure and latency must also be reported. A future objective change
requires a new declaration, not switching metrics after observing a favorable result.

Use training-track interactions for parameter/code search and a declared development
budget for system selection. Freeze the selected program before eventual final
testing. Count every interaction, candidate program, failed trial and optimization
cost. Program evolution driven by racing outcomes is an optimization/training
process even if no neural weights are fitted. It is not zero-training control.
User-only privileged ranks/regrets never become fitness signals or agent feedback.

Operational requirements concern permitted information, valid control interfaces,
finite numerical execution and explicit resource limits. They do not require
physical fidelity. The program must expose action-conditioned state evolution
used by the declared rollout/search mechanism. If a variant instead directly
scores behaviors or outputs an action, label it as a scoring/controller comparison.
Do not relabel it a simulator to preserve a preferred paper story.

## Separate the hypotheses

| Question | Controlled comparison | Possible supported conclusion |
|---|---|---|
| Does selection help? | Same frozen proposer directly, random variation, and simulated selection | Increment from the tested selection mechanism, conditional on its model and candidate family |
| Is target vehicle code required? | Current reused vehicle code versus a separately programmed model with declared observation/action data | The tested system can improve driving without reusing the target dynamics implementation |
| Is realism required? | Prediction-calibrated and driving-selected versions of the same approximate model family | Better predictive fit need not be necessary for better decisions in this setting |
| Does GPU execution help? | Same model/formulation on CPU and GPU, with numerical checks and end-to-end profiling | Measured implementation speed/quality tradeoff at the tested batch sizes |
| Does agentic adaptation help? | Frozen/manual program, parameter-only search, and agentic code search with comparable feedback/work budgets | Incremental value of code adaptation beyond simpler tuning |

For model quality, first match candidate count, horizon, utility and information.
Then separately compare under the same wall-time budget, allowing a faster model
more search. Otherwise faster implementation, more candidates and different
model behavior are confounded. Report both equally informative comparisons.

Start from a small declared approximate model and a numerical reference, then
profile. A GPU port is conditional on an end-to-end benefit: world construction,
road queries, imagined observation generation, recurrent inference, transfers,
synchronization and warm-up all count. Report runtime at small and larger batches;
kernel throughput alone is insufficient. A fidelity-preserving port and a deliberately
different task-useful model are different interventions, even if both use a GPU.

## Access and realism claims need different evidence

The current implementation reuses Gymnasium vehicle code; its state/contact/road
estimates are already approximate. This does not establish code independence or
that arbitrary model errors are harmless. A separate implementation can remove
that dependency, but rewriting inspected source into another language still uses
its design knowledge. We have already inspected upstream code. Document this
history and limit the initial claim to non-reuse; a genuine source-unavailable
design claim needs an independently controlled, blinded development protocol.

The target Gym environment would still generate observations/rewards through its
ordinary interface. That black-box interaction is distinct from reading its
private state, parameters or implementation. Preserve the local information
contract; an independently coded model must not recover hidden tracks or seeds.

Useful imperfect simulation in Gym would support a narrower empirical result
about decision-making under model mismatch. It would motivate physical experiments,
not demonstrate sim-to-real transfer. An accuracy-versus-driving dissociation must
be measured on declared data; deliberately making a model inaccurate is not itself
an achievement. If fidelity and control improve together, report that outcome.

## Assistant assessment, prior art and unresolved choices

**High confidence:** the attribution controls, independent outcome evaluation and
information boundaries are needed. **Moderate confidence:** a compact programmed
model is worth a bounded future test given the measured search cost. **Unknown:**
which abstractions help, whether a GPU wins at 7/21 candidates, and whether program
evolution beats parameter tuning or overfits the development tracks/planner.

The conceptual distinction has prior art. Value equivalence formalizes models
through their usefulness to specified planning operations; it does not guarantee
that our simulator preserves useful choices. [Grimm et al., NeurIPS 2020](https://arxiv.org/abs/2011.03506).
Prediction and control objectives can disagree in model-based RL.
[Lambert et al., L4DC 2020](https://proceedings.mlr.press/v120/lambert20a.html).
The prospective contribution must be our specific mechanism or substantive,
controlled findings, not the general statement that models can be task-specific.

One possible implementation tool is [NVIDIA Warp](https://github.com/NVIDIA/warp),
which supports CPU/GPU kernels. It is not installed or selected for this project;
no speedup is inferred from its availability. Pin and audit any selected backend
before changing the environment. No dependency changes during the frozen campaign.

Next activation gate: finish the current frozen comparison and review the
established-optimizer work and public cost/error evidence. Declare one bounded
model substitution before any GPU/evolution expansion. The current runs continue
under their original source snapshots and settings.
