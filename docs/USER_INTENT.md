# User intent, decisions, and paper goal

Updated 2026-10-29. [Index](../README.md) · [Experiment contract](EXPERIMENT.md) ·
[Research assessment](PREDICTION_SELECTION.md)

The [paper blueprint](PAPER_BLUEPRINT.md) is the single consolidated starting point:
it distinguishes user ideas, assistant refinements, prior work and evidence status.

This document records the user's intended project in plain English. Paraphrases
are identified as such. Assistant judgments are separate from user decisions.
It is a living record, not evidence of a measured result or an accepted novelty claim.
New questions, attributed ideas, motivations and probes are recorded in
[RESEARCH_LOG.md](RESEARCH_LOG.md). Preserve the author and purpose when updating
an idea rather than converting a question into an agreed design.

## Confirmed project constraints

- Work in English, inside Gymnasium CarRacing, using simulation and the current
  workstation. A physical robot and another simulator are not required.
- Build an actual system and aim for a serious scientific paper with competitive,
  reproducible results. Be candid about novelty, feasibility, and comparisons.
- Separate components so perception, prediction, internal simulation, and control
  can be validated independently and in combination.
- Keep the active repository documented and indexed. Preserve obsolete work on
  archive branches, rather than retaining contradictory active plans.
- Before main runs, anticipate partition, environment and experiment-design
  problems so results remain useful and avoidable reruns are minimized. The
  assistant's proposed implementation gates are in [RUN_READINESS.md](RUN_READINESS.md);
  numerical partition counts and budgets remain proposals.

## Main system the user wants

Paraphrase: observe the environment, construct an internal understanding/model,
try candidate behaviors inside that model, select the behavior with the best
predicted outcome, execute a short part, observe again, and repeat. Validate the
actual driving outcome in the real Gym environment.

**Latest explicit correction, 2026-10-28:** the main world model is an algorithmic
simulator written in code. Simulate proposed behaviors and choose by their
outcomes, rather than replacing those simulations with a trained behavior scorer.
The earlier broad allowance for learned/engineered internal models does not make
a learned transition network the current default. The ridge pilot is a baseline,
not the intended final engine. See [SIMULATOR_SEARCH_PLAN.md](SIMULATOR_SEARCH_PLAN.md).

The same programmed simulator should evaluate multiple candidate behaviors.
A behavior proposer can be an existing trained policy or an engineered controller;
compare that original proposer against the same proposer plus simulated selection.
Its training/pretraining still counts if used. We do not need to invent every
perception, reconstruction or proposal component, nor train a model per maneuver.

The main project is this complete prediction/simulation/control system. It is
not restricted to predicting hidden roads. Full versus local information and
perception variants are controlled experiments within the project. Predicting
100 metres of road was an assistant prototype suggestion, not a user requirement.

The user likes the idea of selecting behavior by simulation/search rather than
having to learn a direct driving policy. Learned controllers remain useful
comparisons and possible components/variants. The project does not prohibit all
learning: the predictor, perception, or a downstream controller may be trained.
Which variant needs a learned actor must be explicitly reported.

## Main scientific purpose: understanding versus deciding

**User reaffirmation, 2026-10-28 (paraphrase):** the main goal is to distinguish
understanding the situation from deciding how to act, identify which parts are
already sufficient and which still limit performance, and show whether selecting
among sampled behavior variants improves the underlying model's driving.

World understanding includes road/state representation and action-conditioned
prediction; deciding includes proposing, comparing and executing behaviors.
Use controlled substitutions and frozen components so a whole-system score does
not hide where improvements originate. The useful first comparison remains the
same frozen proposer alone versus sampled variants selected by programmed
simulation, with random sampling and cheap corrections as controls.

Assistant interpretation: report which component is adequate or limiting **for
our tested task, information and budgets**. A search improvement does not by
itself prove perception is solved; a model error does not prove it limits driving.
The [information/control study](INFORMATION_AND_CONTROL_STUDY.md) describes further
interventions. The two-task distinction is central now even though its full-map
matrix is a future experiment. Optional simulator/predictor/evolution ideas must
serve this purpose rather than displace it.

**User's detailed three-stage formulation, 2026-10-29 (paraphrase):** train
comparable policies with ground-truth current spatial information, then additionally
with true future-road geometry, to examine perception and prediction limitations.
At each stage simulate variants of the same policy and select a behavior; compare
ordinary and strong policies both directly and with search. Also test useful
imperfect/potentially independent simulators, with prediction choices selected by
actual driving outcomes, then evaluate the frozen system on wholly untouched data.
The user wants a validated decomposition and evidence about where an end-to-end
policy struggles, with action selection as a central hypothesis, not a predetermined
finding. Keep agentic adaptation as a possible extension. The assistant's critiques
about retraining confounds, irreducible uncertainty, dynamics and compute are
refinements of this proposal, not changes to who originated it.

**User organization instruction, 2026-10-29:** collect the project's ideas in one
place, improve the proposed paper where justified, and clearly distinguish the
user's ideas from the assistant's. The blueprint records both, and separately
attributes established prior work and the status of measured results.

## Future option: a simulator useful for decisions without requiring realism

**User proposal and clarification, 2026-10-28 (paraphrase):** implement our own
simulator, possibly much faster on a GPU; adapt its parameters or code according
to the resulting driving usefulness, perhaps through agentic rewriting/evolution.
It does not need to be realistic, only useful for choosing actions. Physical
similarity is an option, not a requirement; prediction accuracy is diagnostic.

User motivations: test independence from the target simulator's code, test whether
an inexact/abstract model still improves behavior selection, and develop evidence
relevant to a possible eventual real-world approach. These are separate hypotheses.
The [future simulator study](DECISION_USEFUL_SIMULATOR.md) preserves intent,
comparison controls, compute/feedback accounting and claim limits. It remains a
future option and does not launch evolution or replace the current engine.

Assistant critique: actual driving determines usefulness, not optimistic internal
scores. Success without reusing target dynamics would not establish physical
transfer or erase prior source knowledge. Any data-driven program adaptation
counts as optimization; all train/development/test and private-oracle rules persist.
This does not change the earlier default for normally trained learned predictors.

## Predictor training and selection: clarified user intent

Paraphrase: train candidate predictors normally, using ordinary supervised or
self-supervised prediction objectives. Vary what they predict, their representation
space, architecture, hyperparameters, and checkpoints. For each predictor, train
or configure the downstream driving system that uses it. Choose among predictors
using validation racing performance of that complete system.

For a clean first comparison, the assistant recommends freezing each predictor
while fitting/configuring downstream control; this is an experimental design
recommendation rather than an independently stated user constraint. For a planner,
the downstream stage configures search/control; for a learned-controller comparison,
it trains the controller on top of the predictor/representation.

**This is downstream model selection. It is not a request to backpropagate racing
loss through the predictor or to replace normal predictive training with a special
decision-focused loss.** Prediction accuracy and calibration remain useful
measurements, but need not determine which predictor is deployed.

Choosing a better prediction space/model is an additional experiment within the
main system. It must not become a prerequisite that delays the basic working
pipeline or silently replaces the main paper goal. End-to-end decision-focused
predictor training was an assistant suggestion; it is not the current default.

## Expert and known-track experiment

**Further user proposal, 2026-10-28 (future work):** compare learned control and
simulation search under both local and full-map information, including local
search versus a learned driver trained/deployed with the full map. Decompose
road understanding, vehicle prediction and action choice to explain possible
advantages despite imperfect components. The user's ambition is a broader family
of effective approaches, not a measured universal-superiority result. The
[information/control study](INFORMATION_AND_CONTROL_STUDY.md) preserves the intent,
comparison matrix, precise possible conclusions and assistant critiques. It does
not launch global training or relax the private-oracle boundary.

The user wants to preserve an additional expert experiment, motivated by the
possibility that knowing the exact track permits substantially better control.
A hand-engineered expert, a full-map planner, or a high-budget reference may be
useful here. Compare known-track and restricted-information conditions explicitly.

Intended uses include a strong reference, diagnosis of whether failures come
from missing information or control, and potentially training demonstrations.
Expert demonstrations and full-track information are additional supervision/access
and must be labeled. A full-map reference is not an equal-information pixel/local
competitor, nor a proven optimal upper bound. The existing pure-pursuit expert is
retained. This idea does not authorize opening held-out tracks during design.

Possible future distinction: merely providing a map at episode start differs
from optimizing a trajectory/controller specifically for that known track.
The latter is per-track adaptation, with additional compute and a different
generalization claim. If studied, give comparisons matched adaptation access and
budgets, and declare the protocol before final evaluation.

Assistant assessment: **high confidence** that a competent expert/reference is
scientifically useful. Exact-track knowledge may help, but neither that benefit
nor superiority of a particular expert is guaranteed. Comparing the same controller
with global/local inputs is especially useful for isolating the information effect.

## Future option: agentic evolution

The user proposes agentic evolution as a possible future way of creating or
improving the algorithm that consumes predictions and drives the car. This could
include evolving controller/planner code, potentially in Rust, if it integrates
well and improves actual outcomes.

It is an **additional future training/optimization route**, not the current main
implementation path. Rust is an implementation possibility, not a requirement.
The archived evolution tooling should not be restored merely because this idea
is recorded. Do not start an agentic evolution campaign now.

If pursued, compare evolved and conventionally designed controllers using the
same predictor, allowed information, development feedback, and explicit search
budgets. Known-track and local/native variants remain separate. Evolution agents
must never receive held-out results or truth forbidden to the evaluated policy.

Assistant assessment: **medium confidence** in potential value, **low confidence**
in a performance advantage before experiments. It could discover inexpensive
control heuristics and improve inference efficiency. It could also overfit the
validation tracks, exploit predictor errors, or consume substantial search compute.
These are hypotheses to measure, not reasons to expand the current scope.

## Intended comparisons and publication claim

The user wants to compare the complete system with strong existing driving and
model-based methods, and may borrow their predictors/distributions as starting
points. A predictor optimized for conventional prediction and one selected in a
different space by downstream racing performance are potential comparisons.
The user's phrase 'SOTA predictor' is an aspiration to use a strong reference;
no particular checkpoint has been verified as the current benchmark leader.

The paper should demonstrate actual improvement in driving, supported by
component analysis and fair resource/information accounting. A restricted-view
study can explain mechanisms; native-input experiments are needed for the desired
standard-CarRacing comparison. A one-checkpoint development search gain now
exists; general advantage, novelty and standard-benchmark superiority remain
unestablished. See [the measured evidence](FROZEN_POLICY_RESULTS_2026-10-28.md).

Suggested paper framing, paraphrased from the user:

> Why teach an agent a driving policy when it can try candidate movements in an
> internal simulation and choose the best?

This is motivation, not an established claim. Candidate working title:
**Predict, Simulate, Drive: Control Through World-Model Search**.
A question-form alternative is **Can Online Planning Replace a Learned Driver?
A Controlled Study in CarRacing**. The final title should follow the evidence.

Assistant critique: continuous actions cannot all be enumerated; search is finite
and approximate. The selected action is best according to the model/search, not
necessarily best in reality. Planning instead of a learned actor is established
prior work, including PETS, PlaNet, and Valdi. A novel contribution or a sufficiently
substantive experimental result must still be established.

## What the user has not committed to

The user has not fixed a 100 m forecast, hidden-road-only hypothesis, GRU width,
mixture count, particular loss, planner budget, or partition counts. Numerical
settings in DESIGN_DECISIONS.md are assistant-proposed starting points, subject
to development evidence. The user has not authorized a final held-out campaign,
a SOTA claim, or immediate restoration of evolution/Rust tooling.

## Current priority

The common foundation and the earlier learned-dynamics pilot exist. The active
priority is the [programmed-simulator plan](SIMULATOR_SEARCH_PLAN.md): implement
the coded engine, obtain development results, then improve behavior proposals,
selection and observation-based reconstruction under controlled comparisons.
Agentic evolution remains a later experiment.

The user authorized autonomous progress without clarification interruptions and
asked for results plus documentation of uncertain choices for later discussion.
This authorizes implementation decisions within the agreed research boundary;
it does not open final test or private oracle outputs.

**User reaffirmation, 2026-10-28 (paraphrase):** proceed with the work, while
preserving the broader goal and ideas because they may substantially change how
we work. The current ridge model, straight-road prior, pure-pursuit candidates
and 21-point grid are replaceable engineering instruments, not commitments to a
final paper architecture. Correcting this implementation should not consume the
whole research project. Revisit representations, reused predictors, probabilistic
worlds and behavior search at the next evidence review; record why an idea is
activated, deferred or retired. No performance or novelty outcome is assumed.

The user additionally proposes future Bayesian/adaptive search over possible roads
and promising behaviors/controller designs. An immediate candidate-ranking metric
uses privileged outcomes to assess the original choice among fixed candidates.
**Its real outputs are user-only:** it may run online, but no agent, training,
tuning, checkpoint-selection or assistant-analysis process may consume them.
This restriction takes precedence over using that diagnostic for method design.
Details and motivations are preserved in research-log entries R3–R5.
Another future option is regret-based prediction/model selection, with rank as
a diagnostic and potentially continuous/adaptive behavior search instead of the
current finite grid. It preserves normal predictor training as the default and
does not authorize using current private oracle results. See research-log R8 for
the user's purpose and the assistant's caveats.

**User execution instruction, 2026-10-28 (paraphrase):** carry out the agreed plan
and keep iterating toward the data needed for the paper without routine questions.
Document assistant decisions, uncertainty, rationale and ideas for later discussion.
This is ongoing development authorization, not evidence that a SOTA outcome is
achievable or that final-test/private-oracle restrictions have been lifted.

**User instruction, 2026-10-29 (paraphrase):** carry out the intervention study
with the strongest practical design and proceed. The assistant's first concrete
implementation is the [four-cell geometry pilot](INTERVENTION_STUDY.md), separating
visible-reference and hidden-road substitutions before new V/L/F policy training.
Those exact switches, sample sizes and execution order are assistant choices,
not additional user-origin ideas. Idea origin stays distinct from validation status.
