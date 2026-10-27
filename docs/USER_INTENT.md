# User intent, decisions, and paper goal

Updated 2026-10-27. [Index](../README.md) · [Experiment contract](EXPERIMENT.md) ·
[Research assessment](PREDICTION_SELECTION.md)

This document records the user's intended project in plain English. Paraphrases
are identified as such. Assistant judgments are separate from user decisions.
It is a living record, not evidence of a measured result or an accepted novelty claim.

## Confirmed project constraints

- Work in English, inside Gymnasium CarRacing, using simulation and the current
  workstation. A physical robot and another simulator are not required.
- Build an actual system and aim for a serious scientific paper with competitive,
  reproducible results. Be candid about novelty, feasibility, and comparisons.
- Separate components so perception, prediction, internal simulation, and control
  can be validated independently and in combination.
- Keep the active repository documented and indexed. Preserve obsolete work on
  archive branches, rather than retaining contradictory active plans.

## Main system the user wants

Paraphrase: observe the environment, construct an internal understanding/model,
try candidate behaviors inside that model, select the behavior with the best
predicted outcome, execute a short part, observe again, and repeat. Validate the
actual driving outcome in the real Gym environment.

The internal model may be learned, partly engineered, or adapted from existing
research. We do not need to invent every predictor or representation. The same
predictive model should support evaluating multiple candidate behaviors; this
is not a requirement to train a separate model for each maneuver.

The main project is this complete prediction/simulation/control system. It is
not restricted to predicting hidden roads. Full versus local information and
perception variants are controlled experiments within the project. Predicting
100 metres of road was an assistant prototype suggestion, not a user requirement.

The user likes the idea of selecting behavior by simulation/search rather than
having to learn a direct driving policy. Learned controllers remain useful
comparisons and possible components/variants. The project does not prohibit all
learning: the predictor, perception, or a downstream controller may be trained.
Which variant needs a learned actor must be explicitly reported.

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
standard-CarRacing comparison. No advantage is currently established.

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

Implement the common information/data/evaluation foundation and a working modular
prediction → simulation/search → action loop. Retain expert diagnostics and run an
early native-input pilot. Then compare complete systems and optionally investigate
predictor spaces/selection. Agentic evolution remains a later experiment.
