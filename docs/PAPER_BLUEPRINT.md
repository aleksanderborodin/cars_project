# Paper blueprint: what limits a learned driver?

Updated 2026-10-29. [Index](../README.md) · [Experiment contract](EXPERIMENT.md) ·
[Evidence status](STATUS.md) · [Attributed chronology](RESEARCH_LOG.md)

This is the consolidated home for the paper idea and the ideas recorded in this
conversation, USER_INTENT and research-log R1–R47. It connects them to one study;
it does not activate every extension. The experiment contract still governs
access and evaluation. Existing numerical defaults are development choices.
Historical results retain their original protocol, and archived plans stay archived.

## Origin and evidence are separate

- **User:** proposed or explicitly clarified by the user in this project.
- **Assistant:** a proposed refinement, critique, implementation choice or experiment.
- **Prior work:** attributed to a cited external source; inclusion is not a project invention.
- **Evidence status:** measured, implemented but unvalidated, proposed, deferred,
  or rejected as a default. This is separate from who proposed the idea.

"User" records conversational origin, not a claim of worldwide scientific novelty.
When an assistant refinement follows a user idea, both origins are stated. The
register below preserves motivations and links to detailed records. Corrections
supersede earlier defaults without erasing their history.

## Central thesis and candidate title

**User's intended study, paraphrased:** separate (1) understanding the current
scene, (2) predicting what lies ahead and what actions will cause, and (3) choosing
the action. Train comparable policies with current-state and future-road privileges
to bypass the first two difficulties. At each level, test whether trying variants
of the same policy in simulation improves decisions. Then test whether an imperfect,
possibly independently programmed model can recover useful gains under ordinary
partial observations. Identify what remains difficult instead of treating one
end-to-end score as an explanation. Compare competent existing policies and their
search-augmented versions, and evaluate the frozen system on untouched tracks.

**Assistant's proposed scientific formulation:** how much recoverable action-selection
headroom remains after supplying accurate current-scene and future-road information,
and how much can a restricted, task-useful simulator recover at a stated compute cost?
This is a falsifiable question. It must allow perception, forecasting, optimization,
training coverage or model bias to turn out to be the dominant limitation.

Proposed neutral title, **Assistant**:
**What Limits a Learned Driver? Perception, Forecasting, or Action Selection?**
The user's earlier "try movements instead of teaching control" framing remains
motivation. Our current learned proposer prevents a claim of eliminating policy
training; finite search is not enumeration of all continuous behaviors.

## Define the three problems precisely

| Problem | User's proposed intervention | Assistant's identification caution |
|---|---|---|
| Current-scene understanding | Supply ground-truth current spatial information and train the corresponding policy | Match the visible footprint and sensors; revealing otherwise invisible state adds information as well as removing perception error |
| Forecasting | Also supply the true road ahead and train with that privilege | This removes hidden-road uncertainty, not vehicle-response/latent-state error; unforeseeable road choices are not necessarily a defective predictor |
| Action selection | Simulate variants of the same frozen proposer, compare outcomes, execute the selected first control and replan | Improvement concerns this proposer, candidate family, model, utility and compute; it is not proof of global optimality or a universal RL limitation |

**Assistant refinement:** forecasting has two distinct objects. In CarRacing the
road is static: a belief over unseen road depends on observed history, while the
car's future motion and future observations depend on actions. Knowing the complete
road does not identify hidden wheel state or prove the vehicle model correct.
The user has proposed a distribution, not a fixed 100 m forecast or one mandated
representation. With a probabilistic model, evaluate each behavior across common
world hypotheses and respect future observation limits; do not choose its most
optimistic world or reveal scenario identity to the imagined policy.

**Assistant interpretation rule:** say "sufficient for the tested task and budget"
or "remaining limitation." A small score difference is not proof that a module
is solved. An inaccurate prediction can be adequate for a particular decision.
Current local-mask experiments bypass visual perception; they do not show that
an RGB RL policy has learned perception successfully.

## Main experimental matrix

**User:** matched-policy retraining under progressively revealed information.
**Assistant:** explicit rows and paired within-row selection comparisons.

| Information regime, at both training and evaluation | Direct policy | Same frozen policy + random variation | Same frozen policy + simulated selection |
|---|---|---|---|
| V: local visual observations, declared ego/history | V-direct | V-random | V-search |
| L: exact currently visible geometry, same footprint and ego/history | L-direct | L-random | L-search |
| F: L plus declared oracle future-road geometry/full map | F-direct | F-random | F-search |

These are proposed mechanism-study rows. The standard native-pixel benchmark is
a separate protocol: current native zoom/HUD, numeric sensors, repeats, episode
limits and supervision cannot silently change. Existing L-direct/search results
fill part of one row. V and trained F rows are not implemented. Current global
geometry diagnostics do not substitute for F-direct or paired F-search.

**Assistant refinement for explicit forecasting claims:** add an intermediate
P condition, L plus a normally trained forecast, using the same future-road
representation/horizon as F. Compare L/P/F with a fixed downstream interface and
controller as well as the user-proposed retraining study. Report proper prediction
scores/coverage when a probability distribution is claimed. F-minus-L alone is a
value-of-information comparison; it does not measure the accuracy of a specific
forecaster. A no-learned-forecast planner can still contain an implicit straight-road
prior. This additional condition is staged, not permission for a full model grid.

Train a competent policy for each information regime; do not feed new oracle
channels into an unadapted local policy at evaluation. Within a row, freeze the
identical selected checkpoint and match real inputs, memory and action frequency.
The current proposals are smooth residual variations around deterministic recurrent
feedback, not samples from a calibrated road distribution or necessarily from the
policy's own action probability density. Preserve the base behavior as a candidate.

**Assistant additions:** keep the fitted constant correction and add an established
optimizer with the same model and total rollout budget. Avoid a full product of
every representation, optimizer, seed and budget. Start with integration/competence
checks, then expand the prespecified contrasts that answer the paper question.

An optional privileged reference can approach the user's "try in the same simulator"
diagnostic. It needs a separate explicit state/transition-access protocol and cost
accounting; full road alone is not exact-state access. No exact evaluation clone or
privileged ranking stream is enabled by this blueprint. Existing private ranks,
oracle returns and regrets remain unreadable by all agents and selection processes.

## Make the decomposition identifiable

**Assistant refinement A1:** use two complementary experiment types:

1. **Retraining under oracle inputs**, as the user proposed, measures achievable
   performance for a method under that information/representation/training regime.
   It changes the optimization problem and often architecture or supervision.
2. **Controlled module substitution** measures dependence on a component with a
   fixed downstream interface/controller. Train for the relevant input distributions
   or explicitly quantify their shift; substituting perfect inputs into a head
   trained only on noisy inputs can itself cause failures.

Together these are stronger than either alone. A modular auxiliary-label arm
must disclose its supervision; it cannot masquerade as a pure pixel-only method.
A probe that decodes scene information from a frozen policy representation is an
optional **Assistant** diagnostic. Successful decoding does not establish that
the policy uses that information causally to choose actions.

**Assistant refinement A2:** measure interactions, not a supposedly unique additive
partition of error. For a chosen endpoint J, compare search-minus-direct within
V, L and F; compare F-minus-L separately for direct and search. The difference
between those two search gains tests an interaction. Independently trained rows
are not the same checkpoint and their differences are not pure perception errors.
Track pairing alone does not remove training-seed uncertainty.

**Assistant refinement A3:** validate the diagnostic itself. Use a declared
prediction/perception degradation as a positive control and an unchanged/replayed
input as a negative control. Verify equal visibility and input semantics. Before
claiming an oracle gap is small, select a meaningful equivalence margin on
development data and check whether uncertainty supports it. Nonsignificance,
ceiling/floor effects or two equally weak policies are insufficient.

**Assistant refinement A4:** never claim full-map access proves prediction solved.
Report road-information effects separately from vehicle-model/state effects. A
remaining gap with oracle road information supports action-selection headroom
conditional on the remaining model assumptions; it does not identify every cause.

## Stronger controls and alternatives to the desired story

**Assistant proposal:** distill permitted search-chosen actions into a direct
policy, or improve its decision head with matched training feedback. This is
established prior art, not a new invention here. If the gap disappears cheaply,
the interesting result concerns training targets or amortizing computation; it
does not support the claim that a learned policy cannot make the choices.
Use ordinary executed/search-generated training data, never private oracle labels.

Other required safeguards against weak-baseline explanations:

- Learning curves, multiple initialization seeds, adequate tuning and at least
  one independently credible direct-policy family. Do not generalize from one PPO.
- A simple model-free correction, random candidate selection, and iCEM/CEM-style
  planning with the same programmed model. A direct learned scorer or terminal
  value is an optional comparison, not the default primary selector.
- Candidate count/rollout-work comparisons and separately latency-budget comparisons.
  Extra online computation and known mechanics are resources, not free information.
- Report official return, completion, pace, departures and failure types together.
  Preserve failed trials and both training and inference costs.

**User:** compare ordinary and strongest available policies, including improving
each strong frozen proposer with search and testing local search against a
full-map learned policy. **Assistant:** audit implementations and match protocols
before labeling any comparator SOTA. A policy that already plans at deployment
belongs in a different algorithm class from an actor-only baseline.
Recurrent PPO is our initial reference, not established SOTA. POPLIN/iCEM are close
planning controls; DreamerV3, TD-MPC2 and Valdi remain reproduction/adaptation
candidates whose exact role depends on input/history and available resources.

## Latest measured evidence, separate from the hypotheses

The frozen 200-track development campaign finished on 2026-10-28. Same selected
recurrent checkpoint, fixed planning stream, local mask/ego/history:

| Variant | Mean official return | Completion | Pipeline p95 |
|---|---:|---:|---:|
| Direct | 862.42 | 35.5% | 1.50 ms |
| Fixed correction | 865.95 | 39.5% | 1.53 ms |
| Search 7 | 873.05 | 58.5% | 136.77 ms |
| Search 21 | 884.48 | 72.0% | 362.11 ms |

Search 21 minus direct: return +22.06, nominal paired 95% track interval
[10.18, 33.35]; completion +36.5 percentage points [29.5, 44.0]. These are
development intervals for one frozen policy, not independent-training uncertainty.
The 80 ms action interval is exceeded; the environment currently waits for decisions.
See [completed evidence](BROAD_RESULTS_2026-10-29.md) and the provenance reports.

The lower-learning-rate probe completed and was not better: best 50-track
checkpoint 692.18/8%, final 624.51/0%. The original selected driver stays the
reference. Two unsuccessful optimization repairs strengthen the need for a
baseline audit; they do not prove learned control is inherently difficult.
The [completed factory/budget audit](BASELINE_AUDIT.md#factory-and-budget-audit-2026-10-29)
finds that the selected local actor used 251,904 learner decisions, 6.30% of the
reference recipe's nominal 4M decisions. Its repeat, encoder and schedule differ;
neither convergence nor an original native-recipe reproduction is established.
The [Valdi source audit](VALDI_AUDIT.md) likewise distinguishes pixel-derived
dashboard inputs, a different planning budget and evaluation termination behavior.

The [controlled geometry pilot](INTERVENTION_RESULTS_2026-10-29.md) subsequently
found worse driving with oracle-derived road despite more accurate predictions
of recorded controls. The [reference-tail follow-up](REFERENCE_CONTINUATION_2026-10-29.md)
changed only the scoring continuation and recovered 594.21/0%→876.29/80% on five
tracks. Original local search remains 906.99/80%; a remaining timeout prevents
an overall mean advantage. The [20-track replication](REFERENCE_REPLICATION_2026-10-29.md)
retains the conditional gain: true tail 813.85/60%, tangent 527.60/0%, historical
local 885.70/70%. Additional15 gain +287.64 versus tangent but lose 85.57 versus
local. One track is worse even than tangent; eight true-tail episodes time out.
Ten exact pilot replays passed. This supports reference mismatch as a limitation
of this configuration, not perception/forecasting adequacy or a general control
advantage. The oracle variant is not promoted over the local method.

The [matched sequence-search study](SEQUENCE_SEARCH_2026-10-29.md) is complete:
colored shooting 884.86/70%, iCEM adaptation 634.13/15%, historical knot search
905.94/85% on twenty tracks. Adaptive-minus-colored return is −250.72
[−359.57, −147.31], with −240.99 on additional18. Neither new method is promoted.
The trace audit finds stronger interventions and more departures with iCEM, but
does not identify their cause. This is a negative result for a particular small-budget
adaptation, not a rejection of iCEM or of the user's core simulation-selection idea.

This evidence supports investigating selection. It does not yet establish the
three-stage explanation, a perception result, a calibrated forecast, a strong
full-information learned comparator, or a standard-input SOTA result.

## Consolidated user idea register

All rows below have **Origin: User**. Implementation details proposed by the
assistant are recorded separately below and in the linked documents.

| ID | Idea and purpose | Status / detailed record |
|---|---|---|
| U01 | Reconstruct an internal scene, try actions there, execute and replan; robotics analogy motivates the system | Core; [user intent](USER_INTENT.md). Physical transfer is motivation, not current scope/evidence |
| U02 | Separate current understanding, future prediction and action choice; identify what is sufficient or limiting | Main paper question; latest explicit three-stage formulation in this blueprint |
| U03 | Train matched policies on revealed current-state information to bypass perception | Proposed controlled diagnostic; matrix V/L and its access cautions above |
| U04 | Train with true future-road geometry to bypass hidden-road forecasting | Proposed F row; does not reveal every vehicle state or solve dynamics |
| U05 | Try variations around the same proposer and choose through a programmed simulator | Implemented for one local recurrent checkpoint; [current results](BROAD_RESULTS_2026-10-29.md) |
| U06 | Compare against strong existing policies and augment those policies with search | Partially implemented with PPO; strong reproductions remain open, [references](REFERENCES.md) |
| U07 | Study local/global information and visual/state inputs within CarRacing | Local and privileged geometry exist; visual/native performance study pending |
| U08 | Forecast multiple plausible roads from history rather than committing to one continuation | Planned; [prediction study](PREDICTION_SELECTION.md). Current prior is deterministic |
| U09 | Reuse others' predictors; choose target spaces/hyperparameters by downstream validation driving after normal training | Optional study; [clarified intent](USER_INTENT.md). Special task-loss training is not the default |
| U10 | Use simple mathematical/physical assumptions or a learned forecast; do not hard-code a 100 m requirement | Open design alternatives; [design choices](DESIGN_DECISIONS.md) are assistant proposals |
| U11 | Select behavior by expected performance across possible roads; compare alternative risk/selection rules | Planned multi-world study; distinguish world uncertainty from behavior sampling |
| U12 | Bayesian/adaptive allocation over worlds and behaviors; move beyond a fixed finite grid | Future; research-log R3/R8. CEM is not automatically Bayesian inference |
| U13 | Rank the chosen candidate against privileged outcomes; show rank/regret online to the user alone | Sidecar implemented for earlier path; strict no-agent/no-learning feedback, R4–R5 |
| U14 | Possibly use regret/rank for future prediction/model selection | Deferred; no permission to read or reuse current protected outputs, R8 |
| U15 | Expert/known-track diagnostics, and possibly per-track optimization | Privileged geometry diagnostics exist; teacher/adaptation regimes remain separate |
| U16 | Compare restricted search with a full-information trained driver; even incomplete understanding might suffice | Future cross-information contrast; [information study](INFORMATION_AND_CONTROL_STUDY.md) |
| U17 | Own programmed simulator without dependence on the target implementation | Future; [simulator study](DECISION_USEFUL_SIMULATOR.md). Prior source knowledge must be disclosed |
| U18 | The simulator need only be useful for decisions, not physically realistic | Explicit user criterion for the future variant; accuracy is diagnostic |
| U19 | Select/adapt simulator parameters or code using final driving utility | Future outer-loop study; remains action-conditioned simulation; optimization cost counts |
| U20 | GPU implementation or another fast implementation for many imagined trials | Future implementation experiment; measure entire decision pipeline |
| U21 | Agentic evolution for controller/planner or simulator, possibly Rust | Deferred separate optimization routes; archived tooling stays archived |
| U22 | Eventually validate on tracks with no training or tuning; rigorous train/dev/calibration/test and random streams | Infrastructure implemented; final evaluation remains sealed until whole-study freeze |
| U23 | Preserve ideas, author intent, uncertainty, failures and interpretation; archive obsolete plans and index current work | Ongoing; this consolidated registry plus [archive](ARCHIVE.md) and research log |
| U24 | Work autonomously toward a serious paper on this workstation, entirely in Gym simulation initially | Current authorization/scope; no promise of SOTA, acceptance or real-world transfer |
| U25 | If target internals are unavailable, construct a useful model elsewhere and validate the resulting driving | Future model-independence motivation; target benchmark remains CarRacing; no second benchmark is silently introduced |
| U26 | Frame the motivation as testing candidate movements instead of learning every control choice | Retained motivation; current actor training and finite search constrain the eventual title/claims |
| U27 | Execute the intervention study after reviewing the design | Active; bounded fixed-controller geometry pilot first, followed by the declared staged design |

## Assistant refinements and implementation choices

All rows have **Origin: Assistant**, responding to the user goals above. They are
recommendations or engineering choices, not retrospectively user requirements.

| ID | Proposal / purpose | Status |
|---|---|---|
| A01 | Combine oracle retraining with matched-interface module substitution | Proposed validity refinement; separates learnability from component dependence |
| A02 | Distinguish hidden-road belief from action-conditioned vehicle dynamics | Proposed definition needed for forecasting claims |
| A03 | Use positive/negative controls, equivalence margins and interaction contrasts | Proposed decomposition-validation plan |
| A04 | Distill search improvements into a direct controller; compare a learned scorer if needed | Proposed stronger explanatory control; related methods have prior art |
| A05 | Keep cheap fixed corrections and random variation controls | Implemented; broad fixed correction complete, random variation initially measured on 20 tracks |
| A06 | Compare iCEM adaptation at equal total simulated work and separately equal latency | Source audited; [equal-work comparison complete](SEQUENCE_SEARCH_2026-10-29.md), negative adaptation result retained; hard latency-budget control remains open |
| A07 | Use five training seeds for headline learned conditions, track-paired intervals, frozen selection and full cost ledger | Existing methodological plan, pending headline campaign |
| A08 | Current seven/21 candidates, three residual knots, horizon 20, repeat 4, radius 30, observer/contact approximation | Engineering defaults with explicit uncertainty; not user's final architecture commitments |
| A09 | Sparse observation rendering, immutable executable snapshots, replay gates and sequential timing | Implemented/tested; not an algorithmic novelty claim |
| A10 | Audit and repair baseline training before arguing that RL cannot choose actions | Ongoing; KL and lower-LR probes failed, all costs retained |
| A11 | Keep the core diagnostic paper focused; add one task-useful-model extension only if it supports that core | Recommended scope control; no automatic GPU/evolution expansion |
| A12 | Judge novelty against privileged-input driving, policy-guided planning and decision-relevant modeling | Updated assessment below; no unsupported originality claim |
| A13 | Add a predicted-future intermediate P when making explicit forecasting claims | Proposed matched-interface L/P/F comparison; oracle improvement alone does not identify forecasting error |
| A14 | Independently switch visible objective-reference and hidden-road geometry before new oracle-policy training | Implemented/tested; [five-track pilot completed](INTERVENTION_RESULTS_2026-10-29.md) with negative driving effects; no visual/RL-causality claim |
| A15 | Audit prediction of recorded executed controls after the oracle driving failure, then isolate objective compatibility | Completed 15-window audit and reference-tail follow-up: better motion prediction alone did not ensure good selection |
| A16 | Preserve the visible reference exactly and change only its future tail, holding extent/sampling fixed | [Reference-continuation pilot](REFERENCE_CONTINUATION_2026-10-29.md) completed; +282.09 return and 0/5→4/5 completion versus tangent, still below local mean; [20-track replication](REFERENCE_REPLICATION_2026-10-29.md) complete: +286.25 versus tangent, −71.86 versus local; one primary counterexample, ten exact replays |

## Prior work and novelty assessment

These are **Prior work**, not project-origin ideas. Primary sources reviewed
2026-10-29; this search is not proof that an exact study has never been published.

| Work | Overlap / implication for this paper |
|---|---|
| [Learning by Cheating, CoRL 2019](https://arxiv.org/abs/1912.12294) | Learns a privileged driving teacher and transfers to a visual agent; separating perception and control through privileged inputs is established |
| [DriveAdapter, ICCV 2023](https://arxiv.org/abs/2308.00398) | Separates perception and planning and addresses the gap between predicted and true privileged inputs; supports our interface-distribution caution |
| [POPLIN, ICLR 2020](https://arxiv.org/abs/1906.08649) | Policy-guided online planning and policy distillation; search around learned proposals is established |
| [World on Rails, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Learning_To_Drive_From_a_World_on_Rails_ICCV_2021_paper.pdf) | Factorizes ego response from external evolution and distills model-derived choices; our exogenous/endogenous distinction has precedent |
| [Value Equivalence, NeurIPS 2020](https://arxiv.org/abs/2011.03506) and [Objective Mismatch, L4DC 2020](https://proceedings.mlr.press/v120/lambert20a.html) | Planning usefulness and predictive accuracy are different objectives; a task-useful model alone is not a new principle |
| [Valdi, July 2026 preprint](https://arxiv.org/html/2607.00917v1) | Direct CarRacing model-based planning reference; preliminary study with modified proprioception and 600-frame episodes, not a verified standard-task champion |
| [Seyam et al., October 26, 2026 preprint](https://arxiv.org/html/2610.35448v1) | Explicitly diagnoses planning failures despite detection on NeuroNCAP, then improves planning with targeted scenes; the broad proposed failure explanation already has close prior art |

The last preprint reports 95.9% of DiffusionDrive collision episodes with the
threat detected under its recall criterion. **Assistant critique:** this is a
detection-based diagnosis, not our proposed complete three-stage intervention;
it also does not establish that every relevant representation/prediction is correct.
Our contribution would need more precise controlled attribution or a substantive
new tradeoff/result, not simply repeating its perception-versus-planning headline.
No result from that traffic benchmark is transferred to our racing benchmark.

**Assistant assessment:** ingredient novelty is low; a carefully validated,
replicated decomposition with a useful limited-compute intervention has moderate
potential as an empirical paper. The current evidence is insufficient for a
strong-venue/SOTA claim. Interest increases if search gains persist under oracle
information, competent independently trained policies and useful time budgets,
and if an imperfect model recovers those gains. A single weak PPO plus expensive
search, or one simulator with no broader explanatory result, is a weak paper case.
No numerical acceptance probability is justified.

## Execution order and decision gates

**Assistant recommendation, not a new run declaration:**

**Execution update:** following U27, the [controlled intervention pilot](INTERVENTION_STUDY.md)
brings a bounded part of step 3 forward. Its four-cell geometry substitution is
now executable; the full V/L/F training interfaces, strong baselines and established
optimizer remain required. The manifest, not this historical list, declares its run.

1. Preserve the completed broad comparison and failed baseline probe. Review
   baseline implementation, optimization and original-protocol reproduction;
   stop treating single-factor PPO sweeps as sufficient baseline validation.
2. Finish the already audited established-search comparator and its bounded
   equal-work test. This separates the system idea from a particular search rule.
3. Specify and test V/L/F interfaces, privileged training boundaries, common
   representations and intervention sanity checks. Profile before assigning
   training budgets; do not bypass existing global-development-only guards.
4. Establish competent policies in the required rows; freeze them by standalone
   development performance. Run direct/random/search comparisons and a targeted
   interaction analysis, then independent-seed replications.
5. If it answers a measured limitation, compare simple road priors with a reused
   probabilistic predictor, or one task-useful approximate programmed model. Keep
   the planner/proposer fixed for attribution. Code evolution/GPU is conditional.
6. Audit the native-input comparison and freeze all methods, seeds, endpoints,
   uncertainty procedures and resource budgets. Only then execute reserved final
   evaluation once for the frozen study. Do not redesign another protocol from
   results on the same test tracks.

This ordering can change with public development evidence. Neither our private
candidate monitor nor a premature final test can be used to choose the path.
The exact additional budgets, equivalence margins and final baseline roster are
unresolved and must be declared before those experiments, not invented here.

## Outcomes that would change the paper

- If perception interventions dominate, report that and focus on reconstruction;
  do not force a decision-bottleneck story.
- If supplying future road eliminates the gap, information/uncertainty is central;
  poor hidden-road prediction may include irreducible ambiguity.
- If search still improves a competent oracle-informed policy, report remaining
  recoverable decision headroom at the measured resources, not "RL cannot control."
- If distillation closes the gap, explain how training can absorb the useful choices.
- If only a slow/exact model works, limit the practical claim. If a fast/inexact
  model succeeds, report the actual tradeoff without implying real-world transfer.
- If strong comparisons remove the advantage, retain that result and revise the
  claim. No combination of optional ideas is assumed to guarantee publication.

The full origin/history remains in [USER_INTENT](USER_INTENT.md) and
[RESEARCH_LOG](RESEARCH_LOG.md); access rules are in [EXPERIMENT](EXPERIMENT.md).
This blueprint is the single starting point for assembling the paper, not a
replacement for those source records or an unsupported declaration of novelty.
