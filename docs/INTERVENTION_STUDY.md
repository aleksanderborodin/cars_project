# Controlled interventions: implementation and first pilot

2026-10-29. [Index](../README.md) · [Blueprint](PAPER_BLUEPRINT.md) ·
[Contract](EXPERIMENT.md) · [Status](STATUS.md)

**User origin:** separate current understanding, future prediction and action
selection; use oracle inputs and matched policy/search comparisons to find what
limits driving. The user now asks to execute this intervention study.
**Assistant refinement:** begin with a fixed-controller module substitution that
removes the confounding in the earlier full-map diagnostic, then implement the
separately trained information regimes. This first pilot does not replace the
full study or establish that a learned policy perceives correctly.

**Execution:** the pilot and a separately declared executed-action prediction
audit are complete. [Results](INTERVENTION_RESULTS_2026-10-29.md) show worse driving
in all oracle cells despite improved oracle-road prediction on recorded actions.
This motivated a reference/objective compatibility check before a large training launch.
The [subsequent reference-tail control](REFERENCE_CONTINUATION_2026-10-29.md) is now
complete and supports that mismatch explanation in this pilot, with performance
still below the local-search mean. The [20-track replication](REFERENCE_REPLICATION_2026-10-29.md)
retains the primary mean gain but also the local-search deficit and one primary
counterexample. Independent-stream/seed and competence controls remain necessary
before a large training matrix.

## First intervention: two independent geometry switches

The earlier full-map experiment changed both the road model and the objective's
reference path, using an engineered proposer. It did not identify either effect.
Here we keep the selected recurrent proposer, its actual inputs/history, residual
draws, physics/observer, score weights, horizon and candidate count fixed.

| Cell | Objective reference | Road outside current sensor circle |
|---|---|---|
| `baseline` | Existing mask reconstruction and tangent extension | Existing local reconstruction and tangent prior |
| `visible_reference` | True **visible** directed centreline and tangent extension | Same prior as baseline |
| `hidden_road` | Same objective reference as baseline | Oracle-derived road occupancy/margins |
| `both` | True visible directed centreline and tangent extension | Oracle-derived road occupancy/margins |

The complete road model retains the **original local raster inside the root
footprint**. Its outside-footprint cost flag also stays fixed in all conditions;
otherwise removing an uncertainty penalty would introduce another treatment.
The objective reference is independent of the path used to construct the road
prior. Changing the reference cannot silently change tire contact, road margins
or simulated policy observations. The same first-action residual bank is used
at a matched root; subsequent feedback actions may differ because imagined
observations differ. After real trajectories diverge, common decision-counter
random draws do not imply identical physical states or candidate trajectories.

The visible reference clips every directed centreline segment to the sensor
circle **before** joining visible components. It selects the nearest contiguous
component and resamples at the existing 1 m resolution. It never bridges pieces
through hidden arcs. An unavailable/too-short component falls back to the
baseline path and is counted. Both objective references use the existing 100 m
tangent extension; this is a reference-construction default, not a learned road
forecast or a claim that 100 m is an optimal planning horizon.

**What this diagnoses:** sensitivity of this fixed search controller to visible
reference reconstruction, hidden-road information, and their interaction.
The true visible centreline supplies topology/direction labels absent from the
binary raster. It is therefore a privileged representation intervention, **not
a clean removal of visual perception error**. The hidden-road raster is derived
from exact geometry but still uses 1 m sampling/interpolation; the vehicle/contact
model remains approximate. A remaining gap cannot be assigned solely to RL.

## Information boundary and sanity checks

- Use the existing explicitly privileged `global` development adapter. Its
  training/calibration guard stays closed. No new hidden-map access for local APIs.
- The intervention adapter strips map channels before calling the real actor.
  Imagined actor inputs also have exactly the original local schema and footprint.
- No exact simulator clone, wheel/solver truth, track seed/index, progress label,
  scenario identity or private candidate-monitor stream enters the actor/model.
- Original real actor memory advances once per actual observation. Branch memory
  remains isolated. Frozen network weights and evaluation normalization stay fixed.
- The negative-control cell must reproduce the previous five search-7 episodes
  exactly in observations, actions, rewards, termination and departures before
  the other cells run. This checks complete driving, not only average scores.
- Synthetic/constructed-input positive controls verify that the hidden-road
  switch actually changes outside occupancy and the reference switch actually
  changes the objective path. Tests verify no cross-effect on road queries and
  unchanged real/root imagined local observations.
- Changing geometry solely outside the visible component cannot change the
  current-only reference. Clipping, disjoint components, direction, wraparound,
  missing-reference fallback and recurring memory are tested.

These checks validate implementation isolation. They do not establish causal
identification of internal neural perception/forecasting modules, nor suitability
of a particular score function. No real counterfactual candidate ranks are used.

## Declared first development pilot

[Manifest](../configs/campaigns/intervention-pilot-2026-10-29.json).
Five existing development tracks in manifest order, four cells, one already
selected recurrent checkpoint, planning seed 0; seven candidates, 20 decisions
(1.6 seconds), action repeat 4. Residual scales and all costs are unchanged.
Sequential conditions, one CPU thread, frozen executable source. No training
or other racing job during timing. Ordinary workstation load is uncontrolled.

- Maximum **20,000 native evaluation frames**, **2,800,000 internal native
  physics steps**, zero new experiment training frames. Ordinary test fixtures
  train tiny disposable models; they are infrastructure checks, not paper models.
- Run all four cells even if an intervention hurts. Stop for correctness, changed
  artifacts, invalid outcomes or resource violations, never for a disappointing score.
- Report every per-track return/completion/departure outcome, latency and fallback.
  No five-track significance, component-adequacy or SOTA inference.
- Predeclare visible-reference effect with/without oracle road, hidden-road effect
  with/without oracle reference, and interaction `(both-hidden)-(visible-baseline)`.
  These are paired whole-episode effects, not oracle candidate regrets.
- No automatic expansion based on favorable outcomes. A larger panel needs a
  new immutable declaration after implementation/coverage review; these five
  tracks stay development and do not become a holdout.

Executable command:

```bash
uv run python scripts/run_intervention.py --manifest configs/campaigns/intervention-pilot-2026-10-29.json
```

Each condition is also available through `research evaluate --method
policy-intervention`, with `--reference reconstructed|oracle_visible` and
`--continuation prior|oracle`. The method requires a matching local actor and
the privileged development adapter. It cannot launch a reserved final test.

## The full intervention study still required

1. **Validate and replicate module substitutions.** Complete this pilot, inspect
   fallbacks/coverage and failures, then declare a larger panel and planning-stream
   replications. Retain all cells. Keep the established-optimizer comparison and
   direct-policy competence audit as separate tasks; neither is satisfied here.
2. **Define matched V/L/F training interfaces.** V is fixed-footprint local RGB
   with declared ego/history; L replaces current geometry with ideal visible
   geometry; F adds future road. Use the same action frequency, ego sensors,
   permitted history and official task. Do not alter the existing global guard
   to allow training until a new versioned interface and partition tests exist.
   Full-map geometry is not complete vehicle state.
3. **Train competent policies, not just equal-duration failures.** Profile each
   regime, declare learning/tuning budgets and standalone checkpoint selection;
   compare learning curves and independent seeds. Paired direct/random/search
   comparisons use the exact same checkpoint within a row. Across retrained rows,
   declare architecture, label and optimization changes rather than treating
   score differences as pure module errors.
4. **Use matched module interfaces for stronger attribution.** Train downstream
   heads for the relevant input distributions; include unchanged-input and
   degraded-input controls. Add L+predicted future (P) through the same interface
   as oracle F if making forecasting-quality claims. Oracle-vs-local alone
   measures information value, including potentially irreducible uncertainty.
5. **Measure the alternatives to our preferred story.** Strong direct policies,
   established equal-work planning, cost curves and permitted search-action
   distillation test whether gains reflect baseline weakness or training targets.
   Distillation must never use private oracle labels. No claim that a learner
   cannot reproduce search follows merely from our current policy losing.
6. **Freeze before final tests.** Five independent training seeds for headline
   learned comparisons; common reserved test tracks only after all methods,
   hypotheses, contrasts, resource budgets and uncertainty procedures are frozen.
   Set practical equivalence margins from task meaning/development before using
   small oracle gaps to claim conditional adequacy. They remain unresolved.

**Confidence:** high in the need for independent switches, input isolation,
replay controls and paired within-checkpoint comparisons. Moderate in this first
geometry diagnostic's usefulness. Low/unknown in policy competence, the eventual
dominant bottleneck, useful planning budget and any SOTA outcome. Five tracks,
seven candidates and the existing utility are economical pilot choices by the
assistant, not user requirements or claimed optima. Visual implementation,
oracle-informed policy training and the modern-comparator roster remain open.

The user's probabilistic road models, useful approximate simulator, downstream
predictor selection and optional agentic evolution remain in the
[attributed blueprint](PAPER_BLUEPRINT.md); this pilot does not silently activate them.
