# Matched sequence shooting and policy-guided iCEM

[Index](../README.md) · [Specification](EXPERIMENT.md) · [Source audit](SEARCH_OPTIMIZER_AUDIT.md)

## Origin, purpose and current status

**User instruction:** continue the experiments autonomously and document decisions,
findings and their reasons. The user's core idea is frozen behavior proposals
selected by a programmed simulator. **Prior-work origin:** iCEM's colored proposals,
elite fitting, cached within-decision elites and shifted cross-decision plans.
**Assistant design:** a budgeted policy-residual adaptation and a representation-
matched shooting control, keeping the existing local method as a separate reference.

**Validated and ready to execute the frozen stages; no driving result yet.** The preceding
[reference-tail replication](REFERENCE_REPLICATION_2026-10-29.md) supported objective
compatibility as a limitation but did not beat original local search. This experiment
returns to local information and addresses a missing established-optimizer comparison.
It does not assume adaptive optimization will help; it may exploit model errors.

## Decisions, reasons, confidence and alternatives

| Choice | Reason / attribution | Confidence and alternative |
|---|---|---|
| Same selected 1M-frame recurrent actor, local mask/ego/history, coded simulator, observer, tangent road prior and utility | Assistant attribution control for the user's search hypothesis | High need for matching; proposer competence remains unestablished. Stronger independently trained policies are a separate study |
| Full 20-step residual sequences around frozen feedback actions | Assistant adaptation: enables temporal colored noise and one-step shifting | Moderate; alternative is CEM over the old three knots, a different optimizer/representation study |
| Colored shooting and iCEM use identical 13+8 rollout batches | Assistant control for neural batching, model reconstruction overhead and actual work | High; compare the optimizer package, not a batching change. Single-batch shooting is a later efficiency variant |
| Exactly 21 evaluated trajectories: unchanged base +12+8 | Retain the existing 21-candidate budget; every injected/shifted trajectory occupies an evaluated slot | High accounting confidence; 21 is economical, not known optimal. No automatic 63-candidate extension |
| Noise exponent β=1 | Previously proposed economical initial setting; smooth temporal perturbations without forcing almost-constant plans | Low performance confidence; upstream generic default is 3 and task settings differ. No β sweep in this batch |
| Finite DFT filter, cutoff 1/H, unit theoretical marginal variance including DC | Assistant explicit sampling convention using NumPy; preserves Gaussian draws without per-trace rescaling | High implementation clarity; differs from upstream colorednoise normalization, so this is an adaptation |
| Residual scales 0.10 steering / 0.15 gas / 0.15 brake; clip residuals at ±3 original σ | Preserve prior channel scales while defining bounded proposal/elite support | Low confidence in optimality; unbounded residuals or action-range bounds are alternatives. Both new conditions share this choice |
| Four elites, smoothing 0.1, reuse fraction 0.3 (one elite) | Small-budget version of audited source; prevents fractional reuse rounding to zero | Moderate mechanics, low parameter optimality. Fit all elites from current batch plus one cached best at the same root |
| Reset σ every real decision; shift fitted mean, hold its last entry | Audited receding-horizon pattern; residual bias remains relative to the feedback policy | Moderate. No adaptive covariance, variance floor or changed model loss |
| Shift one best residual plan; append a fresh colored tail sample from a separate random stream | Retain temporal reuse while keeping ordinary proposal draws separate | Moderate. It is a residual plan, not a fixed absolute action sequence; always simulate again at the new root |
| Choose the best evaluated candidate across both batches; earliest index breaks ties | Assistant incumbent safeguard, including the unchanged base | High correctness; differs from strictly choosing only the final newly drawn population |

The two new methods share initial proposal draws at the first root. At later roots,
adaptive shifts and fitted distributions intentionally differ. Identical stream
indices after real trajectories diverge do not imply identical physical states.
The base policy, its weights and normalization stay frozen. Online fitting updates
only the search distribution, not the actor or dynamics model. This distribution is
over control residuals, not possible roads; the road prior remains the same single
deterministic tangent continuation. No learned probabilistic road model is introduced.

## Exact implementation contract

1. Process the real observation once through the recurrent actor. Reconstruct the
   same local road/world as before. Every candidate batch starts from copies of
   the same resulting recurrent memory and the same estimated vehicle root state.
2. Evaluate base plus twelve sampled residual sequences. Each sequence invokes the
   feedback actor on its own imagined local observations at steps 1…19.
3. Shooting draws the remaining eight from its unchanged distribution. iCEM fits
   four elites, draws eight from the fitted distribution, and puts the fitted mean
   in one of those eight evaluated slots. One cached best rollout is eligible for
   the next fit, without re-simulation at the unchanged root.
4. Choose the minimum-cost evaluated behavior, execute its first action for four
   official physics frames, and commit **its first-transition wheel estimate**.
   Owned snapshots prevent a winning first-batch state being replaced by the last
   batch. Never commit an imagined recurrent history or the final predicted state.
5. Shift mean/elite for the next decision only in the adaptive method. Costs from
   the previous real root are discarded. A shifted elite consumes one first-batch
   slot and is fully re-simulated under the new observed root/world.

Per real decision, both new methods perform **420 model transitions / 1,680 internal
native physics steps**, **400 actor-forward examples**, and **39 actor calls**.
The original knot search uses the same transition/example count but one rollout
batch (20 actor calls), so it is a representation/implementation reference rather
than the primary optimizer isolation. Actual episode totals can differ because
completion times differ; all costs are recorded, including last partial actions.

Programmed physics, official environment reward/termination, real action cadence,
imagined renderer and local information are unchanged. No global fields, privileged
candidate ranks/regrets, true wheel state, environment clones or final-test data
are available. The mathematical search uses predicted model costs only.

## Staged run declaration

Freeze both stages before the first new race. The full comparison follows the
smoke stage only if replay, finite-output and accounting checks pass, **regardless
of race score**. A code/configuration repair requires a new declaration; do not
overwrite either stage or quietly change settings after seeing the first two tracks.

| Stage | Newly run conditions | Native-frame cap | Internal native physics cap |
|---|---|---:|---:|
| Smoke / parity | Original knot search ×1 episode; colored shooting ×2; iCEM ×2 | 5,000 | 2,100,000 |
| Main comparison | Colored shooting ×20; iCEM ×20 | 40,000 | 16,800,000 |
| Total | 45 episodes, zero new experiment training | 45,000 | 18,900,000 |

The original knot episode must replay the pinned historical episode exactly before
new-method smoke races. The first two main episodes per new method must match its
smoke episodes exactly. Timing is excluded from equality. These four main replay
checks use episodes already in the main budget. Source/checkpoint/dependency hashes,
full configs, ordered development seeds and child commands are frozen and retained.

Frozen manifests: [smoke](../configs/campaigns/sequence-smoke-2026-10-29.json),
SHA256 `def709dfbdca894fe131763d229269b815a1268dcac59bf576c56a20fd4549fb`; [main comparison](../configs/campaigns/sequence-comparison-2026-10-29.json),
SHA256 `69a629e324cf6931daf13dd6203ec720f233d0c22b989dcf117e07e215ee0018`. Campaign IDs are `sequence_smoke_20261029` and
`sequence_comparison_20261029`; public reports use the corresponding
`reports/sequence_smoke_2026-10-29` and `reports/sequence_comparison_2026-10-29` directories.

```bash
uv run python scripts/run_sequence_search.py --manifest configs/campaigns/sequence-smoke-2026-10-29.json
uv run python scripts/run_sequence_search.py --manifest configs/campaigns/sequence-comparison-2026-10-29.json
```

The main runner verifies that the frozen smoke prerequisite completed and its
production sources match. Conditions are never overwritten; replay checks compare
all recorded arrays except elapsed timing. Every decision's physics, actor-example,
cache and shifted-plan counts are checked before reports are published.

The first 20 development tracks are used in fixed order; the first two are the smoke
subset. All were previously used by historical controls. Report all20, smoke2 and
additional18 separately. These are development panels, not new final holdouts.
One selected actor and planning seed zero; independent learning/planning streams
remain necessary for stronger conclusions.

Primary contrast: **iCEM adaptation minus colored sequence shooting** in mean official
return. Co-report completion, progress, departures, lap times with their completion
denominator, sampled speed/braking, p50/p95/p99 latency, deadline misses and actual
model/actor work. Keep failures. Secondary comparisons with historical 21-candidate
three-knot search and the direct actor are explicitly labeled; they do not isolate
optimizer fitting alone. Historical latencies are not newly paired measurements.

Nominal paired-track bootstrap: 5,000 draws, seed zero, all20/additional18 only.
Smoke panels stay descriptive. No selection adjustment, no independent-training
uncertainty, no multiple-comparison correction or confirmatory/SOTA interpretation.
The primary adaptive package includes distribution fitting, mean injection and
memory; this run does not identify which individual iCEM feature causes an effect.

One CPU thread, serial campaigns, no competing project training/racing. The
environment waits for decisions; no real-time deadline fallback. Measure smoke
timing before stating a main ETA. No auto-expansion to 200 tracks, extra seeds,
larger budgets, privileged geometry or a hyperparameter sweep.

## Validation and provenance

**Prelaunch checks:** 100 tests passed in 36.10 s, three upstream SWIG warnings;
CLI entry points, 34 documentation/notice files and both manifest/artifact checks
passed. An added synthetic-record test initially omitted its temporary directory;
the fixture was corrected before any new racing. Production source/configuration
was unchanged by that fixture repair.

Focused tests passed: spectral variance/correlation for even/odd horizons, matched
initial draws, correct mean injection, bounds, exact evaluation counts, previous-root
cost invalidation, shifted-plan reevaluation, repeatability/reset, local-only actor
inputs, unmodified real memory, identical batch roots and correct first-batch state
commit. Existing local/intervention tests passed after extracting the shared rollout
routine. Full-episode historical replay remains a launch gate, not an assumed result.

The prior source and audit are pinned to revision
`98c1c1fe2bfc94a87e4658b6c793f4e81bc30203`; [attribution and MIT notice](../third_party/icem/README.md)
are retained. This is a policy-guided iCEM adaptation with declared differences,
not an unchanged original implementation or a claim that CarRacing was in that paper.

## Interpretation and next decision

If adaptation improves over matched shooting, the result supports usefulness of
that adaptive search package for this actor/model/budget. If it loses, retain the
failure and original method; improved predicted scores cannot substitute for actual
driving. If only the sequence representation differs from knots, attribute it to
that change and its associated implementation, not solely adaptive fitting.
Review all tracks and compute before choosing the next experiment. The stronger-policy,
independent-seed and trained V/L/F studies remain open whatever this result is.
