# Recurrent baseline audit before longer training

**Completed native execution:** the [4M one-seed campaign](NATIVE_BASELINE_RESULTS_2026-10-29.md) selects 1M (843.82 return, 38% completion), while final 4M regresses to 645.80/4%. All 200 evaluation traces and source/resume accounting pass. The strong-baseline gate remains open. The next [inference-device diagnostic](NATIVE_DEVICE_AUDIT_2026-10-29.md) checks GPU training versus CPU deployment without changing the original selection. Earlier local-driver findings below retain their original interpretation.

Updated 2026-10-29. [Next experiments](NEXT_EXPERIMENTS.md) ·
[References](REFERENCES.md) · [Status](STATUS.md)

Status: the listed training-pipeline fixes and CPU/GPU profiling are implemented;
the first configured local-driver block completed 2M frames with a later
regression. The predeclared selection rule keeps the 1M checkpoint; training has
stopped at its health review. This is not yet a converged strong comparator. See the
[execution report](GEOMETRY_AND_DRIVER_2026-10-28.md). This is still not an original
pixel-recipe reproduction. The previous 100k-frame local policy remains an
engineering pilot, not a competitive reference.
The subsequent fresh `target_kl=0.02` probe also completed: best 697.04/4%, below
the original selected driver. It is not adopted; baseline repair remains open.
See the [follow-up report](FROZEN_POLICY_RESULTS_2026-10-28.md) for all checkpoints
and the substantial reduction in optimizer work caused by early stopping.
The lower-learning-rate (3e-5) probe also completed: best 692.18/8%, below the
original 858.11/30% on the same 50 tracks. Neither repair is adopted. See the
[completed broad review](BROAD_RESULTS_2026-10-29.md). The current
[fixed-controller geometry intervention](INTERVENTION_STUDY.md) diagnoses this
one selected actor/search pipeline; it does not establish baseline convergence.
The [next-moves plan](NEXT_MOVES.md) now specifies the proposed implementation,
profiling and training order. Its first driver is a practical proposer/comparator,
not an asserted current CarRacing SOTA method.

## Source and observed differences

Pin the [RL Zoo configuration](https://github.com/DLR-RM/rl-baselines3-zoo/blob/61956d90d9f8ac126043ed4024efa12a462d97e1/hyperparams/ppo_lstm.yml)
at commit `61956d90d9f8ac126043ed4024efa12a462d97e1`. Its CarRacing-v3 entry
specifies eight environments, 512 rollout steps per environment, ten optimization
epochs, a 128-unit LSTM, state-dependent exploration and a linear learning-rate
schedule starting at 1e-4. Input is resized to 64×64 grayscale with a two-frame
stack. Reward normalization is enabled; observation normalization is disabled.
The budget is four million learner steps. The frame-skip wrapper is commented
out in that entry. The factory audit below now resolves its default action cadence;
an executed original-task reproduction is still outstanding.

The earlier engineering adapter uses four environments, 128-step rollouts, five epochs,
a 64-unit LSTM, constant 3e-4 learning rate and default Gaussian exploration. Its
local encoder consumes road/visibility masks, scaled ego sensors and previous
action. Its native variant consumes RGB plus previous action. Neither variant
implements the upstream preprocessing or training configuration. The existing
100,349-native-frame run therefore does not reproduce the upstream baseline.
The new `research train-driver` path implements the configurable settings and
checkpoint changes below. Its local encoder and repeat-four action rate still
make it an adaptation, not a reproduction of the native pixel recipe.

The [official recurrent-PPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
requires maintaining recurrent state and episode-start flags during inference.
Our adapter implements these and tests initial/recurrent reload equivalence.
This validates state handling, not sufficient learning or faithful reproduction.

## Factory and budget audit, 2026-10-29

**Origin and purpose:** the user wants improvement over a strong frozen proposer;
the assistant audited the recipe and our existing artifacts to avoid interpreting
an early training failure as a general limitation of learned control. No new
training, evaluation or setting change was made for this audit. Exact source URLs,
file hashes, installed versions and artifact hashes are in the
[public audit record](../reports/references/rlzoo_factory_audit_2026-10-29.json).

The pinned [environment factory](https://github.com/DLR-RM/rl-baselines3-zoo/blob/61956d90d9f8ac126043ed4024efa12a462d97e1/rl_zoo3/exp_manager.py)
applies the configured resize/grayscale wrappers, reward normalization, frame
stacking and image-axis conversion. The audited default CarRacing path adds no
action repetition. The resize helper only converts the shape argument. Installed
SB3's generic vector factory adds monitoring and the requested wrappers; installed
Gymnasium takes one vehicle/world step per external action. This conclusion covers
that pinned default path with our recorded dependencies, not arbitrary command-line
overrides, third-party registry overrides or an unspecified historical installation.

Recurrent PPO increments its learner counter by the number of environments on
each vector step: four million is **aggregate across eight workers**, not four
million per worker. With 8×512 transitions per full rollout, an uninterrupted
four-million-step run would round up to 4,001,792 external steps in 977 rollouts.
This is a source-derived count, not a run we performed. Evaluation and the physics
step inside each reset are additional work, excluded from that interaction count.

| Quantity | Reference recipe, inferred | Selected local checkpoint, measured | Stopped local run, measured |
|---|---:|---:|---:|
| Action repeat | 1 | 4 | 4 |
| Learner decisions | 4,000,000 requested | 251,904 | 501,760 |
| External native frames | ≈4,000,000 requested | 1,007,379 | 2,006,414 |
| Completed rollout updates | 977 at full rollout rounding | 123 | 245 |
| Fraction of nominal reference decision budget | 100% | 6.30% | 12.54% |

The local `updates` artifact counts epochs entered (1,230/2,450 here), not
independent rollout updates or minibatch gradient steps. Four local workers also
halve the data per rollout relative to eight reference workers. These figures
describe differences; they do not prove that extra training would solve regression.

Both recipes use γ=0.99 **per learner decision**, so the physical discount differs.
For discounting alone, four upstream steps correspond to γ⁴=0.96059601; our local
repeat-four recipe uses 0.99. SDE resampling every four decisions also covers up to
16 native frames locally versus four in the reference. Merely changing γ would
not restore parity: reward aggregation within repeats, GAE, observation history,
encoder, update batch and noise cadence also differ. The selected local checkpoint
is only 12.59% through its declared 8M-native-frame learning-rate schedule; the
stopped run is 25.08% through it. Even that full local cap would provide about 2M
decisions, rather than the reference's 4M, subject to early episode endings.

**Decision:** retain the existing actor as a frozen mechanism-study proposer and
keep its positive search results with this limit attached. Do not label it a
converged baseline, silently resume the failed run, or change its discount under an
existing run ID. Before a separate native-recipe experiment, implement and verify
the exact pixel/stack/action interface with partition-enforced training, profile
throughput, then freeze its full decision budget and checkpoint-selection rule.
Use the same local information and action rate for a later stronger local proposer;
declare any discount/budget repair as its own adaptation. High confidence in the
accounting correction; unknown whether any proposed repair will improve driving.

## Implemented prerequisites for the first configured baseline campaign

1. Make training settings explicit/configurable rather than hard-coded. Log the
   entire resolved algorithm and encoder configuration, package versions, action
   scaling, reward transformation, discount per decision and decision frequency.
2. Add declared learner-side reward normalization with saved running statistics.
   Evaluation reports original simulator reward; statistics must not update from
   development episodes. Record whether the policy needs any observation transform.
3. Save scheduled checkpoints and learning diagnostics during a single declared
   training budget. A linear schedule must advance on that overall budget, not
   restart or reach zero at every intermediate milestone. Continuation must
   preserve optimizer, normalization, RNG and schedule progress, while documenting
   fresh episode resets instead of exact mid-physics continuation.
4. Test action bounds, recurrence, transforms and reproducible reload; profile
   throughput and peak memory. Choose CPU/GPU and vector-worker count using that
   profile, before promising a runtime or running multiple seeds.
5. Use a recorded train stream and the same fixed development tracks for selection.
   Log actual native frames, decisions, updates, trial count and validation cost.
   Preserve failed configurations. Test performance never determines tuning.

## Two separate result families

- **Matched local adaptation:** keep the local information contract and repeat 4.
  Adapt the encoder; declare all departures from the upstream configuration. The
  updated proposal is one training seed with an 8M-native-frame maximum, an initial
  health review at 2M, and checkpoints at 0.2/0.5/1/2/4/8M. The global schedule must
  span the declared maximum from the outset; do not restart it at each milestone.
  Compare learning curves and completion, not only a favorable checkpoint. This
  proposed cap does not establish convergence or guarantee a competent baseline.
- **Original-task recipe reproduction:** audit the upstream factory and reproduce
  its pixel preprocessing, control rate, architecture and full training budget.
  Account explicitly for our disjoint train/development seed procedure and version
  differences. Our repeat-4 native smoke is not this experiment. Keep original-task
  results separate from local-information adaptations.

These steps do not choose the final paper's strongest comparator. Valdi remains
the priority actor-free reproduction audit, followed by a feasible modern
world-model comparison from the reference roster. Competitive performance and
novelty must survive those comparisons; they cannot be inferred from our weak
initial recurrent pilot.
