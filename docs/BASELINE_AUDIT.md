# Recurrent baseline audit before longer training

Updated 2026-10-28. [Next experiments](NEXT_EXPERIMENTS.md) ·
[References](REFERENCES.md) · [Status](STATUS.md)

Status: source/configuration audit only. No new baseline training is part of the
control-audit batch. The previous 100k-frame local policy remains an engineering
pilot, not a competitive reference.
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
out in that entry; audit the full factory before declaring the native-frame count.

Our current adapter instead uses four environments, 128-step rollouts, five epochs,
a 64-unit LSTM, constant 3e-4 learning rate and default Gaussian exploration. Its
local encoder consumes road/visibility masks, scaled ego sensors and previous
action. Its native variant consumes RGB plus previous action. Neither variant
implements the upstream preprocessing or training configuration. The existing
100,349-native-frame run therefore does not reproduce the upstream baseline.

The [official recurrent-PPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
requires maintaining recurrent state and episode-start flags during inference.
Our adapter implements these and tests initial/recurrent reload equivalence.
This validates state handling, not sufficient learning or faithful reproduction.

## Required implementation before the next baseline campaign

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
