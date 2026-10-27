# Task 2: learned state policies

Implemented BC, DAgger, PPO, and SAC for StateObs v1 (74 float32 entries).
Task 1 environment, reward, termination, physics, metrics, and split definitions
are unchanged. All training and smoke evaluation used `seen-*` only.

## Commands

Use Python 3.12 and `uv run`. These are the actual smoke-training commands:

```bash
uv run python -m cars.cli train-bc --split seen-10 --seed 0 --episodes 30 --epochs 20 --run-id task2_smoke
uv run python -m cars.cli train-dagger --split seen-10 --seed 0 --episodes 12 --iterations 3 --bootstrap-episodes 3 --epochs 10 --run-id task2_smoke
uv run python -m cars.cli train-rl --algo ppo --split seen-10 --seed 0 --steps 50000 --n-envs 12 --n-steps 256 --n-epochs 5 --eval-freq 25000 --eval-episodes 2 --run-id task2_smoke
uv run python -m cars.cli train-rl --algo sac --split seen-10 --seed 0 --steps 2000 --learning-starts 500 --batch-size 64 --buffer-size 10000 --eval-freq 1000 --eval-episodes 2 --run-id task2_smoke

for method in il_bc il_dagger rl_ppo rl_sac; do
  uv run python -m cars.cli eval --method "$method" --model-path "runs/$method/task2_smoke" --split seen-10 --n 10 --out "runs/$method/task2_smoke"
done

uv run pytest -q
uv run tensorboard --logdir runs
```

Run IDs cannot be reused for training. Choose a new ID when repeating commands.
`--runs-dir` changes the parent directory (default `runs`). Evaluation accepts a
run directory or a model file. PPO/SAC require the matching `vecnormalize.pkl`
beside that file. For a best checkpoint, pass `best/best_model.zip` explicitly.
The final model, rather than the best checkpoint, was used in the table below.

IL accepts an episode cap, a step cap, or both; collection stops at the first
cap. With neither cap it collects 100 episodes. A step cap may end collection
mid-episode; history marks this as incomplete without changing the environment's
TimeLimit. DAgger distributes the budget over bootstrap and aggregation rounds;
the episode cap must cover bootstrap plus at least one episode per round.
RL accepts `--steps`, or `--episodes` as a nominal `episodes * 1000` decision budget.
It does not stop at an exact completed-episode count. PPO completes its final
rollout, so actual decisions can exceed the requested budget; SAC rounds to a
vector step/train-frequency boundary. Actual cost is recorded.

## Models and training behavior

`MLPPolicy` implements `reset(track)` and `act(CarState, StateObs)`. The default
network is 256/256 ReLU, with tanh steering and affine tanh gas/brake, giving
action bounds `[-1, 1]`, `[0, 1]`, `[0, 1]`. Dataset observation means and population
standard deviations are stored as model buffers. Standard deviation has a 0.01
floor and normalized inputs are clipped at 10. IL uses action MSE, Adam at 0.001,
30 epochs and batch size 256 by default. Inference is deterministic on CPU.

BC samples a seen seed for each expert rollout. DAgger first collects expert
bootstrap rollouts, then labels every visited state with `BaselinePolicy`.
At each subsequent step, the expert action is executed with probability beta;
otherwise the learner drives. Beta decreases linearly from 0.5 to 0 by default.
A single aggregation iteration uses the ending beta. Every queried label counts
as cost, including bootstrap and expert-executed steps. Each round refits a new
MLP on all labels using `seed + round`; this permits re-estimating normalization
without invalidating previously trained weights. The default is ten aggregation
rounds after ten bootstrap episodes, within a total 100-episode budget.

PPO/SAC use `SubprocVecEnv` with spawn, Task 1's standard 1000-step environment,
Monitor, and VecNormalize for observations only (`norm_reward=False`). Finite
seen splits are sampled uniformly with replacement at each reset. `seen-inf`
draws fresh seeds >= 1,000,000 without replacement within each worker; disjoint
worker residue classes prevent overlap across workers. SB3's initial seed plus
rank initializes each worker stream. All streams are reproducible.

Both RL algorithms support two optional agent interfaces:

- `--action-repeat N` (positive integer, default 1) holds each action for up to N
  environment frames and sums their official rewards for the training transition.
  Repetition stops immediately on termination or truncation. The inner TimeLimit
  remains 1000 frames: N=4 gives 250 decisions for a full-length episode. Gamma
  remains the configured per-decision discount, without exponentiation.
- `--throttle-axis` (default off) exposes `Box(-1, 1, (2,))` actions
  `[steer, throttle]`. Gas is `max(throttle, 0)` and brake is `max(-throttle, 0)`.
  Without the flag, the original three-control action space is used.

With action repeat, SB3 budgets and schedules (`--steps`, `--n-steps`,
`--eval-freq`, warmup and train frequency) count agent decisions. `cost.json`
records actual training `env_steps`, `decision_steps`, requested decision budget,
nominal requested frame budget (`steps * action_repeat`), and actual evaluation
frames separately. Monitor stays inside the interface wrappers, preserving
official episode returns and frame lengths. VecNormalize sees decision-frame
observations only. Periodic evaluation uses the identical interface wrappers
and frozen observation statistics.

Both settings are stored in `config.json` under `hyperparameters`. Loading a
saved model (including `best/best_model.zip`) automatically restores them. Missing
keys in older configs mean repeat 1 and the original three-control interface.
The evaluation runner still steps and logs every environment frame: the SB3
adapter normalizes observations and predicts only on decision frames, holds the
mapped action between decisions, and clears the hold on every episode reset.
Reward, termination, physics, metrics, and the per-frame log format are unchanged.

PPO defaults: 12 environments, CPU, 1024 rollout steps per environment, ten
epochs, batch 256, learning rate 0.0003, gamma 0.99, GAE lambda 0.95, clipping 0.2,
separate 256/256 tanh actor/value networks; `--use-sde` enables gSDE. SAC defaults:
one subprocess environment, CUDA when available, 256/256 ReLU networks, batch 256,
1M replay capacity, 10k warmup steps, one gradient update per environment step,
automatic entropy, tau 0.005, gamma 0.99 and learning rate 0.0003. All remaining
SB3 algorithm/policy constructor defaults are captured in new run configs.
Use the typed config classes for parameters beyond the common CLI options.

Periodic deterministic RL evaluation uses the same seen split, ten episodes
every 50k training steps by default. Finite evaluation splits cycle their seeds,
so seen-10's default ten episodes cover all ten tracks. Evaluation freezes and
synchronizes observation statistics, and records official returns and success.
Best model checkpoints save matching normalization statistics. This follows
the [SB3 normalization/evaluation workflow](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#pybullet-normalizing-input-features).

## Artifacts and evaluation

Each `runs/<method>/<run_id>/` contains:

| File | Contents |
|---|---|
| `config.json` | Training seed, full public hyperparameters, implementation defaults, environment/schema, package versions, source and lock hashes |
| `model.pt` | IL architecture, weights, normalization buffers, method, training seed and observation schema |
| `dataset.npz`, `history.json` | IL observations/actions, per-label seed/round, episode seeds, expert config, round costs and losses |
| `model.zip`, `vecnormalize.pkl` | Final SB3 model and matching observation statistics |
| `monitor/`, `tensorboard/` | RL episode returns and scalar logs, including actual training/evaluation step costs |
| `training_eval/evaluations.npz` | Periodic RL evaluation returns, episode lengths, successes and training steps |
| `best/best_model.zip`, `best/vecnormalize.pkl` | Best periodic seen-evaluation checkpoint and synchronized statistics |
| `cost.json` | IL label count or actual RL training frames and decisions, separately counted RL evaluation frames, wall time |
| `eval/<split>/` | Task 1 trajectory logs, metrics CSV, evaluation config when evaluating inside a training run |

Evaluation preserves an existing training config, writes a separate evaluation
config, and refuses to replace an existing evaluation for that split. It verifies
the policy belongs to that training run. A separate new output root continues
to work as in Task 1. Method keys and training seeds propagate to CSV/logs for
subsequent reporting. Adapters are deepcopyable/pickleable for spawn workers;
SB3 inference loads lazily on CPU and never updates normalization statistics.

Programmatic entry points:

```python
from cars.il.bc import BCConfig, train_bc
from cars.il.dagger import DAggerConfig, train_dagger
from cars.rl.train import RLConfig, train_rl
from cars.policies.sb3 import load_learned_policy
from cars.eval.runner import evaluate

# Reproduction: read config.json["hyperparameters"] into its method's config class,
# preserve checkout/uv.lock, and call the corresponding trainer with a fresh ID.
policy = load_learned_policy("il_bc", "runs/il_bc/task2_smoke")
evaluate(policy, "seen-10", 10, "runs/il_bc/task2_repeat", workers=2)
```

## Measured smoke results

One training seed (0), final deterministic models, evaluation seeds 0–9, standard
1000-step limit. Returns use mean ± sample standard deviation across episodes.
Training wall times exclude separate final evaluation and interpreter startup;
the RL times include worker startup and periodic evaluation. Small runs overlapped
briefly, so these timings are approximate.

| Method | Training cost | Training wall time | Seen-10 return | Success | Finished lap time, mean |
|---|---:|---:|---:|---:|---:|
| BC | 23,996 labels / 30 episodes | 8.83 s | 764.55 ± 316.89 | 7/10 | 15.81 s |
| DAgger | 9,422 labels / 12 episodes | 5.63 s | 920.06 ± 9.49 | 10/10 | 15.99 s |
| PPO | 52,224 training steps | 13.05 s | -49.03 ± 32.67 | 0/10 | — |
| SAC | 2,000 training steps | 11.26 s | -54.38 ± 35.56 | 0/10 | — |

PPO requested 50k steps and finished 17 rollouts of 12 × 256 steps. PPO and SAC
each used another 4,000 steps for periodic seen evaluation. SAC used CUDA on the
RTX 5060 Ti. These budgets verify the pipeline; they do not support a comparison
of trained RL and IL performance. No held-out evaluation was run.

### Agent-interface follow-up smoke

Executed separately from the active full-training runs:

```bash
uv run python -m cars.cli train-rl --algo ppo --split seen-10 --seed 0 --steps 50000 --n-envs 4 --n-steps 256 --n-epochs 5 --eval-freq 25000 --eval-episodes 2 --action-repeat 4 --throttle-axis --run-id task2_interface_smoke_s0
uv run python -m cars.cli eval --method rl_ppo --model-path runs/rl_ppo/task2_interface_smoke_s0 --split seen-10 --n 10 --out runs/rl_ppo/task2_interface_smoke_s0
```

PPO completed 50,176 decisions / 200,704 environment frames, with another 4,000
frames for periodic evaluation. Training wall time was 28.25 seconds. Gamma
remained 0.99. The final deterministic model on seen seeds 0–9 returned
**277.27 ± 116.92** (sample standard deviation), finished **0/10** laps, and had
no crashes. All episodes reached the unchanged 1000-frame limit. Mean off-track
events were 2.5 and mean off-track frames were 683.5. This small smoke validates
the interface and saved-model evaluation; it does not establish a driving-quality
improvement. Artifacts and frame logs are under
`runs/rl_ppo/task2_interface_smoke_s0/`. The active full-training run directories
were not modified; no long training, held-out evaluation, or git commit was run.

## Recommended full-training budgets (not executed)

Per training seed on 12 CPU cores and an RTX 5060 Ti:

| Method | Suggested initial budget | Estimated wall time per seed |
|---|---|---:|
| BC | 1,000 expert episodes, at most 1M labels; 30 epochs | 5–10 min |
| DAgger | 500 total episodes: 50 bootstrap + 10 × 45 aggregation; at most 500k labels; 20 epochs per refit | 10–20 min |
| PPO | 5M env steps, 12 envs, 1024 steps/env, ten epochs, batch 256 | 15–40 min |
| SAC | 2M env steps, one env, CUDA, batch 256, one update/step | 2–5 h |

These are planning estimates, not measured full-run timings. The raw state env's
~5700 steps/s implies about 175 seconds per million steps in one process before
policy inference, expert queries, IPC, optimization, and evaluation. The estimates
also use the smoke timings above; SAC is dominated by gradient updates, while
DAgger repeatedly fits the growing aggregate dataset. Benchmark a longer seen-only
pilot before reserving machine time. Use five training seeds per method/split as
specified by EXPERIMENT.md; the table is per seed. Select budgets and methods
using seen results only.

Example full-run commands, intentionally not executed:

```bash
uv run python -m cars.cli train-bc --split seen-10 --seed 0 --episodes 1000 --epochs 30 --run-id full_s0
uv run python -m cars.cli train-dagger --split seen-10 --seed 0 --episodes 500 --bootstrap-episodes 50 --iterations 10 --epochs 20 --run-id full_s0
uv run python -m cars.cli train-rl --algo ppo --split seen-10 --seed 0 --steps 5000000 --run-id full_s0
uv run python -m cars.cli train-rl --algo sac --split seen-10 --seed 0 --steps 2000000 --run-id full_s0
```

## Validation and deviations

`uv run pytest -q`: **48 passed**, with three upstream SWIG deprecation warnings.
The new tests train all four methods with tiny budgets, exercise actual optimizer
updates, saved actions and episode determinism, action bounds, frozen normalization,
expert-label accounting, beta-zero learner rollouts, seen-only sampling/validation,
best-checkpoint statistics, training-config preservation, and spawn evaluation.
The agent-interface tests additionally cover summed rewards, full and partial
repeats at termination/TimeLimit, signed throttle bounds, legacy config defaults,
checkpoint option restoration, actual frame-cost accounting, and identical
per-frame runner trajectories versus the wrapped training interface for both
PPO and SAC. They count normalization/prediction calls to verify held frames do
not trigger observations or decisions, and verify episode resets clear holds.

No scope deviations. The explicit choices are action MSE, mixture-policy DAgger
with aggregate refitting, CPU PPO, and a single CUDA SAC training environment.
RL `--episodes` denotes a nominal step budget, as described above. No full
training, unseen/hard seed/result access, environment-semantic edits, or git
commit was performed.
