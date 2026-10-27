Task 2 of this research codebase. Read AGENTS.md, docs/EXPERIMENT.md and docs/TASK1.md first; task 1 (env, metrics, splits, eval runner, baseline expert, Rust ABI) is done and tested. Reuse its APIs; do not change env/metrics semantics.

Implement the learned Python methods for setting A (state observations, StateObs vector):

1. src/cars/policies/torch_mlp.py: MLP policy (obs -> steer, gas, brake with correct bounds), save/load, implements the Policy protocol so the eval runner can evaluate it.
2. src/cars/il/bc.py: collect expert demonstrations with BaselinePolicy on a seen-K split (each episode a seed from the split), train behaviour cloning (MSE or Gaussian NLL, observation normalisation stored with the model), save to runs/il_bc/<run_id>/.
3. src/cars/il/dagger.py: DAgger with a beta schedule, the learner drives, the baseline labels visited states; aggregate the dataset; count expert labels as cost and record it.
4. src/cars/rl/train.py: Stable-Baselines3 PPO and SAC on the StateObs env. Training env samples a seed from the seen-K split at each reset (seen-inf: fresh seed). OFFICIAL reward only, no shaping. SubprocVecEnv (CPU has 12 cores), VecNormalize for obs (save stats with the model), standard TimeLimit 1000. Sensible defaults for CarRacing-state (PPO: n_envs 12, gSDE optional; SAC: device cuda if available). Periodic evaluation on a seen-split during training ONLY (never unseen/hard). Log tensorboard and env steps as cost. Save model + config.json with all hyperparameters and the seed.
5. A wrapper so trained SB3 models and MLP policies can be evaluated with cars.eval.runner (method keys il_bc, il_dagger, rl_ppo, rl_sac).
6. CLI commands: `train-bc`, `train-dagger`, `train-rl --algo ppo|sac`, each with --split seen-K, --seed, --steps/--episodes, --run-id; and `eval --method il_bc|il_dagger|rl_ppo|rl_sac --model-path ...`.
7. Tests: short smoke tests (tiny budgets) for each trainer + save/load + eval through the runner, determinism of a saved deterministic policy.

Then run small smoke trainings only (a few minutes total; e.g. BC on seen-10, PPO 50k steps) and evaluate them on seen-10 to show the pipeline works end to end. Do NOT run full trainings, do NOT evaluate on unseen or hard, do NOT git commit. Write docs/TASK2.md with commands and a recommended full-training budget per method (env steps / episodes, estimated wall time on this machine: 12 CPU cores, RTX 5060 Ti; state env ~5700 steps/s single process). Run `uv run pytest -q` before finishing. End with a short report: files, tests, smoke numbers, deviations.
