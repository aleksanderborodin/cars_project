"""SB3 state-only PPO/SAC. Reward normalization and reward shaping are disabled."""
from dataclasses import dataclass, replace
from functools import partial
import inspect
from pathlib import Path
import time

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import torch

from cars.env.make import make_env
from cars.rl.interface import agent_interface
from cars.training import TrainingSeeds, create_run, require_seen, seed_everything, write_json


def _constructor_defaults(cls: type) -> dict:
    """Freeze SB3's implicit algorithm/policy defaults alongside our overrides."""
    def encode(value):
        if isinstance(value, type):
            return f"{value.__module__}.{value.__qualname__}"
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        return value

    return {name: encode(p.default) for name, p in inspect.signature(cls).parameters.items()
            if p.default is not inspect.Parameter.empty}


@dataclass(frozen=True)
class RLConfig:
    algo: str = "ppo"
    split: str = "seen-10"
    seed: int = 0
    steps: int = 1_000_000
    n_envs: int | None = None
    action_repeat: int = 1
    throttle_axis: bool = False
    device: str = "auto"
    torch_threads: int = 1
    hidden_sizes: tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 256
    use_sde: bool = False
    sde_sample_freq: int = 4
    clip_obs: float = 10.0
    eval_freq: int = 50_000
    eval_episodes: int = 10
    n_steps: int = 1024
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    train_freq: int = 1
    gradient_steps: int = 1
    tau: float = 0.005
    sac_ent_coef: str | float = "auto"


def _training_env(split: str, seed: int, rank: int, workers: int, directory: str,
                  action_repeat: int = 1, throttle_axis: bool = False) -> gym.Env:
    torch.set_num_threads(1)
    env = Monitor(TrainingSeeds(split, seed, rank, workers), str(Path(directory) / f"worker_{rank}"))
    return agent_interface(env, action_repeat, throttle_axis)


def _eval_env(split: str, action_repeat: int = 1, throttle_axis: bool = False) -> gym.Env:
    require_seen(split)
    return agent_interface(Monitor(make_env(split)), action_repeat, throttle_axis)


class SeenEvalCallback(EvalCallback):
    """SB3 synchronizes obs stats; save matching stats whenever best model changes."""
    evaluation_steps: int = 0
    training_steps: int = 0

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        self.evaluation_steps += locals_["info"].get("action_repeat_frames", 1)
        if locals_["done"]:
            metrics = locals_["info"].get("episode_metrics", {})
            self._is_success_buffer.append(bool(metrics.get("success", False)))

    def _on_step(self) -> bool:
        self.training_steps += sum(info.get("action_repeat_frames", 1) for info in self.locals["infos"])
        best = self.best_mean_reward
        result = super()._on_step()
        if self.best_mean_reward > best:
            self.eval_env.save(Path(self.best_model_save_path) / "vecnormalize.pkl")
        if self.n_calls % 100 == 0:
            self.logger.record("cost/env_steps", self.training_steps)
            self.logger.record("cost/decision_steps", self.num_timesteps)
            self.logger.record("cost/evaluation_env_steps", self.evaluation_steps)
        return result


def train_rl(config: RLConfig, run_id: str, runs_dir: str | Path = "runs") -> Path:
    require_seen(config.split)
    if config.seed < 0:
        raise ValueError("seed must be nonnegative")
    if config.algo not in ("ppo", "sac"):
        raise ValueError("algo must be ppo or sac")
    if isinstance(config.action_repeat, bool) or not isinstance(config.action_repeat, int) or config.action_repeat < 1:
        raise ValueError("action_repeat must be a positive integer")
    workers = config.n_envs if config.n_envs is not None else (12 if config.algo == "ppo" else 1)
    device = config.device if config.device != "auto" else (
        "cuda" if config.algo == "sac" and torch.cuda.is_available() else "cpu")
    config = replace(config, n_envs=workers, device=device)
    for name in ("steps", "n_envs", "torch_threads", "batch_size", "eval_freq", "eval_episodes",
                 "n_steps", "n_epochs", "buffer_size", "train_freq", "gradient_steps"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if config.algo == "ppo" and (config.n_steps * workers < 2 or config.batch_size < 2):
        raise ValueError("PPO needs at least two rollout samples and batch_size >= 2")
    if config.learning_rate <= 0 or config.clip_obs <= 0 or config.learning_starts < 0:
        raise ValueError("Invalid learning/normalization parameters")
    seed_everything(config.seed, config.torch_threads)
    algorithm = PPO if config.algo == "ppo" else SAC
    root = create_run(f"rl_{config.algo}", run_id, config, runs_dir, implementation={
        "algorithm_defaults": _constructor_defaults(algorithm),
        "policy_defaults": _constructor_defaults(algorithm.policy_aliases["MlpPolicy"]),
        "vecnormalize": {"norm_obs": True, "norm_reward": False, "clip_obs": config.clip_obs,
                         "epsilon": 1e-8}, "start_method": "spawn",
        "ppo_device_default": "cpu", "sac_device_default": "cuda if available, else cpu",
        "initial_worker_sampler_seed": "SB3 reset seed = training seed + worker rank",
        "step_unit": "agent decisions; cost/env_steps counts actual environment frames",
        "wrapper_order": "per-frame env -> Monitor -> optional ThrottleAxis -> ActionRepeat -> VecNormalize",
    })
    started = time.monotonic()
    (root / "monitor").mkdir()
    training = None
    evaluation = None
    try:
        training = VecNormalize(SubprocVecEnv([
            partial(_training_env, config.split, config.seed, rank, workers, str(root / "monitor"),
                    config.action_repeat, config.throttle_axis)
            for rank in range(workers)
        ], start_method="spawn"), norm_obs=True, norm_reward=False, clip_obs=config.clip_obs)
        evaluation = VecNormalize(DummyVecEnv([partial(_eval_env, config.split, config.action_repeat,
                                                       config.throttle_axis)]),
                                  training=False, norm_obs=True, norm_reward=False, clip_obs=config.clip_obs)
        common = dict(policy="MlpPolicy", env=training, seed=config.seed, device=config.device,
                      learning_rate=config.learning_rate, gamma=config.gamma, batch_size=config.batch_size,
                      use_sde=config.use_sde, sde_sample_freq=config.sde_sample_freq,
                      policy_kwargs={"net_arch": list(config.hidden_sizes)},
                      tensorboard_log=str(root / "tensorboard"), verbose=0)
        if config.algo == "ppo":
            model = PPO(**common, n_steps=config.n_steps, n_epochs=config.n_epochs,
                        gae_lambda=config.gae_lambda, clip_range=config.clip_range,
                        ent_coef=config.ent_coef, vf_coef=config.vf_coef, max_grad_norm=config.max_grad_norm)
        else:
            model = SAC(**common, buffer_size=config.buffer_size, learning_starts=config.learning_starts,
                        train_freq=config.train_freq, gradient_steps=config.gradient_steps,
                        tau=config.tau, ent_coef=config.sac_ent_coef)
        callback = SeenEvalCallback(evaluation, best_model_save_path=str(root / "best"),
                                   log_path=str(root / "training_eval"),
                                   eval_freq=max(1, (config.eval_freq + workers - 1) // workers),
                                   n_eval_episodes=config.eval_episodes, deterministic=True, verbose=0)
        model.learn(total_timesteps=config.steps, callback=callback, tb_log_name="train")
        model.save(root / "model.zip")
        training.save(root / "vecnormalize.pkl")
        model.logger.record("cost/env_steps", callback.training_steps)
        model.logger.record("cost/decision_steps", model.num_timesteps)
        model.logger.record("cost/evaluation_env_steps", callback.evaluation_steps)
        model.logger.dump(model.num_timesteps)
        write_json(root / "cost.json", {"env_steps": callback.training_steps,
                                        "decision_steps": model.num_timesteps,
                                        "requested_decision_steps": config.steps,
                                        "requested_env_steps": config.steps * config.action_repeat,
                                        "evaluation_env_steps": callback.evaluation_steps,
                                        "wall_time_s": time.monotonic() - started})
        # SB3's own serialized constructor settings complement our public hyperparameters.
        write_json(root / "resolved_model.json", {"device": str(model.device),
                                                  "policy_kwargs": model.policy_kwargs,
                                                  "norm_reward": False, "training_split": config.split})
    finally:
        if training is not None:
            training.close()
        if evaluation is not None:
            evaluation.close()
    return root
