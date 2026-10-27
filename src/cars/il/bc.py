"""Behaviour cloning from the privileged pure-pursuit expert on seen tracks."""
from dataclasses import dataclass, replace
from pathlib import Path
import time

import numpy as np

from cars.il.data import Demonstrations, collect, fit
from cars.policies.baseline import BaselinePolicy
from cars.policies.torch_mlp import MLPConfig, MLPPolicy
from cars.training import TrainingSeeds, create_run, require_seen, seed_everything, write_json


@dataclass(frozen=True)
class BCConfig:
    split: str = "seen-10"
    seed: int = 0
    episodes: int | None = None
    steps: int | None = None
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 0.001
    hidden_sizes: tuple[int, ...] = (256, 256)
    normalization_floor: float = 0.01
    clip_obs: float = 10.0
    torch_threads: int = 1


def validate(config: BCConfig) -> None:
    require_seen(config.split)
    if config.seed < 0:
        raise ValueError("seed must be nonnegative")
    for name in ("episodes", "steps", "epochs", "batch_size", "torch_threads"):
        value = getattr(config, name)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")
    if config.learning_rate <= 0 or config.normalization_floor <= 0 or config.clip_obs <= 0:
        raise ValueError("Learning rate and normalization scales must be positive")
    if not config.hidden_sizes or any(s < 1 for s in config.hidden_sizes):
        raise ValueError("hidden_sizes must be positive")


def train_bc(config: BCConfig, run_id: str, runs_dir: str | Path = "runs") -> Path:
    validate(config)
    if config.episodes is None and config.steps is None:
        config = replace(config, episodes=100)
    seed_everything(config.seed, config.torch_threads)
    root = create_run("il_bc", run_id, config, runs_dir, implementation={
        "expert": BaselinePolicy().configuration(), "loss": "action MSE",
        "optimizer": {"name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0},
        "normalization": "dataset mean and population std, floored, then clipped",
    })
    started = time.monotonic()
    data = Demonstrations()
    env = TrainingSeeds(config.split, config.seed)
    try:
        collect(env, data, learner=None, beta=1.0,
                episodes=config.episodes,
                steps=config.steps, round_index=0, rng=np.random.default_rng(config.seed))
    finally:
        env.close()
    policy = MLPPolicy(MLPConfig(config.hidden_sizes, config.normalization_floor, config.clip_obs),
                       training_seed=config.seed)
    losses = fit(policy, data, epochs=config.epochs, batch_size=config.batch_size,
                 learning_rate=config.learning_rate, seed=config.seed)
    policy.save(root / "model.pt")
    data.save(root)
    write_json(root / "history.json", {"loss": losses, "episodes": data.episodes,
                                       "expert": BaselinePolicy().configuration()})
    write_json(root / "cost.json", {"expert_labels": len(data.actions), "env_steps": len(data.actions),
                                    "episodes": len(data.episodes), "wall_time_s": time.monotonic() - started})
    return root
