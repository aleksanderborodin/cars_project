"""DAgger: bootstrap, query the expert at visited states, refit on all labels."""
from dataclasses import dataclass, replace
from pathlib import Path
import time

import numpy as np

from cars.il.bc import BCConfig, validate
from cars.il.data import Demonstrations, collect, fit
from cars.policies.baseline import BaselinePolicy
from cars.policies.torch_mlp import MLPConfig, MLPPolicy
from cars.training import TrainingSeeds, create_run, seed_everything, write_json


@dataclass(frozen=True)
class DAggerConfig(BCConfig):
    iterations: int = 10
    bootstrap_episodes: int = 10
    beta_start: float = 0.5
    beta_end: float = 0.0


def train_dagger(config: DAggerConfig, run_id: str, runs_dir: str | Path = "runs") -> Path:
    validate(config)
    episodes = config.episodes if config.episodes is not None else (100 if config.steps is None else None)
    config = replace(config, episodes=episodes)
    if config.iterations < 1 or config.bootstrap_episodes < 1:
        raise ValueError("iterations and bootstrap_episodes must be positive")
    if not 0 <= config.beta_end <= config.beta_start <= 1:
        raise ValueError("Require 0 <= beta_end <= beta_start <= 1")
    if episodes is not None and episodes < config.bootstrap_episodes + config.iterations:
        raise ValueError("Episode budget must cover bootstrap and every DAgger iteration")
    stages = config.iterations + 1
    if config.steps is not None and config.steps < stages:
        raise ValueError("Step budget must cover every stage")
    seed_everything(config.seed, config.torch_threads)
    root = create_run("il_dagger", run_id, config, runs_dir, implementation={
        "expert": BaselinePolicy().configuration(), "loss": "action MSE",
        "optimizer": {"name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0},
        "beta_schedule": "linear beta_start to beta_end; one iteration uses beta_end",
        "refit": "from scratch on all labels after each round, seed + round",
        "normalization": "aggregate mean and population std, floored, then clipped",
    })
    started = time.monotonic()
    env = TrainingSeeds(config.split, config.seed)
    data = Demonstrations()
    learner = None
    history = []
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1000]))
    try:
        for stage in range(stages):
            betas = np.linspace(config.beta_start, config.beta_end, config.iterations) if config.iterations > 1 else [config.beta_end]
            beta = 1.0 if stage == 0 else float(betas[stage - 1])
            if episodes is None:
                stage_episodes = config.bootstrap_episodes if stage == 0 else None
            elif stage == 0:
                stage_episodes = config.bootstrap_episodes
            else:
                remaining = episodes - config.bootstrap_episodes
                stage_episodes = remaining // config.iterations + int(stage <= remaining % config.iterations)
            stage_steps = None if config.steps is None else (config.steps - len(data.actions)) // (stages - stage)
            collect(env, data, learner=learner, beta=beta, episodes=stage_episodes,
                    steps=stage_steps, round_index=stage, rng=rng)
            # Refit from scratch: re-estimated normalization cannot invalidate old weights.
            seed_everything(config.seed + stage, config.torch_threads)
            learner = MLPPolicy(MLPConfig(config.hidden_sizes, config.normalization_floor, config.clip_obs),
                                method="il_dagger", training_seed=config.seed)
            losses = fit(learner, data, epochs=config.epochs, batch_size=config.batch_size,
                         learning_rate=config.learning_rate, seed=config.seed + stage)
            history.append({"round": stage, "beta": beta, "expert_labels": len(data.actions), "loss": losses})
    finally:
        env.close()
    assert learner is not None
    learner.save(root / "model.pt")
    data.save(root)
    write_json(root / "history.json", {"rounds": history, "episodes": data.episodes,
                                       "expert": BaselinePolicy().configuration(), "refit": "from scratch each round"})
    write_json(root / "cost.json", {"expert_labels": len(data.actions), "env_steps": len(data.actions),
                                    "episodes": len(data.episodes), "wall_time_s": time.monotonic() - started})
    return root
