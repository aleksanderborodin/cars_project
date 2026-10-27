"""Expert labels and supervised fitting shared by BC and DAgger."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from cars.env.state import Array, CarState
from cars.policies.baseline import BaselinePolicy
from cars.policies.torch_mlp import MLPPolicy
from cars.training import TrainingSeeds


@dataclass
class Demonstrations:
    observations: list[Array] = field(default_factory=list)
    actions: list[Array] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    rounds: list[int] = field(default_factory=list)
    episodes: list[dict] = field(default_factory=list)

    def save(self, root: Path) -> None:
        np.savez_compressed(root / "dataset.npz", observations=np.stack(self.observations),
                            actions=np.stack(self.actions), seeds=self.seeds, rounds=self.rounds)


def collect(env: TrainingSeeds, data: Demonstrations, *, learner: MLPPolicy | None,
            beta: float, episodes: int | None, steps: int | None, round_index: int,
            rng: np.random.Generator) -> None:
    """One label per visited pre-action state, including expert-driven states."""
    if episodes is None and steps is None:
        raise ValueError("Collection needs a budget")
    expert = BaselinePolicy()
    collected = 0
    episode_index = 0
    while (episodes is None or episode_index < episodes) and (steps is None or collected < steps):
        obs, reset_info = env.reset()
        expert.reset(env.unwrapped.track_info)
        if learner is not None:
            learner.reset(env.unwrapped.track_info)
        total_return = 0.0
        expert_actions = 0
        episode_steps = 0
        while True:
            state = CarState.from_env(env)
            label = expert.act(state, obs)
            data.observations.append(obs.copy())
            data.actions.append(label.copy())
            data.seeds.append(reset_info["seed"])
            data.rounds.append(round_index)
            use_expert = learner is None or rng.random() < beta
            action = label if use_expert else learner.act(state, obs)
            expert_actions += int(use_expert)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            collected += 1
            episode_steps += 1
            if terminated or truncated or (steps is not None and collected >= steps):
                data.episodes.append({"round": round_index, "seed": reset_info["seed"],
                                      "beta": beta, "steps": episode_steps, "return": total_return,
                                      "expert_actions": expert_actions,
                                      "complete": bool(terminated or truncated),
                                      "metrics": info.get("episode_metrics")})
                break
        episode_index += 1


def fit(policy: MLPPolicy, data: Demonstrations, *, epochs: int, batch_size: int,
        learning_rate: float, seed: int) -> list[float]:
    x = torch.from_numpy(np.stack(data.observations))
    y = torch.from_numpy(np.stack(data.actions))
    with torch.no_grad():
        policy.network.obs_mean.copy_(x.mean(0))
        policy.network.obs_std.copy_(x.std(0, unbiased=False).clamp_min(policy.config.normalization_floor))
    optimizer = torch.optim.Adam(policy.network.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    policy.network.train()
    for _ in range(epochs):
        total = 0.0
        order = torch.randperm(len(x), generator=generator)
        for indices in order.split(batch_size):
            loss = torch.nn.functional.mse_loss(policy.network(x[indices]), y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        losses.append(total / len(x))
    policy.network.eval()
    return losses
