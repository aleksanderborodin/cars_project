"""Factory: raw state -> official time limit -> metrics -> optional log -> StateObs."""
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym

from cars.env.carracing import StateCarRacing
from cars.env.metrics import EpisodeMetrics
from cars.env.splits import SPLITS, episode_seeds
from cars.env.state import Array, StateObs


class SeededReset(gym.Wrapper):
    def __init__(self, env: gym.Env, split_or_seed: str | int):
        super().__init__(env)
        self.source = split_or_seed
        self.episode_index = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Array, dict]:
        if seed is None:
            if isinstance(self.source, int):
                seed = self.source
            else:
                seed = episode_seeds(self.source, self.episode_index + 1)[-1]
        self.episode_index += 1
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, {**info, "seed": seed}


def make_env(split_or_seed: str | int = "seen-10", *, obs: Literal["state", "raw"] = "state",
             hard: bool | None = None, log: str | Path | dict[str, Any] | None = None) -> gym.Env:
    if obs not in ("state", "raw"):
        raise ValueError(f"Unknown observation view: {obs}")
    if isinstance(split_or_seed, str):
        split_spec = SPLITS[split_or_seed]
        if split_spec.hard and hard is False:
            raise ValueError("The hard split requires the hard generator")
        hard = split_spec.hard if hard is None else hard
    hard = bool(hard)
    env = EpisodeMetrics(gym.wrappers.TimeLimit(StateCarRacing(hard=hard), max_episode_steps=1000))
    env = SeededReset(env, split_or_seed)
    if log is not None:
        from cars.eval.episode_log import EpisodeLogger
        settings = dict(log) if isinstance(log, dict) else {"out_dir": str(log)}
        settings.setdefault("split", split_or_seed if isinstance(split_or_seed, str) else "seed")
        env = EpisodeLogger(env, **settings)
    if obs == "state":
        env = StateObs(env)
    return env
