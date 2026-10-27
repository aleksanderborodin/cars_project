"""Lazy, pickleable deterministic SB3 adapter for the privileged Policy protocol."""
import json
from pathlib import Path
import pickle

import numpy as np

from cars.env.state import Array, CarState, TrackInfo
from cars.policies.base import Policy
from cars.rl.interface import throttle_action


class SB3Policy:
    def __init__(self, path: str | Path, *, method: str, stats_path: str | Path | None = None):
        if method not in ("rl_ppo", "rl_sac"):
            raise ValueError(method)
        self.path = Path(path).resolve()
        if self.path.is_dir():
            self.path = self.path / "model.zip"
        self.stats_path = Path(stats_path).resolve() if stats_path else self.path.parent / "vecnormalize.pkl"
        if not self.path.is_file() or not self.stats_path.is_file():
            raise FileNotFoundError("Both model.zip and matching vecnormalize.pkl are required")
        self.method = method
        config_path = self.path.parent / "config.json"
        if not config_path.exists() and self.path.parent.name == "best":
            config_path = self.path.parent.parent / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        if config.get("method", method) != method:
            raise ValueError("Model method does not match requested method")
        self.training_seed = config.get("training_seed")
        settings = config.get("hyperparameters", {})
        self.action_repeat = settings.get("action_repeat", 1)
        self.throttle_axis = settings.get("throttle_axis", False)
        if isinstance(self.action_repeat, bool) or not isinstance(self.action_repeat, int) or self.action_repeat < 1:
            raise ValueError("action_repeat must be a positive integer")
        self.training_run = str(config_path.parent)
        self._model = None
        self._normalizer = None
        self._remaining = 0
        self._action: Array | None = None

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_model": None, "_normalizer": None}

    def _load(self) -> None:
        from stable_baselines3 import PPO, SAC
        import torch

        torch.set_num_threads(1)
        cls = PPO if self.method == "rl_ppo" else SAC
        self._model = cls.load(self.path, device="cpu")
        # VecNormalize pickles exclude the training env. normalize_obs needs no env.
        with self.stats_path.open("rb") as stream:
            self._normalizer = pickle.load(stream)
        self._normalizer.training = False
        self._normalizer.norm_reward = False

    def reset(self, track: TrackInfo) -> None:
        self._remaining = 0
        self._action = None
        if self._model is None:
            self._load()

    def act(self, state: CarState, obs: Array) -> Array:
        if self._model is None:
            self._load()
        if self._remaining == 0:
            normalized = self._normalizer.normalize_obs(obs)
            action, _ = self._model.predict(normalized, deterministic=True)
            self._action = throttle_action(action) if self.throttle_axis else np.asarray(action, dtype=np.float32)
            self._remaining = self.action_repeat
        self._remaining -= 1
        return self._action.copy()

    def configuration(self) -> dict:
        from cars.training import sha256

        return {"method": self.method, "training_seed": self.training_seed,
                "training_run": self.training_run, "artifact": str(self.path),
                "sha256": sha256(self.path), "normalization": str(self.stats_path),
                "normalization_sha256": sha256(self.stats_path), "deterministic": True,
                "action_repeat": self.action_repeat, "throttle_axis": self.throttle_axis}


def load_learned_policy(method: str, path: str | Path) -> Policy:
    if method in ("rl_ppo", "rl_sac"):
        return SB3Policy(path, method=method)
    from cars.policies.torch_mlp import MLPPolicy

    policy = MLPPolicy.load(path)
    if policy.method != method:
        raise ValueError("Model method does not match requested method")
    return policy
