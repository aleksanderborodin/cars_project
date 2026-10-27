"""Seen-only training infrastructure over the unmodified per-frame environment."""
from dataclasses import asdict, is_dataclass
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from cars.env.make import make_env
from cars.env.state import OBS_SCHEMA, OBS_VERSION

SEEN_SPLITS = ("seen-1", "seen-10", "seen-100", "seen-inf")


def require_seen(split: str) -> None:
    if split not in SEEN_SPLITS:
        raise ValueError(f"Training/evaluation during training requires a seen split: {split}")


def seed_everything(seed: int, threads: int = 1) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


class SeedSampler:
    """Uniform finite sampling; unique fresh seeds in disjoint worker residues."""
    def __init__(self, split: str, seed: int, rank: int = 0, workers: int = 1):
        require_seen(split)
        if not 0 <= rank < workers:
            raise ValueError("Invalid sampler worker")
        self.split, self.rank, self.workers = split, rank, workers
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, rank]))
        self.used: set[int] = set()

    def sample(self) -> int:
        if self.split != "seen-inf":
            size = int(self.split.removeprefix("seen-"))
            return int(self.rng.integers(size))
        limit = (2**31 - 1 - 1_000_000 - self.rank) // self.workers + 1
        if len(self.used) == limit:
            raise RuntimeError("Fresh seed space exhausted")
        while True:
            value = 1_000_000 + self.rank + self.workers * int(self.rng.integers(limit))
            if value not in self.used:
                self.used.add(value)
                return value


class TrainingSeeds(gym.Wrapper):
    def __init__(self, split: str, seed: int, rank: int = 0, workers: int = 1):
        self.sampler = SeedSampler(split, seed, rank, workers)
        super().__init__(make_env(0))

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        # SB3's initial env.seed() seeds this stream, never an out-of-split track.
        if seed is not None:
            self.sampler = SeedSampler(self.sampler.split, seed, self.sampler.rank, self.sampler.workers)
        return self.env.reset(seed=self.sampler.sample(), options=options)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def create_run(method: str, run_id: str, config: Any, runs_dir: str | Path,
               *, implementation: dict | None = None) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a single directory name")
    root = Path(runs_dir).resolve() / method / run_id
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    settings = asdict(config) if is_dataclass(config) else config
    write_json(root / "config.json", {
        "kind": "training", "method": method, "run_id": run_id,
        "seed": settings["seed"], "training_seed": settings["seed"], "hyperparameters": settings,
        "implementation": implementation or {},
        "environment": {"id": "CarRacing-v3", "max_episode_steps": 1000,
                        "lap_complete_percent": 0.95, "reward": "official, unmodified"},
        "seed_sampler": "numpy SeedSequence([seed, rank]); uniform seen-K; unique seen-inf worker residues",
        "obs_version": OBS_VERSION, "obs_schema": OBS_SCHEMA,
        "versions": {p: version(p) for p in ("cars", "gymnasium", "numpy", "torch", "stable-baselines3", "Box2D")},
        "source_sha256": {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob("*.py"))},
        "uv_lock_sha256": sha256(Path(__file__).parents[2] / "uv.lock")
        if (Path(__file__).parents[2] / "uv.lock").exists() else None,
    })
    return root
