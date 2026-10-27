"""Episode-isolated evaluation. Explicit seeds make worker count immaterial."""
from concurrent.futures import ProcessPoolExecutor
import copy
from importlib.metadata import version
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from cars.env.make import make_env
from cars.env.splits import SPLITS, episode_seeds
from cars.env.state import CarState, OBS_SCHEMA, OBS_VERSION, STATE_FIELDS, STATE_VERSION
from cars.policies.base import Policy
from cars.policies.rust import RustPolicy


def _episode(job: tuple[Policy, int, int, str, dict[str, Any]]) -> dict:
    policy, index, seed, directory, meta = job
    env = make_env(seed, hard=meta["hard"], log={"out_dir": directory, "method": meta["method"],
                   "run_id": meta["run_id"], "split": meta["split"], "episode_index": index, "write_config": False})
    try:
        obs, _ = env.reset()
        policy.reset(env.unwrapped.track_info)
        while True:
            action = np.asarray(policy.act(CarState.from_env(env), obs), dtype=np.float32)
            if not env.action_space.contains(action):
                raise ValueError(f"Invalid policy action: {action}")
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                return {**meta, "episode": index, "seed": seed, **info["episode_metrics"],
                        "log": str(Path(info["episode_log"]).relative_to(Path(directory).parents[1]))}
    finally:
        env.close()
        if hasattr(policy, "close"):
            policy.close()


def evaluate(policy: Policy, split: str, n_episodes: int, out_dir: str | Path,
             workers: int = 1, *, evaluation_seed: int = 0) -> pd.DataFrame:
    """out_dir is runs/<method>/<run_id>; results go in eval/<split>.

    Policy must be deepcopyable and, for multiprocessing, pickleable. Rust
    libraries are loaded inside spawn workers. A completed run is never replaced.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    seeds = episode_seeds(split, n_episodes, seed=evaluation_seed)
    root = Path(out_dir).resolve()
    directory = root / "eval" / split
    method = getattr(policy, "method", "rust" if isinstance(policy, RustPolicy) else ("base" if type(policy).__name__ == "BaselinePolicy" else type(policy).__name__))
    meta = {"method": method, "run_id": root.name, "split": split, "hard": SPLITS[split].hard}
    configuration = policy.configuration() if hasattr(policy, "configuration") else {"class": f"{type(policy).__module__}.{type(policy).__qualname__}", "repr": repr(policy)}
    if getattr(policy, "training_seed", None) is not None:
        meta["training_seed"] = policy.training_seed
    config_path = root / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if (existing.get("kind") != "training" or existing.get("method") != method
                or configuration.get("training_run") != str(root)):
            raise FileExistsError(f"Run already exists: {root}")
        config_path = directory / "config.json"
    if config_path.exists() or (directory / "metrics.csv").exists():
        raise FileExistsError(f"Evaluation already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(policy, RustPolicy):
        candidate = root / "candidate.so"
        shutil.copy2(policy.path, candidate)
        policy = RustPolicy(candidate)
        configuration["artifact"] = "candidate.so"
    source_root = Path(__file__).parents[1]
    sources = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(source_root.rglob("*.py"))}
    config = {**meta, "n_episodes": n_episodes, "seeds": seeds, "evaluation_seed": evaluation_seed,
              "workers": workers, "policy": configuration, "environment": {"id": "CarRacing-v3",
              "lap_complete_percent": 0.95, "max_episode_steps": 1000, "fps": 50},
              "state_version": STATE_VERSION, "state_fields": STATE_FIELDS,
              "obs_version": OBS_VERSION, "obs_schema": OBS_SCHEMA, "source_sha256": sources,
              "versions": {package: version(package) for package in ("cars", "gymnasium", "numpy", "Box2D", "pandas")}}
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    jobs = [(copy.deepcopy(policy), i, seed, str(directory), meta) for i, seed in enumerate(seeds)]
    if workers == 1:
        rows = [_episode(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            rows = list(pool.map(_episode, jobs))
    frame = pd.DataFrame(rows).sort_values("episode").reset_index(drop=True)
    frame.to_csv(directory / "metrics.csv", index=False)
    return frame
