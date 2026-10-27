"""Versioned, portable gzip JSON logs. Step rows describe post-action state."""
import gzip
import json
from pathlib import Path
from importlib.metadata import version
from typing import Any

import gymnasium as gym
from gymnasium.envs.box2d.car_racing import FPS

from cars.env.state import Array, CarState, OBS_SCHEMA, OBS_VERSION, STATE_FIELDS, STATE_VERSION

LOG_VERSION = 1
STEP_FIELDS = ("t", "x", "y", "heading", "speed", "slip_angle", "steer", "gas", "brake",
               "reward", "wheels_on_road", "tile_idx")


def write_episode(path: str | Path, episode: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Fixed gzip timestamp and no source filename make identical data byte-identical.
    with path.open("wb") as file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as compressed:
            compressed.write(json.dumps(episode, allow_nan=False, separators=(",", ":")).encode())


def read_episode(path: str | Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        episode = json.load(file)
    if episode.get("version") != LOG_VERSION:
        raise ValueError("Unsupported episode log version")
    lengths = {len(episode["steps"][field]) for field in STEP_FIELDS}
    if len(lengths) != 1:
        raise ValueError("Inconsistent per-step array lengths")
    return episode


class EpisodeLogger(gym.Wrapper):
    def __init__(self, env: gym.Env, out_dir: str | Path, *, method: str = "unknown",
                 run_id: str | None = None, split: str = "seed", episode_index: int = 0,
                 write_config: bool = True):
        super().__init__(env)
        self.out_dir = Path(out_dir)
        self.method = method
        self.run_id = run_id or self.out_dir.name
        self.split = split
        self.episode_index = episode_index
        self.write_config = write_config

    def reset(self, **kwargs: Any) -> tuple[Array, dict]:
        obs, info = self.env.reset(**kwargs)
        path = self.out_dir / f"episode_{self.episode_index:05d}_seed_{info['seed']}.json.gz"
        if path.exists():
            raise FileExistsError(f"Episode log already exists: {path}")
        if self.write_config:
            # Direct factory users receive an environment manifest too. The
            # evaluation runner writes the richer policy manifest at run root.
            self.out_dir.mkdir(parents=True, exist_ok=True)
            config_path = self.out_dir / "config.json"
            config = json.loads(config_path.read_text()) if config_path.exists() else {
                "method": self.method, "run_id": self.run_id, "split": self.split,
                "hard": self.unwrapped.hard, "log_version": LOG_VERSION,
                "gymnasium_version": version("gymnasium"),
                "lap_complete_percent": 0.95, "max_episode_steps": 1000,
                "state_version": STATE_VERSION, "state_fields": STATE_FIELDS,
                "obs_version": OBS_VERSION, "obs_schema": OBS_SCHEMA, "episodes": []}
            config["episodes"].append({"episode_index": self.episode_index, "seed": info["seed"]})
            config_path.write_text(json.dumps(config, indent=2) + "\n")
        track = self.unwrapped.track_info
        self.episode = {
            "version": LOG_VERSION,
            "meta": {"method": self.method, "run_id": self.run_id, "split": self.split,
                     "seed": info["seed"], "hard": self.unwrapped.hard,
                     "episode_index": self.episode_index, "fps": FPS,
                     "state_version": STATE_VERSION, "state_fields": STATE_FIELDS,
                     "obs_version": OBS_VERSION, "obs_schema": OBS_SCHEMA},
            "track": {"centre": track.centre.tolist(), "left": track.left.tolist(),
                      "right": track.right.tolist(), "half_width": track.half_width.tolist(),
                      "tangent": track.tangent.tolist()},
            "initial_state": CarState.from_env(self).as_array().tolist(),
            "steps": {field: [] for field in STEP_FIELDS},
            "events": {"off_track_steps": [], "off_track_events": [], "crash": None, "finish_step": None},
        }
        self.was_on_road = CarState.from_env(self).wheels_on_road > 0
        return obs, info

    def step(self, action: Array) -> tuple[Array, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = CarState.from_env(self)
        rows = self.episode["steps"]
        step = len(rows["t"]) + 1
        values = (step / FPS, state.x, state.y, state.heading, state.speed, state.slip_angle,
                  *map(float, action), float(reward), state.wheels_on_road, state.tile_idx)
        for field, value in zip(STEP_FIELDS, values, strict=True):
            rows[field].append(value)
        if state.wheels_on_road == 0:
            self.episode["events"]["off_track_steps"].append(step)
            if self.was_on_road:
                self.episode["events"]["off_track_events"].append(step)
        self.was_on_road = state.wheels_on_road > 0
        if terminated or truncated:
            summary = info["episode_metrics"]
            self.episode["summary"] = summary
            if summary["crash"]:
                self.episode["events"]["crash"] = step
            if info.get("lap_finished"):
                self.episode["events"]["finish_step"] = step
            path = self.out_dir / f"episode_{self.episode_index:05d}_seed_{self.episode['meta']['seed']}.json.gz"
            write_episode(path, self.episode)
            info["episode_log"] = str(path)
            self.episode_index += 1
        return obs, reward, terminated, truncated, info
