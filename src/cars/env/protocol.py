"""Information-restricted observations, separated from evaluator truth."""
from dataclasses import asdict, dataclass
from typing import Literal
import time

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.car_racing import CarRacing

from cars.env.make import make_env
from cars.env.metrics import EpisodeMetrics
from cars.env.partitions import Partitions, TrainingStream, digest
from cars.env.state import CarState, TrackInfo

Observation = dict[str, np.ndarray]


@dataclass(frozen=True)
class ProtocolConfig:
    view: Literal["local", "native"] = "local"
    radius: float = 30.0
    resolution: float = 1.0
    action_repeat: int = 4
    version: int = 1

    def __post_init__(self) -> None:
        if self.view not in ("local", "native") or self.radius <= 0 or self.resolution <= 0:
            raise ValueError("Invalid observation protocol")
        if isinstance(self.action_repeat, bool) or not isinstance(self.action_repeat, int) or self.action_repeat < 1 or self.version != 1:
            raise ValueError("Invalid action repeat or protocol version")
        if not np.isclose(2 * self.radius / self.resolution, round(2 * self.radius / self.resolution)):
            raise ValueError("Grid extent must be an integer number of cells")

    @property
    def size(self) -> int:
        return int(round(2 * self.radius / self.resolution)) + 1

    @property
    def identifier(self) -> str:
        return f"{self.view}-v{self.version}-{digest(asdict(self))[:12]}"


class NativeCarRacing(CarRacing):
    """Official pixels; track metadata is for evaluator metrics only."""

    def _create_track(self) -> bool:
        success = super()._create_track()
        if success:
            self.track_info = TrackInfo.from_env(self)
        return success

    def _render(self, mode: str) -> np.ndarray:
        started = time.perf_counter()
        image = super()._render(mode)
        self.sensor_seconds = time.perf_counter() - started
        return image


class LocalSensor:
    """Exact road occupancy sampled at visible car-frame grid points.

    Axes: row = forward, column = left, increasing from -radius to +radius.
    No global index, connected-component ordering, or hidden geometry is emitted.
    Boundary cells describe their sample location, not the whole cell footprint.
    """

    def __init__(self, config: ProtocolConfig):
        axis = np.linspace(-config.radius, config.radius, config.size)
        x, y = np.meshgrid(axis, axis, indexing="ij")
        self.points = np.stack((x, y), axis=-1)
        self.visible = np.sum(self.points**2, axis=-1) <= config.radius**2
        self.config = config

    def road(self, track: TrackInfo, state: CarState) -> np.ndarray:
        c, s = np.cos(state.heading), np.sin(state.heading)
        rotation = np.array([[c, -s], [s, c]])
        left = (track.left - [state.x, state.y]) @ rotation
        right = (track.right - [state.x, state.y]) @ rotation
        quads = np.stack((left, right, np.roll(right, 1, axis=0), np.roll(left, 1, axis=0)), axis=1)
        radius = self.config.radius
        intersects = np.all(quads.max(axis=1) >= -radius, axis=1) & np.all(quads.min(axis=1) <= radius, axis=1)
        road = np.zeros(self.visible.shape, dtype=bool)
        # Rasterize bounded patches to avoid full-grid x full-track work.
        for quad in quads[intersects]:
            low = np.maximum(0, np.ceil((quad.min(axis=0) + radius) / self.config.resolution).astype(int))
            high = np.minimum(self.config.size - 1, np.floor((quad.max(axis=0) + radius) / self.config.resolution).astype(int))
            if np.any(high < low):
                continue
            a, b = slice(low[0], high[0] + 1), slice(low[1], high[1] + 1)
            relative = self.points[a, b, None, :] - quad
            edges = np.roll(quad, -1, axis=0) - quad
            cross = edges[:, 0] * relative[..., 1] - edges[:, 1] * relative[..., 0]
            road[a, b] |= np.all(cross >= -1e-7, axis=-1) | np.all(cross <= 1e-7, axis=-1)
        return (road & self.visible).astype(np.float32)

    def observe(self, track: TrackInfo, state: CarState, previous: np.ndarray) -> Observation:
        return {"road": self.road(track, state), "visible": self.visible.astype(np.float32),
                "ego": np.asarray([*state.body_velocity, state.yaw_rate], dtype=np.float32),
                "previous_action": previous.copy()}


class DrivingEnv(gym.Env):
    """Restricted agent observations; seed selection belongs to infrastructure.

    The internal simulator and evaluator methods are never passed to a policy.
    This is an interface boundary, not a sandbox against hostile Python code.
    """
    metadata = {"render_modes": []}

    def __init__(self, config: ProtocolConfig, partition: str = "train", *, seed: int = 0,
                 worker: int = 0, workers: int = 1, partitions: Partitions | None = None):
        super().__init__()
        if partition not in ("train", "development", "calibration"):
            raise ValueError("Final test and legacy held-out environments are sealed")
        self.config, self.partition = config, partition
        self.partitions = partitions or Partitions()
        self.stream = TrainingStream(seed, worker, workers)
        self.index = 0
        self.track_seeds: list[int] = []
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.sensor = LocalSensor(config)
        if config.view == "native":
            self._sim = EpisodeMetrics(gym.wrappers.TimeLimit(
                NativeCarRacing(lap_complete_percent=0.95, continuous=True), 1000))
            fields = {"image": gym.spaces.Box(0, 255, (96, 96, 3), np.uint8)}
        else:
            self._sim = make_env(0, obs="raw")
            fields = {"road": gym.spaces.Box(0, 1, (config.size, config.size), np.float32),
                      "visible": gym.spaces.Box(0, 1, (config.size, config.size), np.float32),
                      "ego": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)}
        fields["previous_action"] = gym.spaces.Box(np.array([-1, 0, 0], np.float32), 1, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(fields)
        self.action_space = self._sim.action_space
        self.last_metrics: dict = {}
        self.completed_episodes: list[dict] = []
        self.last_native_frames = 0
        self.done = True
        self._restored_stream = False

    def restore_training_stream(self, state: dict) -> None:
        if self.partition != "train":
            raise ValueError("Only training streams support continuation")
        self.stream.load_state_dict(state)
        self._restored_stream = True

    def _observation(self, raw: np.ndarray) -> Observation:
        started = time.perf_counter()
        if self.config.view == "native":
            observation = {"image": raw.copy(), "previous_action": self.previous_action.copy()}
            self.sensor_seconds = self._sim.unwrapped.sensor_seconds + time.perf_counter() - started
        else:
            observation = self.sensor.observe(self._sim.unwrapped.track_info, CarState.from_env(self._sim), self.previous_action)
            self.sensor_seconds = time.perf_counter() - started
        return observation

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Observation, dict]:
        if options:
            raise ValueError("Episode options are fixed by the protocol")
        if seed is not None:
            # Gym/SB3 seeds a stream, never supplies an arbitrary track identity.
            super().reset(seed=seed)
            if not self._restored_stream:
                self.stream = TrainingStream(seed, self.stream.worker, self.stream.workers)
            self.index = 0
        self._restored_stream = False
        if self.partition == "train":
            track_seed = self.stream.sample()
        else:
            seeds = self.partitions.seeds(self.partition)
            if self.index >= len(seeds):
                raise RuntimeError("Partition exhausted; create a new evaluation pass explicitly")
            track_seed = seeds[self.index]
            self.index += 1
        assert self.partitions.contains(self.partition, track_seed)
        self.track_seeds.append(track_seed)
        raw, _ = self._sim.reset(seed=track_seed)
        self.previous_action.fill(0)
        self.last_metrics = {}
        self.done = False
        return self._observation(raw), {}

    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, bool, dict]:
        if self.done:
            raise RuntimeError("Reset before stepping a finished episode")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Invalid action: no silent clipping at evaluator boundary")
        reward = 0.0
        for count in range(1, self.config.action_repeat + 1):
            raw, value, terminated, truncated, info = self._sim.step(action)
            reward += value
            if terminated or truncated:
                self.last_metrics = dict(info["episode_metrics"])
                self.completed_episodes.append({"track_seed": self.track_seeds[-1], **self.last_metrics})
                break
        self.previous_action[:] = action
        self.last_native_frames = count
        self.done = terminated or truncated
        return self._observation(raw), reward, terminated, truncated, {"native_frames": count}

    def close(self) -> None:
        self._sim.close()
