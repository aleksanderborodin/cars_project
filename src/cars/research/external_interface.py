"""Restricted local-array boundary for external, feedforward control algorithms.

The harness owns the wrapper; an agent receives arrays, never this environment.
This is a declared local adaptation, not native pixels or recurrent-policy parity.
"""
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np

from cars.env.protocol import DrivingEnv, Observation, ProtocolConfig


def _action(value: np.ndarray, low: tuple[float, float, float]) -> np.ndarray:
    value = np.asarray(value)
    if (value.dtype.kind not in "iuf" or value.ndim < 1 or value.shape[-1] != 3 or not np.isfinite(value).all()
            or (value < low).any() or (value > 1).any()):
        raise ValueError("Finite bounded three-channel actions required; no silent clipping")
    return value.astype(np.float64)


def physical_action(normalized: np.ndarray) -> np.ndarray:
    """Bijective affine convention before the final float32 conversion."""
    action = _action(normalized, (-1., -1., -1.)).copy()
    action[..., 1:] = (action[..., 1:] + 1.) / 2.
    return action.astype(np.float32)


def normalized_action(physical: np.ndarray) -> np.ndarray:
    action = _action(physical, (-1., 0., 0.)).copy()
    action[..., 1:] = action[..., 1:] * 2. - 1.
    return action.astype(np.float32)


@dataclass(frozen=True)
class LocalVectorCodec:
    """Lossless float32 road + raw ego + actual previous-action representation.

    The constant visible footprint is checked and reconstructible from protocol.
    No resizing, clipping, fitted normalization, padding or extra history is added.
    """
    protocol: ProtocolConfig

    def __post_init__(self) -> None:
        if self.protocol.view != "local":
            raise ValueError("Only restricted local observations are supported")

    @property
    def dimension(self) -> int:
        return self.protocol.size**2 + 6

    @property
    def visible(self) -> np.ndarray:
        axis = np.linspace(-self.protocol.radius, self.protocol.radius, self.protocol.size)
        forward, left = np.meshgrid(axis, axis, indexing="ij")
        return (forward**2 + left**2 <= self.protocol.radius**2).astype(np.float32)

    def encode(self, observation: Observation) -> np.ndarray:
        if set(observation) != {"road", "visible", "ego", "previous_action"}:
            raise ValueError("Only the exact allowed local observation fields are accepted")
        road, visible = np.asarray(observation["road"]), np.asarray(observation["visible"])
        ego, previous = np.asarray(observation["ego"]), np.asarray(observation["previous_action"])
        if any(v.dtype.kind not in "biuf" for v in (road, visible, ego, previous)):
            raise ValueError("Real numerical observation arrays required")
        if (road.shape != (self.protocol.size, self.protocol.size)
                or visible.shape != road.shape or ego.shape != (3,) or previous.shape != (3,)):
            raise ValueError("Observation shapes do not match the declared local protocol")
        if not np.array_equal(visible, self.visible):
            raise ValueError("Visibility must equal the declared fixed footprint")
        if not np.logical_or(road == 0, road == 1).all() or (road[visible == 0] != 0).any():
            raise ValueError("Road must be binary and zero outside the visible footprint")
        _action(previous, (-1., 0., 0.))
        vector = np.concatenate((road.reshape(-1), ego, previous)).astype(np.float32)
        if not np.isfinite(vector).all():
            raise ValueError("Nonfinite observation")
        return vector

    def decode(self, vector: np.ndarray) -> Observation:
        vector = np.asarray(vector)
        if vector.shape != (self.dimension,):
            raise ValueError("Expected one observation in the declared local vector space")
        observation = {"road": vector[:-6].reshape(self.protocol.size, self.protocol.size).copy(),
            "visible": self.visible, "ego": vector[-6:-3].copy(),
            "previous_action": vector[-3:].copy()}
        self.encode(observation)  # same validation at both sides of replay
        return observation

    def configuration(self) -> dict:
        return {"version": 1, "protocol": asdict(self.protocol), "protocol_id": self.protocol.identifier,
            "observation": "row_major_binary_road; raw_body_velocity_yaw_rate; actual_previous_action",
            "dimension": self.dimension, "dtype": "float32", "history": 1,
            "visibility": "constant_protocol_footprint_reconstructed_and_checked",
            "action": "normalized_steer; gas=(u+1)/2; brake=(u+1)/2; no_clipping",
            "normalization": "none; raw_numeric_units", "agent_receives_environment": False}


class ExternalLocalEnv(gym.Wrapper):
    """Infrastructure-only Gym adapter; environment seeds/info never reach agent arrays."""

    def __init__(self, env: DrivingEnv):
        self.codec = LocalVectorCodec(env.config)
        if env.partition not in ("train", "development"):
            raise ValueError("External baseline collection is restricted to train/development")
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.codec.dimension,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (3,), np.float32)
        self.real_decisions = 0
        self.native_frames = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        observation, _ = self.env.reset(seed=seed, options=options)
        return self.codec.encode(observation), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if np.asarray(action).shape != (3,):
            raise ValueError("The environment expects a single action")
        observation, reward, terminated, truncated, info = self.env.step(physical_action(action))
        frames = info["native_frames"]
        if isinstance(frames, bool) or not isinstance(frames, (int, np.integer)) or not 1 <= frames <= self.env.config.action_repeat:
            raise ValueError("Invalid actual native-frame count")
        self.real_decisions += 1
        self.native_frames += int(frames)
        return self.codec.encode(observation), reward, terminated, truncated, {"native_frames": int(frames)}
