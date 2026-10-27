"""State v1: metres, seconds, radians; heading points forward (+x at zero).

Car frame is (forward, left). Wheel order is upstream's front-left,
front-right, rear-left, rear-right. Raw observations use STATE_FIELDS.
"""
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float32]
STATE_VERSION = 1
STATE_FIELDS = (["x", "y", "heading", "vx", "vy", "yaw_rate"]
                + [f"wheel_omega_{i}" for i in range(4)]
                + [f"wheel_steer_{i}" for i in range(4)]
                + [f"wheel_on_road_{i}" for i in range(4)]
                + ["tile_idx", "progress"])
OBS_VERSION = 1
LOOKAHEAD = 20
OBS_SCHEMA = (["speed/100", "v_forward/100", "v_left/100", "yaw_rate/5",
               "heading_error/pi", "lateral_offset/half_width"]
              + [f"wheel_on_road_{i}" for i in range(4)]
              + [f"lookahead_{i}_{field}" for i in range(1, LOOKAHEAD + 1)
                 for field in ("forward/100", "left/100", "curvature*10")]
              + ["progress", "previous_steer", "previous_gas", "previous_brake"])


def wrap_angle(angle: Any) -> Any:
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class TrackInfo:
    centre: NDArray[np.float64]
    tangent: NDArray[np.float64]
    half_width: NDArray[np.float64]

    @classmethod
    def from_env(cls, env: gym.Env) -> "TrackInfo":
        raw = env.unwrapped
        from gymnasium.envs.box2d.car_racing import TRACK_WIDTH

        track = np.asarray(raw.track, dtype=np.float64)
        return cls(track[:, 2:4].copy(), wrap_angle(track[:, 1] + np.pi / 2),
                   np.full(len(track), getattr(raw, "track_half_width", TRACK_WIDTH), dtype=np.float64))

    @property
    def left(self) -> NDArray[np.float64]:
        normal = np.column_stack((-np.sin(self.tangent), np.cos(self.tangent)))
        return self.centre + self.half_width[:, None] * normal

    @property
    def right(self) -> NDArray[np.float64]:
        return 2 * self.centre - self.left

    @property
    def curvature(self) -> NDArray[np.float64]:
        ds = np.linalg.norm(np.roll(self.centre, -1, axis=0)
                            - np.roll(self.centre, 1, axis=0), axis=1)
        return wrap_angle(np.roll(self.tangent, -1) - np.roll(self.tangent, 1)) / np.maximum(ds, 1e-6)

    def as_array(self) -> Array:
        return np.ascontiguousarray(np.column_stack((self.centre, self.tangent, self.half_width)), dtype=np.float32)


@dataclass(frozen=True)
class CarState:
    x: float
    y: float
    heading: float
    vx: float
    vy: float
    yaw_rate: float
    wheel_omega: tuple[float, ...]
    wheel_steer: tuple[float, ...]
    wheel_on_road: tuple[bool, ...]
    tile_idx: int
    progress: float

    @classmethod
    def from_env(cls, env: gym.Env, track: TrackInfo | None = None) -> "CarState":
        raw = env.unwrapped
        car = raw.car
        assert car is not None
        centre = track.centre if track is not None else raw.track_info.centre
        x, y = car.hull.position
        vx, vy = car.hull.linearVelocity
        idx = int(np.argmin(np.sum((centre - (x, y)) ** 2, axis=1)))
        return cls(float(x), float(y), float(wrap_angle(car.hull.angle + np.pi / 2)),
                   float(vx), float(vy), float(car.hull.angularVelocity),
                   tuple(float(w.omega) for w in car.wheels),
                   tuple(float(w.joint.angle) for w in car.wheels),
                   tuple(bool(w.tiles) for w in car.wheels), idx,
                   raw.tile_visited_count / len(centre))

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    @property
    def body_velocity(self) -> tuple[float, float]:
        c, s = np.cos(self.heading), np.sin(self.heading)
        return float(c * self.vx + s * self.vy), float(-s * self.vx + c * self.vy)

    @property
    def slip_angle(self) -> float:
        forward, left = self.body_velocity
        return float(np.arctan2(left, max(abs(forward), 1e-8)))

    @property
    def wheels_on_road(self) -> int:
        return sum(self.wheel_on_road)

    def as_array(self) -> Array:
        return np.asarray([self.x, self.y, self.heading, self.vx, self.vy, self.yaw_rate,
                           *self.wheel_omega, *self.wheel_steer, *self.wheel_on_road,
                           self.tile_idx, self.progress], dtype=np.float32)


class StateObs(gym.Wrapper):
    """Version 1: 20 points, spaced two tiles apart (~7 m), up to ~140 m.

    Units/scales are encoded in OBS_SCHEMA. Values are scaled, not clipped;
    observation bounds remain infinite to preserve off-road information.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (len(OBS_SCHEMA),), np.float32)
        self.previous_action = np.zeros(3, dtype=np.float32)

    def observation(self) -> Array:
        track = self.unwrapped.track_info
        state = CarState.from_env(self, track)
        idx = state.tile_idx
        c, s = np.cos(state.heading), np.sin(state.heading)
        delta = track.centre[(idx + 2 * np.arange(1, LOOKAHEAD + 1)) % len(track.centre)] - (state.x, state.y)
        points = np.column_stack(((delta[:, 0] * c + delta[:, 1] * s) / 100,
                                  (-delta[:, 0] * s + delta[:, 1] * c) / 100,
                                  track.curvature[(idx + 2 * np.arange(1, LOOKAHEAD + 1)) % len(track.centre)] * 10))
        normal = np.array([-np.sin(track.tangent[idx]), np.cos(track.tangent[idx])])
        lateral = np.dot(np.array([state.x, state.y]) - track.centre[idx], normal)
        forward, left = state.body_velocity
        return np.asarray([state.speed / 100, forward / 100, left / 100, state.yaw_rate / 5,
                           wrap_angle(track.tangent[idx] - state.heading) / np.pi,
                           lateral / track.half_width[idx], *state.wheel_on_road,
                           *points.ravel(), state.progress, *self.previous_action], dtype=np.float32)

    def reset(self, **kwargs: Any) -> tuple[Array, dict]:
        _, info = self.env.reset(**kwargs)
        self.previous_action[:] = 0
        return self.observation(), info

    def step(self, action: Array) -> tuple[Array, float, bool, bool, dict]:
        _, reward, terminated, truncated, info = self.env.step(action)
        self.previous_action[:] = action
        return self.observation(), reward, terminated, truncated, info
