"""Local road reconstruction and a declared hand-engineered control reference."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from cars.env.protocol import Observation, ProtocolConfig


def centreline(observation: Observation, config: ProtocolConfig) -> np.ndarray:
    """Follow connected row intervals using only visible raster cells.

    Forward-driving heuristic, not a general map reconstruction method. Fails
    gracefully when no forward interval exists; no track direction is provided.
    """
    road = observation["road"] > .5
    axis = np.linspace(-config.radius, config.radius, config.size)
    points: list[list[float]] = []
    last = 0.
    for i in np.flatnonzero(axis >= -3):
        occupied = np.flatnonzero(road[i])
        if not len(occupied):
            if len(points) > 3:
                break
            continue
        intervals = np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1)
        centres = [(axis[group[0]] + axis[group[-1]]) / 2 for group in intervals]
        selected = int(np.argmin(np.abs(np.asarray(centres) - last)))
        group = intervals[selected]
        # A clipped cross-section is not evidence that the road bends toward
        # the centre of the visibility circle. Stop before this sensor artifact.
        complete = (group[0] > 0 and group[-1] + 1 < config.size
                    and observation["visible"][i, group[0] - 1] > .5
                    and observation["visible"][i, group[-1] + 1] > .5)
        if not complete and len(points) > 3:
            break
        value = centres[selected]
        # Stop at a disconnected forward branch rather than jumping across grass.
        if points and abs(value - last) > 6:
            break
        points.append([axis[i], value])
        last = value
    if len(points) < 3:
        return np.array([[0., 0.], [2., 0.], [4., 0.]])
    path = np.asarray(points)
    path[:, 1] = gaussian_filter1d(path[:, 1], 1.5, mode="nearest")
    return path


def extend(path: np.ndarray, length: float = 100.) -> np.ndarray:
    """Constant terminal tangent prior; no hidden-road simulator queries."""
    direction = path[-1] - path[max(0, len(path) - 6)]
    direction /= max(np.linalg.norm(direction), 1e-6)
    return np.concatenate((path, path[-1] + np.arange(1, length + 1)[:, None] * direction))


def controls(pose: np.ndarray, ego: np.ndarray, path: np.ndarray,
             speeds: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Vectorized candidate feedback controllers with common local geometry."""
    distances = np.sum((pose[:, None, :2] - path[None])**2, axis=-1)
    nearest = np.argmin(distances, axis=1)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    lookahead = 4 + .22 * np.maximum(ego[:, 0], 0)
    indices = np.searchsorted(arc, arc[nearest] + lookahead).clip(0, len(path) - 1)
    delta = path[indices] - pose[:, :2]
    left = -np.sin(pose[:, 2]) * delta[:, 0] + np.cos(pose[:, 2]) * delta[:, 1]
    curvature = 2 * left / np.maximum(np.sum(delta**2, axis=1), 1.)
    steer = np.clip(-gains * np.arctan(4 * curvature), -1, 1)
    error = speeds - ego[:, 0]
    slip = np.arctan2(ego[:, 1], np.maximum(abs(ego[:, 0]), 1.))
    gas = np.clip(.12 * error, 0, .65) * np.clip(1 - abs(slip) / .6, .1, 1)
    brake = np.clip(-.08 * error, 0, .8)
    return np.stack([steer, gas, brake], axis=-1).astype(np.float32)


@dataclass(frozen=True)
class LocalConfig:
    max_speed: float = 70.
    lateral_accel: float = 100.
    braking_accel: float = 50.
    visibility_braking: bool = True


class LocalPursuit:
    method = "local_pursuit"

    def __init__(self, protocol: ProtocolConfig, config: LocalConfig | None = None):
        self.protocol, self.config = protocol, config or LocalConfig()
        self.diagnostics: dict = {}

    def configuration(self) -> dict:
        return asdict(self.config)

    def reset(self) -> None:
        self.diagnostics = {}

    def act(self, observation: Observation) -> np.ndarray:
        path = centreline(observation, self.protocol)
        tangent = np.arctan2(np.gradient(path[:, 1]), np.gradient(path[:, 0]))
        curvature = np.gradient(np.unwrap(tangent)) / np.maximum(np.linalg.norm(np.gradient(path, axis=0), axis=1), 1e-6)
        speed = min(self.config.max_speed, np.sqrt(self.config.lateral_accel / max(np.max(abs(curvature)), 1e-5)))
        if self.config.visibility_braking:
            speed = min(speed, np.sqrt(2 * self.config.braking_accel * max(np.linalg.norm(path[-1]) - 4, 2)))
        self.diagnostics = {"target_speed": float(speed), "model_transitions": 0}
        return controls(np.zeros((1, 3)), observation["ego"][None], path,
                        np.array([speed]), np.ones(1))[0]


class FixedBehavior:
    """Same candidate controller/reconstruction as search, no imagined rollouts."""
    method = "fixed_behavior"

    def __init__(self, protocol: ProtocolConfig, speed: float = 60., gain: float = 1.):
        if speed <= 0 or gain <= 0:
            raise ValueError("Positive behavior parameters required")
        self.protocol, self.speed, self.gain = protocol, speed, gain
        self.reset()

    def reset(self) -> None:
        self.diagnostics = {"model_transitions": 0, "target_speed": self.speed, "steering_gain": self.gain}

    def configuration(self) -> dict:
        return {"speed": self.speed, "gain": self.gain, "road_prior": "constant_terminal_tangent",
                "control_family": "same_as_model_search_without_simulation"}

    def act(self, observation: Observation) -> np.ndarray:
        path = extend(centreline(observation, self.protocol))
        return controls(np.zeros((1, 3)), observation["ego"][None], path,
                        np.array([self.speed]), np.array([self.gain]))[0]
