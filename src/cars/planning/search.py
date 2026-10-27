"""Finite behavior search in one fitted internal model (engineering baseline)."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from cars.env.protocol import Observation, ProtocolConfig
from cars.models.dynamics import Dynamics
from cars.planning.local import centreline, controls, extend
from cars.training import sha256


@dataclass(frozen=True)
class SearchConfig:
    horizon: int = 20
    speeds: tuple[float, ...] = (30., 40., 50., 60., 70., 80., 90.)
    steering_gains: tuple[float, ...] = (.8, 1., 1.2)
    departure_cost: float = 8.
    centre_cost: float = .04
    unknown_cost: float = .2
    half_width_prior: float = 6.67


class BehaviorSearch:
    method = "model_search"

    def __init__(self, protocol: ProtocolConfig, model: Dynamics, config: SearchConfig | None = None,
                 model_artifact: Path | None = None):
        if model.repeat != protocol.action_repeat:
            raise ValueError("Dynamics and control frequencies differ")
        self.protocol, self.model, self.config = protocol, model, config or SearchConfig()
        self.model_artifact = model_artifact
        if self.config.horizon < 1 or not self.config.speeds or not self.config.steering_gains:
            raise ValueError("Empty search")
        self.diagnostics: dict = {}

    def configuration(self) -> dict:
        return {**asdict(self.config), "road_prior": "constant_terminal_tangent",
                "search": "finite_speed_and_steering_gain_feedback_candidates",
                "model": "observed_ego_ridge_dynamics",
                "model_artifact": str(self.model_artifact.resolve()) if self.model_artifact else None,
                "model_sha256": sha256(self.model_artifact / "model.npz") if self.model_artifact else None}

    def reset(self) -> None:
        self.diagnostics = {}
        self.candidate_actions = None
        self.candidate_costs = None

    def act(self, observation: Observation) -> np.ndarray:
        path = extend(centreline(observation, self.protocol))
        speeds, gains = np.meshgrid(self.config.speeds, self.config.steering_gains)
        speeds, gains = speeds.ravel(), gains.ravel()
        count = len(speeds)
        pose = np.zeros((count, 3))
        ego = np.repeat(observation["ego"][None], count, axis=0)
        previous = np.repeat(observation["previous_action"][None], count, axis=0)
        road = observation["road"] > .5
        visible_mask = observation["visible"] > .5
        # Unknown space is not an observed road boundary.
        signed = (distance_transform_edt(road | ~visible_mask)
                  - distance_transform_edt(~road & visible_mask)) * self.protocol.resolution
        total = np.zeros(count)
        first = controls(pose, ego, path, speeds, gains)
        sequences = []
        start_index = int(np.argmin(np.sum(path**2, axis=1)))
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        for step in range(self.config.horizon):
            action = first if step == 0 else controls(pose, ego, path, speeds, gains)
            sequences.append(action.copy())
            grid = ((pose[:, :2] + self.protocol.radius) / self.protocol.resolution).T
            grip = map_coordinates(observation["road"], grid, order=1, mode="constant", cval=1.)
            pose, ego = self.model.advance(pose, ego, action, previous, grip)
            previous = action
            distances = np.sum((pose[:, None, :2] - path[None])**2, axis=-1)
            nearest = np.argmin(distances, axis=1)
            centre_distance = np.sqrt(distances[np.arange(count), nearest])
            visible = np.sum(pose[:, :2]**2, axis=1) <= self.protocol.radius**2
            grid = ((pose[:, :2] + self.protocol.radius) / self.protocol.resolution).T
            margin = map_coordinates(signed, grid, order=1, mode="constant", cval=-self.protocol.radius)
            margin = np.where(visible, margin, self.config.half_width_prior - centre_distance)
            total += self.config.departure_cost * np.maximum(1.5 - margin, 0)**2
            total += self.config.centre_cost * centre_distance**2
            total += self.config.unknown_cost * (~visible)
            total += .01 * ego[:, 1]**2
        total -= arc[nearest] - arc[start_index]
        selected = int(np.argmin(total))
        self.candidate_actions = np.stack(sequences, axis=1)
        self.candidate_costs = total.copy()
        self.diagnostics = {"model_transitions": count * self.config.horizon,
                            "candidates": count, "target_speed": float(speeds[selected]),
                            "steering_gain": float(gains[selected]),
                            "selected_candidate": selected,
                            "predicted_cost": float(total[selected])}
        return first[selected]
