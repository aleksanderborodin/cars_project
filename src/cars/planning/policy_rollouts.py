"""Frozen feedback rollouts with owned first-transition snapshots for batched search."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from cars.env.protocol import LocalSensor, Observation
from cars.models.physics import PhysicsDynamics
from cars.planning.occupancy import binary_road
from cars.planning.world import LocalRoadWorld

if TYPE_CHECKING:
    from cars.planning.policy_search import PolicySearchConfig

Memory = tuple[np.ndarray, ...]


class Predictor(Protocol):
    def predict(self, observation: Observation, *, state: Memory,
                episode_start: np.ndarray, deterministic: bool) -> tuple[np.ndarray, Memory]: ...


@dataclass
class RolloutBatch:
    actions: np.ndarray
    costs: np.ndarray
    first_estimates: np.ndarray
    first_ego: np.ndarray


def imagined_observations(world: LocalRoadWorld, sensor: LocalSensor, tree,
                          pose: np.ndarray, ego: np.ndarray, previous: np.ndarray,
                          renderer: str = "reference") -> Observation:
    """Moving local footprint of the assumed world, without hidden evaluation state."""
    points = sensor.points[sensor.visible]
    c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    rotation = np.stack((np.stack((c, -s), axis=-1), np.stack((s, c), axis=-1)), axis=1)
    positions = (np.einsum("bij,pj->bpi", rotation, points) + pose[:, None, :2]).reshape(-1, 2)
    occupancy = (binary_road(world, tree, positions) if renderer == "sparse" else
                 world.query(positions, centre_distance=tree.query(positions)[0]).grip >= .5)
    road = np.zeros((len(pose), *sensor.visible.shape), dtype=np.float32)
    road[:, sensor.visible] = occupancy.reshape(len(pose), -1).astype(np.float32)
    return {"road": road, "visible": np.broadcast_to(sensor.visible, road.shape).astype(np.float32),
            "ego": ego.astype(np.float32), "previous_action": previous.astype(np.float32)}


def bounded_actions(base: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return np.clip(base + residual, [-1., 0., 0.], 1.).astype(np.float32)


def evaluate_feedback(model: PhysicsDynamics, predictor: Predictor, sensor: LocalSensor,
                      observation: Observation, world: LocalRoadWorld, path: np.ndarray,
                      base: np.ndarray, root_memory: Memory, residuals: np.ndarray,
                      config: "PolicySearchConfig") -> RolloutBatch:
    """All branches start at the same real root; no commit or real-memory mutation.

    Arithmetic/order matches the original knot search. Road queries depend on
    world.path even when a privileged diagnostic changes the objective path.
    Returned snapshots survive model.begin() destroying/replacing the next batch.
    """
    count, horizon, channels = residuals.shape
    if channels != 3 or horizon != config.horizon or count < 1:
        raise ValueError("Expected nonempty candidate-by-horizon-by-three residuals")
    first = bounded_actions(base, residuals[:, 0])
    model.begin(observation, world, count)
    pose = np.zeros((count, 3))
    ego = np.repeat(observation["ego"][None], count, axis=0)
    previous = np.repeat(observation["previous_action"][None], count, axis=0)
    memory = tuple(np.repeat(value, count, axis=1).copy() for value in root_memory)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    start_index = int(np.argmin(np.sum(path**2, axis=1)))
    total = np.zeros(count)
    road_query = world.query(pose[:, :2])
    sequences = []
    for step in range(horizon):
        if step == 0:
            action = first
        else:
            imagined = imagined_observations(world, sensor, model.tree, pose, ego, previous,
                                             config.observation_renderer)
            proposed, memory = predictor.predict(imagined, state=memory,
                episode_start=np.zeros(count, dtype=bool), deterministic=True)
            action = bounded_actions(proposed, residuals[:, step])
        sequences.append(action.copy())
        pose, ego = model.advance(pose, ego, action, previous, road_query.grip)
        previous = action
        distances = np.sum((pose[:, None, :2] - path[None])**2, axis=-1)
        nearest = np.argmin(distances, axis=1)
        centre_distance = np.sqrt(distances[np.arange(count), nearest])
        road_distance = (centre_distance if path is world.path
                         else model.tree.query(pose[:, :2])[0])
        road_query = world.query(pose[:, :2], centre_distance=road_distance)
        total += config.departure_cost * np.maximum(1.5 - road_query.margin, 0)**2
        total += config.centre_cost * centre_distance**2
        total += config.unknown_cost * (~road_query.visible)
        total += .01 * ego[:, 1]**2
    total -= arc[nearest] - arc[start_index]
    assert model.first_estimates is not None and model.first_ego is not None
    return RolloutBatch(np.stack(sequences, axis=1), total.copy(),
                        np.asarray(model.first_estimates).copy(), model.first_ego.copy())
