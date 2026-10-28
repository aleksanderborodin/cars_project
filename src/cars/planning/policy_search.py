"""Frozen recurrent behavior proposals selected by programmed simulation outcomes."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cars.env.protocol import LocalSensor, Observation, ProtocolConfig
from cars.models.physics import PhysicsDynamics
from cars.planning.local import centreline, extend
from cars.planning.occupancy import binary_road
from cars.planning.world import LocalRoadWorld
from cars.research.recurrent import RecurrentDriver
from cars.training import sha256


@dataclass(frozen=True)
class PolicySearchConfig:
    candidates: int = 7
    horizon: int = 20
    planning_seed: int = 0
    steer_sigma: float = .10
    gas_sigma: float = .15
    brake_sigma: float = .15
    departure_cost: float = 8.
    centre_cost: float = .04
    unknown_cost: float = .2
    half_width_prior: float = 6.67
    observation_renderer: str = "reference"

    def __post_init__(self) -> None:
        if self.candidates < 1 or self.horizon < 1 or self.planning_seed < 0:
            raise ValueError("Positive search sizes and nonnegative seed required")
        if min(self.steer_sigma, self.gas_sigma, self.brake_sigma) < 0:
            raise ValueError("Nonnegative residual scales required")
        if self.observation_renderer not in ("reference", "sparse"):
            raise ValueError("Unknown imagined observation renderer")


def imagined_observations(world: LocalRoadWorld, sensor: LocalSensor, tree,
                          pose: np.ndarray, ego: np.ndarray, previous: np.ndarray,
                          renderer: str = "reference") -> Observation:
    """Rasterize each branch's current sensor footprint in the same assumed world.

    Future observations expose only a moving local footprint. Unknown road comes
    from this world's prior; no evaluation world or scenario ID is consulted.
    """
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


class FrozenProposals:
    def __init__(self, checkpoint: Path, protocol: ProtocolConfig, config: PolicySearchConfig | None = None):
        self.actor = RecurrentDriver(checkpoint)
        if protocol.view != "local" or self.actor.protocol != protocol:
            raise ValueError("Frozen-policy search requires the same declared local protocol as its actor")
        self.protocol, self.config = protocol, config or PolicySearchConfig()
        self.sensor = LocalSensor(protocol)
        self.reset()

    def reset(self) -> None:
        self.actor.reset()
        self.decision = 0
        self.diagnostics = {"model_transitions": 0}

    def configuration(self) -> dict:
        return {**asdict(self.config), "actor": self.actor.configuration(),
                "model_sha256": sha256(self.actor.root / "model.zip"),
                "proposals": "three linearly interpolated residual knots around frozen feedback policy; candidate zero unchanged",
                "planning_rng": "seed plus decision counter; reset per episode; selection stream separate",
                "training_at_inference": False}

    def _propose(self, observation: Observation) -> tuple[np.ndarray, np.ndarray]:
        base = self.actor.act(observation)  # only actual observations advance real actor memory
        generator = np.random.default_rng(np.random.SeedSequence([self.config.planning_seed, self.decision, 0]))
        knots = generator.normal(size=(self.config.candidates, 3, 3))
        knots *= [self.config.steer_sigma, self.config.gas_sigma, self.config.brake_sigma]
        knots[0] = 0.
        times = np.linspace(0., 2., self.config.horizon)
        residuals = np.stack([np.stack([np.interp(times, [0., 1., 2.], row[:, channel])
                              for channel in range(3)], axis=-1) for row in knots])
        self.decision += 1
        return base, residuals

    @staticmethod
    def _actions(base: np.ndarray, residual: np.ndarray) -> np.ndarray:
        return np.clip(base + residual, [-1., 0., 0.], 1.).astype(np.float32)


class RandomPolicyVariation(FrozenProposals):
    method = "policy_variation"

    def act(self, observation: Observation) -> np.ndarray:
        base, residuals = self._propose(observation)
        generator = np.random.default_rng(np.random.SeedSequence([self.config.planning_seed, self.decision - 1, 1]))
        selected = int(generator.integers(self.config.candidates))
        action = self._actions(base, residuals[selected, 0])
        self.diagnostics = {"model_transitions": 0, "actor_forward_examples": 1,
            "selected_candidate": selected, "base_action": base.tolist(),
            "overrode_base": bool(np.max(abs(action - base)) > 1e-6)}
        return action

    def configuration(self) -> dict:
        return {**super().configuration(), "selection": "uniform_candidate; no simulation"}


class PolicyPhysicsSearch(FrozenProposals):
    method = "policy_physics_search"

    def __init__(self, checkpoint: Path, protocol: ProtocolConfig, config: PolicySearchConfig | None = None):
        self.model = PhysicsDynamics(protocol.action_repeat)
        super().__init__(checkpoint, protocol, config)

    def reset(self) -> None:
        super().reset()
        self.model.reset()
        self.candidate_actions = self.candidate_costs = None

    def configuration(self) -> dict:
        return {**super().configuration(), "simulator": self.model.configuration(),
                "selection": "minimum_simulated_cost; same_progress_clearance_center_slip_unknown_utility",
                "road_prior": "constant_terminal_tangent", "road_query": LocalRoadWorld.version,
                "actor_memory": "independent imagined branches; real memory advances only on actual observations"}

    def act(self, observation: Observation) -> np.ndarray:
        base, residuals = self._propose(observation)
        count, horizon = self.config.candidates, self.config.horizon
        first = self._actions(base, residuals[:, 0])
        path = extend(centreline(observation, self.protocol))
        world = LocalRoadWorld(observation, self.protocol, path, self.config.half_width_prior)
        self.model.begin(observation, world, count)
        pose = np.zeros((count, 3))
        ego = np.repeat(observation["ego"][None], count, axis=0)
        previous = np.repeat(observation["previous_action"][None], count, axis=0)
        # The actor has processed the actual root observation once. Every future
        # branch starts from a copy of that state, never a shared mutable view.
        memory = tuple(np.repeat(value, count, axis=1).copy() for value in self.actor.memory)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        start_index = int(np.argmin(np.sum(path**2, axis=1)))
        total = np.zeros(count)
        road_query = world.query(pose[:, :2])
        sequences = []
        for step in range(horizon):
            if step == 0:
                action = first
            else:
                imagined = imagined_observations(world, self.sensor, self.model.tree, pose, ego, previous,
                                                 self.config.observation_renderer)
                proposed, memory = self.actor.model.predict(imagined, state=memory,
                    episode_start=np.zeros(count, dtype=bool), deterministic=True)
                action = self._actions(proposed, residuals[:, step])
            sequences.append(action.copy())
            pose, ego = self.model.advance(pose, ego, action, previous, road_query.grip)
            previous = action
            distances = np.sum((pose[:, None, :2] - path[None])**2, axis=-1)
            nearest = np.argmin(distances, axis=1)
            centre_distance = np.sqrt(distances[np.arange(count), nearest])
            road_query = world.query(pose[:, :2], centre_distance=centre_distance)
            total += self.config.departure_cost * np.maximum(1.5 - road_query.margin, 0)**2
            total += self.config.centre_cost * centre_distance**2
            total += self.config.unknown_cost * (~road_query.visible)
            total += .01 * ego[:, 1]**2
        total -= arc[nearest] - arc[start_index]
        selected = int(np.argmin(total))
        self.model.commit(selected)
        action = first[selected]
        self.candidate_actions, self.candidate_costs = np.stack(sequences, axis=1), total.copy()
        self.diagnostics = {"model_transitions": count * horizon,
            "internal_physics_frames": count * horizon * self.protocol.action_repeat,
            "actor_forward_examples": 1 + count * (horizon - 1), "candidates": count,
            "selected_candidate": selected, "predicted_cost": float(total[selected]),
            "base_action": base.tolist(), "overrode_base": bool(np.max(abs(action - base)) > 1e-6),
            "predicted_next_ego": self.model.first_ego[selected].tolist()}
        return action
