"""Frozen recurrent behavior proposals selected by programmed simulation outcomes."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cars.env.protocol import LocalSensor, Observation, ProtocolConfig
from cars.models.physics import PhysicsDynamics
from cars.planning.local import centreline, extend
from cars.planning.policy_rollouts import bounded_actions, evaluate_feedback, imagined_observations
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
        return bounded_actions(base, residual)


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
        path = extend(centreline(observation, self.protocol))
        world = LocalRoadWorld(observation, self.protocol, path, self.config.half_width_prior)
        return self._search(observation, world, path)

    def _search(self, observation: Observation, world: LocalRoadWorld,
                path: np.ndarray) -> np.ndarray:
        """Search with separate road and objective-path inputs.

        The local default shares one path. Explicit privileged interventions may
        replace the objective path without changing the world's road prior.
        """
        base, residuals = self._propose(observation)
        count, horizon = self.config.candidates, self.config.horizon
        batch = evaluate_feedback(self.model, self.actor.model, self.sensor, observation,
            world, path, base, self.actor.memory, residuals, self.config)
        selected = int(np.argmin(batch.costs))
        self.model.commit(selected)
        action = batch.actions[selected, 0]
        self.candidate_actions, self.candidate_costs = batch.actions, batch.costs
        self.diagnostics = {"model_transitions": count * horizon,
            "internal_physics_frames": count * horizon * self.protocol.action_repeat,
            "actor_forward_examples": 1 + count * (horizon - 1), "candidates": count,
            "selected_candidate": selected, "predicted_cost": float(batch.costs[selected]),
            "base_action": base.tolist(), "overrode_base": bool(np.max(abs(action - base)) > 1e-6),
            "predicted_next_ego": self.model.first_ego[selected].tolist()}
        return action
