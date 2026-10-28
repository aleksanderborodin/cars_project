"""Simulation-based candidate selection with a programmed vehicle model."""
from cars.env.protocol import Observation, ProtocolConfig
from cars.models.physics import PhysicsDynamics
from cars.planning.search import BehaviorSearch, SearchConfig
from cars.planning.world import LocalRoadWorld


class PhysicsSearch(BehaviorSearch):
    method = "physics_search"
    allowed_views = ("local",)

    def __init__(self, protocol: ProtocolConfig, config: SearchConfig | None = None):
        if protocol.view not in self.allowed_views:
            raise ValueError(f"This physics planner requires {self.allowed_views} observations")
        super().__init__(protocol, PhysicsDynamics(protocol.action_repeat), config)
        self.model: PhysicsDynamics
        self.reset()

    def configuration(self) -> dict:
        return {**super().configuration(), "model": self.model.kind,
                "simulator": self.model.configuration(),
                "proposal": "fixed_parameter_feedback_controllers; no learned behavior scorer"}

    def reset(self) -> None:
        super().reset()
        self.model.reset()

    def _begin_rollout(self, observation: Observation, world: LocalRoadWorld, count: int) -> None:
        self.model.begin(observation, world, count)

    def _commit_rollout(self, selected: int) -> None:
        self.model.commit(selected)
        assert self.model.first_ego is not None
        self.diagnostics.update({
            "internal_physics_frames": self.diagnostics["model_transitions"] * self.protocol.action_repeat,
            "predicted_next_ego": self.model.first_ego[selected].tolist(),
        })
