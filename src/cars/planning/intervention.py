"""Explicit privileged module substitutions around an unchanged local actor.

This development diagnostic never consumes private candidate-ranking outputs or
an evaluation simulator clone. Its map channel is the existing global protocol.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from cars.env.protocol import Observation, ProtocolConfig
from cars.planning.global_search import FullRoadWorld
from cars.planning.local import centreline, extend
from cars.planning.policy_search import PolicyPhysicsSearch, PolicySearchConfig
from cars.planning.visible_reference import visible_reference
from cars.planning.world import LocalRoadWorld, RoadQuery

LOCAL_FIELDS = frozenset(("road", "visible", "ego", "previous_action"))
MAP_FIELDS = frozenset(("map_centre", "map_left", "map_right"))


@dataclass(frozen=True)
class InterventionConfig:
    reference: str = "reconstructed"
    continuation: str = "prior"
    version: int = 1

    def __post_init__(self) -> None:
        if (self.reference not in ("reconstructed", "oracle_visible")
                or self.continuation not in ("prior", "oracle") or self.version != 1):
            raise ValueError("Unknown intervention")


class OracleContinuationWorld(FullRoadWorld):
    """Replace hidden occupancy/margins, retaining current observations/penalty.

    The visibility flag remains the original root sensor footprint in BOTH arms.
    It indexes a fixed objective penalty, not a claim that oracle road is unknown.
    The road world's reference stays the original local reconstruction, regardless
    of the independent objective-path intervention.
    """
    version = "oracle_continuation_fixed_visibility_v1"

    def __init__(self, observation: Observation, protocol: ProtocolConfig,
                 path: np.ndarray, half_width: float):
        super().__init__(observation, protocol, half_width)
        self.path = path

    def query(self, positions: np.ndarray, *, centre_distance: np.ndarray | None = None) -> RoadQuery:
        local = LocalRoadWorld.query(self, positions, centre_distance=centre_distance)
        grid = ((positions - self.origin) / self.protocol.resolution).T
        grip = map_coordinates(self.full_road, grid, order=1, mode="constant", cval=0.)
        margin = map_coordinates(self.full_margin, grid, order=1, mode="constant", cval=-100.)
        return RoadQuery(np.where(local.visible, local.grip, grip),
                         np.where(local.visible, local.margin, margin), local.visible)


def intervention_worlds(observation: Observation, protocol: ProtocolConfig,
                        config: InterventionConfig, half_width: float
                        ) -> tuple[Observation, LocalRoadWorld, np.ndarray, bool]:
    if set(observation) != LOCAL_FIELDS | MAP_FIELDS:
        raise ValueError("Explicit global diagnostic schema required; no raw state or extra fields")
    local = {key: observation[key] for key in LOCAL_FIELDS}
    local_path = extend(centreline(local, protocol))
    world = (LocalRoadWorld(local, protocol, local_path, half_width) if config.continuation == "prior"
             else OracleContinuationWorld(observation, protocol, local_path, half_width))
    path, fallback = local_path, False
    if config.reference == "oracle_visible":
        visible = visible_reference(observation["map_centre"], protocol.radius, protocol.resolution)
        fallback = visible is None
        path = local_path if fallback else extend(visible)
    return local, world, path, fallback


class PolicyGeometryIntervention(PolicyPhysicsSearch):
    method = "policy_geometry_intervention"

    def __init__(self, checkpoint: Path, protocol: ProtocolConfig,
                 config: PolicySearchConfig | None = None,
                 intervention: InterventionConfig | None = None):
        if protocol.view != "global":
            raise ValueError("Module interventions require the explicit global diagnostic protocol")
        self.input_protocol = protocol
        self.intervention = intervention or InterventionConfig()
        # Only this adapter sees map fields. The actor and all imagined actor
        # observations retain the exact training-time local schema and footprint.
        super().__init__(checkpoint, replace(protocol, view="local"), config)

    def act(self, observation: Observation) -> np.ndarray:
        local, world, path, fallback = intervention_worlds(
            observation, self.protocol, self.intervention, self.config.half_width_prior)
        action = self._search(local, world, path)
        self.diagnostics["visible_reference_fallback"] = fallback
        return action

    def configuration(self) -> dict:
        return {**super().configuration(), "intervention": asdict(self.intervention),
            "input_protocol": asdict(self.input_protocol),
            "geometry_access": "privileged_development_module_intervention",
            "actor_real_and_imagined_inputs": sorted(LOCAL_FIELDS),
            "road_prior": ("constant_terminal_tangent" if self.intervention.continuation == "prior"
                           else "oracle_road_outside_root_footprint; original_raster_inside"),
            "road_query": (LocalRoadWorld.version if self.intervention.continuation == "prior"
                           else OracleContinuationWorld.version),
            "objective_reference": self.intervention.reference,
            "reference_extension": "same_100m_terminal_tangent; never_hidden_centreline",
            "unknown_space_penalty": "original_root_visibility_mask_in_all_conditions",
            "world_path": "unchanged_local_reconstruction_independent_of_objective_reference",
            "private_candidate_oracle_access": False,
            "oracle_vehicle_state_access": False,
            "training_at_inference": False}
