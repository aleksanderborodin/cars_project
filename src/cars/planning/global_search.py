"""Explicit privileged geometry diagnostic; the vehicle-state observer is unchanged."""
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from cars.env.protocol import Observation, ProtocolConfig
from cars.planning.physics_search import PhysicsSearch
from cars.planning.world import LocalRoadWorld, RoadQuery


def ordered_path(centre: np.ndarray, resolution: float) -> np.ndarray:
    """Full ordered loop, starting just behind the car; uniformly sampled in arc length.

    This replaces local row reconstruction and supplies track direction/topology.
    It is a declared representation/information intervention, not hidden local input.
    """
    nearest = int(np.argmin(np.sum(centre**2, axis=1)))
    points = np.roll(centre, 2 - nearest, axis=0)
    points = np.concatenate((points, points[:1]))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    samples = np.arange(0., arc[-1], resolution)
    return np.column_stack([np.interp(samples, arc, points[:, axis]) for axis in (0, 1)])


class FullRoadWorld(LocalRoadWorld):
    version = "global_geometry_raster_v1"

    def __init__(self, observation: Observation, protocol: ProtocolConfig, half_width: float):
        path = ordered_path(observation["map_centre"], protocol.resolution)
        super().__init__(observation, protocol, path, half_width)
        left, right = observation["map_left"], observation["map_right"]
        quads = np.stack((left, right, np.roll(right, 1, axis=0), np.roll(left, 1, axis=0)), axis=1)
        scale = protocol.resolution
        self.origin = np.floor(quads.min(axis=(0, 1)) / scale) * scale - 4 * scale
        extent = np.ceil((quads.max(axis=(0, 1)) - self.origin) / scale).astype(int) + 5
        road = np.zeros(tuple(extent), dtype=bool)
        for quad in quads:
            low = np.maximum(0, np.ceil((quad.min(axis=0) - self.origin) / scale).astype(int))
            high = np.minimum(extent - 1, np.floor((quad.max(axis=0) - self.origin) / scale).astype(int))
            if np.any(high < low):
                continue
            a, b = slice(low[0], high[0] + 1), slice(low[1], high[1] + 1)
            x, y = np.meshgrid(np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1), indexing="ij")
            relative = self.origin + np.stack((x, y), axis=-1)[..., None, :] * scale - quad
            edges = np.roll(quad, -1, axis=0) - quad
            cross = edges[:, 0] * relative[..., 1] - edges[:, 1] * relative[..., 0]
            road[a, b] |= np.all(cross >= -1e-7, axis=-1) | np.all(cross <= 1e-7, axis=-1)
        self.full_road = road.astype(float)
        self.full_margin = (distance_transform_edt(road) - distance_transform_edt(~road)) * scale

    def query(self, positions: np.ndarray, *, centre_distance: np.ndarray | None = None) -> RoadQuery:
        local = super().query(positions, centre_distance=centre_distance)
        grid = ((positions - self.origin) / self.protocol.resolution).T
        grip = map_coordinates(self.full_road, grid, order=1, mode="constant", cval=0.)
        margin = map_coordinates(self.full_margin, grid, order=1, mode="constant", cval=-100.)
        return RoadQuery(np.where(local.visible, local.grip, grip),
                         np.where(local.visible, local.margin, margin),
                         np.ones(len(positions), dtype=bool))


class GlobalPhysicsSearch(PhysicsSearch):
    method = "global_physics_search"
    allowed_views = ("global",)

    def _world(self, observation: Observation) -> FullRoadWorld:
        return FullRoadWorld(observation, self.protocol, self.config.half_width_prior)

    def configuration(self) -> dict:
        return {**super().configuration(), "road_query": FullRoadWorld.version,
                "road_prior": "full_observed_geometry; no unseen continuation",
                "path": "ordered_full_loop_resampled_at_sensor_resolution; replaces_local_row_reconstruction",
                "geometry_access": "privileged_global; no raw vehicle state",
                "vehicle_state": "same_observed_ego_and_estimated_wheels_as_local",
                "unknown_space_penalty": "same_weight; no_unknown_road_in_full_map",
                "local_raster": "unchanged_inside_original_sensor_circle"}
