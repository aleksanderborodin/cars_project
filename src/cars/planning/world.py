"""One local road hypothesis shared by dynamics and candidate scoring."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from cars.env.protocol import Observation, ProtocolConfig


@dataclass(frozen=True)
class RoadQuery:
    grip: np.ndarray
    margin: np.ndarray
    visible: np.ndarray


class LocalRoadWorld:
    """Observed raster inside the circle; constant-width path prior outside.

    Visible occupancy uses bilinear interpolation normalized by visible weight,
    so masked padding is not interpreted as observed grass. Unknown occupancy
    and margin both use distance to the same sampled continuation. Observed
    margins retain the original visibility-aware distance transform; occupancy
    is a center-contact proxy, not a physical tire/friction simulation.
    """
    version = "shared_road_query_v2"

    def __init__(self, observation: Observation, protocol: ProtocolConfig,
                 path: np.ndarray, half_width: float):
        self.protocol, self.path, self.half_width = protocol, path, half_width
        visible = observation["visible"] > .5
        road = (observation["road"] > .5) & visible
        self.weights = visible.astype(float)
        self.road = road.astype(float)
        self.signed = (distance_transform_edt(road | ~visible)
                       - distance_transform_edt(~road & visible)) * protocol.resolution

    def query(self, positions: np.ndarray, *, centre_distance: np.ndarray | None = None) -> RoadQuery:
        if centre_distance is None:
            centre_distance = np.sqrt(np.min(np.sum(
                (positions[:, None, :] - self.path[None])**2, axis=-1), axis=1))
        prior_margin = self.half_width - centre_distance
        grid = ((positions + self.protocol.radius) / self.protocol.resolution).T
        weights = map_coordinates(self.weights, grid, order=1, mode="constant", cval=0.)
        visible = (np.sum(positions**2, axis=1) <= self.protocol.radius**2) & (weights > 0)
        occupancy = map_coordinates(self.road, grid, order=1, mode="constant", cval=0.)
        observed_grip = np.divide(occupancy, weights, out=np.zeros_like(weights), where=weights > 0)
        observed_margin = map_coordinates(self.signed, grid, order=1, mode="constant", cval=0.)
        return RoadQuery(
            grip=np.where(visible, observed_grip, (prior_margin >= 0).astype(float)),
            margin=np.where(visible, observed_margin, prior_margin),
            visible=visible,
        )
