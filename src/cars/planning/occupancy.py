"""Binary road queries with the same semantics as LocalRoadWorld.query.

Imagined policy observations need occupancy, not a signed margin. Avoid computing
unused margins and nearest-center distances for observed or definitely far points.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from cars.planning.world import LocalRoadWorld


def binary_road(world: LocalRoadWorld, tree: cKDTree, positions: np.ndarray) -> np.ndarray:
    if type(world) is not LocalRoadWorld:
        # A geometry subclass may implement different outside-road semantics.
        return world.query(positions, centre_distance=tree.query(positions)[0]).grip >= .5
    radius, resolution = world.protocol.radius, world.protocol.resolution
    inside = np.sum(positions**2, axis=1) <= radius**2
    observed = np.zeros(len(positions), dtype=bool)
    road = np.zeros(len(positions), dtype=bool)
    index = np.flatnonzero(inside)
    grid = ((positions[index] + radius) / resolution).T
    weights = map_coordinates(world.weights, grid, order=1, mode="constant", cval=0.)
    occupancy = map_coordinates(world.road, grid, order=1, mode="constant", cval=0.)
    grip = np.divide(occupancy, weights, out=np.zeros_like(weights), where=weights > 0)
    observed[index] = weights > 0
    road[index] = grip >= .5
    # Unknown geometry is the same distance-to-sampled-path constant-width prior.
    # Expanded bounds only reject points that cannot be within half_width.
    low = np.nextafter(world.path.min(axis=0) - world.half_width, -np.inf)
    high = np.nextafter(world.path.max(axis=0) + world.half_width, np.inf)
    candidates = np.flatnonzero(~observed & np.all((positions >= low) & (positions <= high), axis=1))
    distances = tree.query(positions[candidates], distance_upper_bound=np.nextafter(world.half_width, np.inf))[0]
    road[candidates] = distances <= world.half_width
    return road
