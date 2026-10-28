"""Preserve binary occupancy at observed boundaries and prior-width thresholds."""
import numpy as np
from scipy.spatial import cKDTree

from cars.env.protocol import LocalSensor, ProtocolConfig
from cars.planning.occupancy import binary_road
from cars.planning.policy_search import imagined_observations
from cars.planning.world import LocalRoadWorld


def test_sparse_occupancy_matches_reference_with_holes_and_thresholds():
    protocol = ProtocolConfig()
    sensor = LocalSensor(protocol)
    rng = np.random.default_rng(123)
    visible = sensor.visible & (rng.random(sensor.visible.shape) > .2)
    road = visible & (rng.random(visible.shape) > .6)
    path = np.array([[40., 0.], [50., 3.], [60., 5.]])
    world = LocalRoadWorld({"road": road, "visible": visible}, protocol, path, 6.67)
    tree = cKDTree(path)
    width = world.half_width
    boundary = np.array([[40., width], [40., np.nextafter(width, 0)],
        [40., np.nextafter(width, np.inf)], [30., 0.], [0., 30.], [30., .1],
        [0., 0.], [-30., 0.], [60. + width, 5.]])
    positions = np.concatenate((rng.uniform(-70, 100, (10000, 2)), boundary, sensor.points.reshape(-1, 2)))
    expected = world.query(positions, centre_distance=tree.query(positions)[0]).grip >= .5
    np.testing.assert_array_equal(binary_road(world, tree, positions), expected)
    pose = np.column_stack((rng.uniform(-60, 100, (21, 2)), rng.uniform(-np.pi, np.pi, 21)))
    ego, previous = rng.normal(size=(21, 3)), np.zeros((21, 3))
    a = imagined_observations(world, sensor, tree, pose, ego, previous, "reference")
    b = imagined_observations(world, sensor, tree, pose, ego, previous, "sparse")
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_sparse_occupancy_respects_subclass_query():
    class AllRoad(LocalRoadWorld):
        def query(self, positions, *, centre_distance=None):
            result = super().query(positions, centre_distance=centre_distance)
            result.grip[:] = 1
            return result
    protocol = ProtocolConfig()
    sensor = LocalSensor(protocol)
    world = AllRoad({"road": np.zeros_like(sensor.visible), "visible": sensor.visible}, protocol,
                    np.array([[0., 0.], [1., 0.]]), 1.)
    assert binary_road(world, cKDTree(world.path), np.array([[100., 100.]])).all()
