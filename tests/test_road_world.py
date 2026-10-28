"""Synthetic road semantics; no real track or privileged ranking outputs."""
import numpy as np
import pytest

from cars.env.protocol import LocalSensor, ProtocolConfig
from cars.planning.world import LocalRoadWorld


@pytest.mark.parametrize("angle", [0., np.pi / 4, np.pi / 2])
def test_unknown_road_and_grass_agree_across_raster_edges(angle):
    protocol = ProtocolConfig()
    sensor = LocalSensor(protocol)
    tangent = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-np.sin(angle), np.cos(angle)])
    path = np.arange(-40., 101.)[:, None] * tangent
    road = (np.abs(sensor.points @ normal) < 4) & sensor.visible
    observation = {"road": road.astype(float), "visible": sensor.visible.astype(float)}
    world = LocalRoadWorld(observation, protocol, path, half_width=4.)
    points = np.array([35 * tangent, 45 * tangent, 35 * tangent + 8 * normal,
                       45 * tangent + 8 * normal, np.zeros(2), 10 * normal])
    result = world.query(points)
    np.testing.assert_array_equal(result.visible, [False, False, False, False, True, True])
    np.testing.assert_array_equal(result.grip, [1, 1, 0, 0, 1, 0])
    np.testing.assert_allclose(result.margin[:4], [4, 4, -4, -4], atol=1e-8)
    assert result.margin[4] > 0 and result.margin[5] < 0
    # Unknown cells may not leak into visible interpolation or the road prior.
    observation["road"][~sensor.visible] = 999
    altered = LocalRoadWorld(observation, protocol, path, half_width=4.).query(points)
    np.testing.assert_array_equal(result.grip, altered.grip)
    np.testing.assert_array_equal(result.margin, altered.margin)


def test_visibility_boundary_normalizes_observed_occupancy():
    protocol = ProtocolConfig()
    sensor = LocalSensor(protocol)
    path = np.stack((np.arange(-40., 101.), np.zeros(141)), axis=-1)
    observation = {"road": ((abs(sensor.points[..., 1]) < 4) & sensor.visible).astype(float),
                   "visible": sensor.visible.astype(float)}
    world = LocalRoadWorld(observation, protocol, path, half_width=4.)
    result = world.query(np.array([[29.8, .2], [30.2, .2], [35., 10.]]))
    np.testing.assert_array_equal(result.visible, [True, False, False])
    np.testing.assert_allclose(result.grip, [1, 1, 0])
    assert result.margin[0] > 0 and result.margin[1] > 0 and result.margin[2] < 0
