"""True-reference continuation changes the tail, not the prefix or road world."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from cars.env.protocol import DrivingEnv, LocalSensor, ProtocolConfig
from cars.planning.intervention import InterventionConfig, intervention_worlds
from cars.planning.local import extend
from cars.planning.policy_search import imagined_observations
from cars.planning.reference_extension import extend_true_reference
from cars.planning.visible_reference import visible_reference


def test_true_extension_preserves_prefix_and_matches_straight_negative_control():
    loop = np.array([[-200., 0.], [200., 0.], [200., 200.], [-200., 200.]])
    visible = visible_reference(loop, 30., 1.)
    result = extend_true_reference(visible, loop)
    np.testing.assert_array_equal(result, extend(visible))
    np.testing.assert_array_equal(result[:len(visible)], visible)
    assert result.shape == (len(visible) + 100, 2)


def test_curved_extension_arc_direction_vertex_wrap_and_map_origin():
    loop = np.array([[-40., 0.], [40., 0.], [40., 80.], [-40., 80.]])
    visible = visible_reference(loop, 30., 1.)
    result = extend_true_reference(visible, loop)
    expected_tail = np.r_[np.column_stack((np.arange(31., 41.), np.zeros(10))),
        np.column_stack((np.full(80, 40.), np.arange(1., 81.))),
        np.column_stack((np.arange(39., 29., -1.), np.full(10, 80.)))]
    np.testing.assert_array_equal(result[len(visible):], expected_tail)
    for shift in range(len(loop)):
        np.testing.assert_array_equal(extend_true_reference(visible, np.roll(loop, shift, axis=0)), result)
    duplicated = np.insert(loop, 1, loop[1], axis=0)
    np.testing.assert_array_equal(extend_true_reference(visible, duplicated), result)
    at_vertex = np.array([[39., 0.], [40., 0.]])
    np.testing.assert_array_equal(extend_true_reference(at_vertex, loop, 3)[2:], [[40., 1.], [40., 2.], [40., 3.]])
    # Crossing the final array segment and wrapping is continuous and directed.
    wrapped = extend_true_reference(np.array([[-40., 2.], [-40., 1.]]), loop, 3)
    np.testing.assert_array_equal(wrapped[-3:], [[-40., 0.], [-39., 0.], [-38., 0.]])
    # Removing zero-length segments and multiple cycles cannot cause a hang.
    tiny = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    repeated = extend_true_reference(np.array([[0., .5], [0., 0.]]), tiny, 9)
    np.testing.assert_array_equal(repeated[2], repeated[6])
    with pytest.raises(ValueError, match="does not lie"):
        extend_true_reference(visible + 1000, loop)
    with pytest.raises(ValueError, match="direction"):
        extend_true_reference(np.zeros((2, 2)), loop)


def test_true_tail_keeps_visible_geometry_world_queries_and_imagined_inputs_identical():
    config = ProtocolConfig(view="global")
    env = DrivingEnv(config, "development")
    try:
        obs, _ = env.reset()
        old = intervention_worlds(obs, config, InterventionConfig("oracle_visible", "oracle"), 6.67)
        new = intervention_worlds(obs, config, InterventionConfig("oracle_continued", "oracle"), 6.67)
        prefix = visible_reference(obs["map_centre"], config.radius, config.resolution)
        assert not old[3] and not new[3] and old[2].shape == new[2].shape
        np.testing.assert_array_equal(old[2][:len(prefix)], new[2][:len(prefix)])
        assert np.max(np.abs(old[2][len(prefix):] - new[2][len(prefix):])) > 1
        np.testing.assert_array_equal(old[1].path, new[1].path)
        positions = np.r_[obs["map_centre"], [[0., 0.], [500., 500.]]]
        a, b = old[1].query(positions), new[1].query(positions)
        for field in ("grip", "margin", "visible"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        sensor = LocalSensor(config)
        pose = np.array([[0., 0., 0.], [40., 10., .3]])
        ego, previous = np.zeros((2, 3)), np.zeros((2, 3))
        a = imagined_observations(old[1], sensor, cKDTree(old[1].path), pose, ego, previous, "sparse")
        b = imagined_observations(new[1], sensor, cKDTree(new[1].path), pose, ego, previous, "sparse")
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    finally:
        env.close()
