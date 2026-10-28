"""Full-map diagnostics are explicit; local input and actual physics stay unchanged."""
import numpy as np
import pytest

from cars.env.protocol import DrivingEnv, ProtocolConfig
from cars.planning.global_search import FullRoadWorld, GlobalPhysicsSearch
from cars.planning.physics_search import PhysicsSearch
from cars.planning.search import SearchConfig
from cars.planning.world import LocalRoadWorld


def test_global_protocol_is_separate_and_cannot_collect_training():
    local, global_config = ProtocolConfig(), ProtocolConfig(view="global")
    assert local.identifier != global_config.identifier
    with pytest.raises(ValueError, match="development-only"):
        DrivingEnv(global_config, "train")
    with pytest.raises(ValueError, match="requires"):
        PhysicsSearch(global_config)
    with pytest.raises(ValueError, match="requires"):
        GlobalPhysicsSearch(local)
    a, b = DrivingEnv(local, "development"), DrivingEnv(global_config, "development")
    try:
        obs, _ = a.reset()
        full, info = b.reset()
        assert info == {} and b.observation_space.contains(full)
        assert set(full) - set(obs) == {"map_centre", "map_left", "map_right"}
        for _ in range(3):
            for key in obs:
                np.testing.assert_array_equal(obs[key], full[key])
            action = np.array([.05, .3, 0.], np.float32)
            obs, r, d, t, i = a.step(action)
            full, rr, dd, tt, ii = b.step(action)
            assert (r, d, t, i) == (rr, dd, tt, ii)
    finally:
        a.close()
        b.close()


def test_global_query_preserves_local_sensor_and_adds_known_geometry():
    config = ProtocolConfig(view="global")
    env = DrivingEnv(config, "development")
    try:
        obs, _ = env.reset()
        full = FullRoadWorld(obs, config, 6.67)
        local = LocalRoadWorld(obs, config, full.path, 6.67)
        positions = np.array([[0., 0.], [10., 0.], [0., 15.], [-10., 1.]])
        a, b = full.query(positions), local.query(positions)
        np.testing.assert_array_equal(a.grip, b.grip)
        np.testing.assert_array_equal(a.margin, b.margin)
        assert a.visible.all()
        # Actual supplied road vertices far from the circle are known road;
        # arbitrarily far outside the full track is known grass, not a prior.
        points = obs["map_centre"][np.linalg.norm(obs["map_centre"], axis=1) > 60][::10]
        assert (full.query(points).grip > .9).all()
        outside = full.query(np.array([[10000., 10000.]]))
        assert outside.visible[0] and outside.grip[0] == 0 and outside.margin[0] < 0
        policy = GlobalPhysicsSearch(config, SearchConfig(horizon=2, speeds=(30., 50.), steering_gains=(1.2,)))
        action = policy.act(obs)
        first = policy.model.first_ego.copy()
        policy.reset()
        np.testing.assert_array_equal(action, policy.act(obs))
        np.testing.assert_array_equal(first, policy.model.first_ego)
        assert not policy.configuration()["simulator"]["evaluation_environment_state_access"]
        assert policy.diagnostics["internal_physics_frames"] == 16
        policy.reset()
    finally:
        env.close()
