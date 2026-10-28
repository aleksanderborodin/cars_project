"""Synthetic checks of programmed physics; no hidden evaluation state."""
from Box2D import b2World
from gymnasium.envs.box2d.car_dynamics import Car
import numpy as np

from cars.env.protocol import LocalSensor, ProtocolConfig
from cars.models.physics import PhysicsDynamics, RoadTile
from cars.planning.physics_search import PhysicsSearch
from cars.planning.search import SearchConfig
from cars.planning.world import LocalRoadWorld


def scene(speed=0.):
    config = ProtocolConfig()
    sensor = LocalSensor(config)
    observation = {"road": ((abs(sensor.points[..., 1]) < 6) & sensor.visible).astype(float),
                   "visible": sensor.visible.astype(float), "ego": np.array([speed, 0., 0.]),
                   "previous_action": np.zeros(3, np.float32)}
    path = np.stack((np.arange(-30., 131.), np.zeros(161)), axis=-1)
    return config, observation, LocalRoadWorld(observation, config, path, 6.)


def test_synthetic_road_rollout_matches_upstream_car_for_identical_initial_state():
    _, obs, road = scene()
    model = PhysicsDynamics()
    model.begin(obs, road, 1)
    reference = Car(b2World(gravity=(0, 0)), 0., 0., 0.)
    for wheel in reference.wheels:
        wheel.tiles = {RoadTile()}
    pose, ego = np.zeros((1, 3)), obs["ego"][None]
    action = np.array([[0., .5, 0.]])
    try:
        for _ in range(10):
            pose, ego = model.advance(pose, ego, action, action, np.ones(1))
            for _ in range(4):
                reference.steer(0.)
                reference.gas(.5)
                reference.brake(0.)
                reference.step(.02)
                reference.world.Step(.02, 180, 60)
            x, y = reference.hull.position
            vx, vy = reference.hull.linearVelocity
            np.testing.assert_array_equal(pose[0], [y, -x, reference.hull.angle])
            c, s = np.cos(reference.hull.angle), np.sin(reference.hull.angle)
            np.testing.assert_array_equal(ego[0], [-s * vx + c * vy, -c * vx - s * vy,
                                                  reference.hull.angularVelocity])
    finally:
        reference.destroy()
        model.reset()


def test_physics_candidates_are_independent_and_commit_only_first_transition():
    _, obs, road = scene(20.)
    model = PhysicsDynamics()
    model.begin(obs, road, 3)
    pose, ego = np.zeros((3, 3)), np.repeat(obs["ego"][None], 3, axis=0)
    action = np.array([[-.15, .3, 0.], [.15, .3, 0.], [-.15, .3, 0.]])
    for _ in range(8):
        pose, ego = model.advance(pose, ego, action, action, np.ones(3))
    np.testing.assert_array_equal(pose[0], pose[2])
    np.testing.assert_array_equal(ego[0], ego[2])
    assert pose[0, 1] > 0 and pose[1, 1] < 0
    first = model.first_estimates[0].copy()
    model.commit(0)
    np.testing.assert_array_equal(model.estimate, first)
    assert not np.allclose(first[:, 0], [w.omega for w in model.cars[0].wheels])
    model.reset()
    assert model.estimate is None and model.cars == []


def test_physics_search_resets_and_reports_real_physics_work():
    config, obs, _ = scene()
    policy = PhysicsSearch(config, SearchConfig(horizon=3, speeds=(20., 40.), steering_gains=(1.,)))
    first = policy.act(obs)
    costs = policy.candidate_costs.copy()
    assert policy.diagnostics["model_transitions"] == 6
    assert policy.diagnostics["internal_physics_frames"] == 24
    assert len(policy.diagnostics["predicted_next_ego"]) == 3
    assert not policy.configuration()["simulator"]["evaluation_environment_state_access"]
    policy.reset()
    np.testing.assert_array_equal(policy.act(obs), first)
    np.testing.assert_array_equal(policy.candidate_costs, costs)
    policy.reset()
