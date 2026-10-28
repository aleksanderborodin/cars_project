"""Frozen policy control, imagined observations and recurrent branch isolation."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from cars.env.protocol import DrivingEnv, LocalSensor, ProtocolConfig
from cars.planning.policy_search import PolicyPhysicsSearch, PolicySearchConfig, RandomPolicyVariation, imagined_observations
from cars.planning.world import LocalRoadWorld
from cars.research.driver_config import DriverTrainingConfig
from cars.research.driver_training import train_driver
from cars.research.recurrent import RecurrentDriver


@pytest.fixture(scope="module")
def actor(tmp_path_factory):
    config = DriverTrainingConfig(workers=1, vector_backend="dummy", total_native_frames=256,
        checkpoints=(), n_steps=16, batch_size=16, n_epochs=1, lstm_hidden_size=16, features_dim=32)
    return train_driver("actor", config, stop_at=64, runs_dir=tmp_path_factory.mktemp("actor"))


def test_zero_candidate_and_imagined_history_never_change_real_memory(actor):
    protocol = ProtocolConfig()
    base = RecurrentDriver(actor)
    policy = PolicyPhysicsSearch(actor, protocol, PolicySearchConfig(candidates=1, horizon=4))
    env = DrivingEnv(protocol, "train", seed=473)
    try:
        observation, _ = env.reset()
        first_observation = {key: value.copy() for key, value in observation.items()}
        first = None
        for _ in range(4):
            expected, actual = base.act(observation), policy.act(observation)
            np.testing.assert_array_equal(actual, expected)
            for direct, imagined in zip(base.memory, policy.actor.memory, strict=True):
                np.testing.assert_array_equal(direct, imagined)
            assert policy.diagnostics["internal_physics_frames"] == 16
            assert policy.diagnostics["actor_forward_examples"] == 4
            assert not policy.diagnostics["overrode_base"]
            first = actual if first is None else first
            observation, _, _, _, _ = env.step(actual)
        policy.reset()
        np.testing.assert_array_equal(policy.act(first_observation), first)
        assert policy.candidate_actions.shape == (1, 4, 3)
    finally:
        policy.reset()
        env.close()


def test_random_control_and_search_share_proposal_bank_and_reset(actor):
    protocol = ProtocolConfig()
    config = PolicySearchConfig(candidates=3, horizon=2, planning_seed=19)
    search, random = PolicyPhysicsSearch(actor, protocol, config), RandomPolicyVariation(actor, protocol, config)
    env = DrivingEnv(protocol, "train", seed=474)
    try:
        observation, _ = env.reset()
        base_a, residual_a = search._propose(observation)
        base_b, residual_b = random._propose(observation)
        np.testing.assert_array_equal(base_a, base_b)
        np.testing.assert_array_equal(residual_a, residual_b)
        assert not residual_a[0].any()
        search.reset()
        random.reset()
        action = random.act(observation)
        random.reset()
        np.testing.assert_array_equal(random.act(observation), action)
        assert env.action_space.contains(action) and random.diagnostics["model_transitions"] == 0
        result = search.act(observation)
        assert env.action_space.contains(result) and search.candidate_actions.shape == (3, 2, 3)
        assert search.diagnostics["internal_physics_frames"] == 24
        with pytest.raises(ValueError, match="same declared local"):
            PolicyPhysicsSearch(actor, replace(protocol, view="native"), config)
    finally:
        search.reset()
        env.close()


def test_imagined_sensor_has_same_root_footprint_and_only_permitted_fields():
    config = ProtocolConfig()
    sensor = LocalSensor(config)
    observation = {"road": ((abs(sensor.points[..., 1]) < 6) & sensor.visible).astype(np.float32),
        "visible": sensor.visible.astype(np.float32), "ego": np.array([20., 0., 0.], np.float32),
        "previous_action": np.array([0., .2, 0.], np.float32)}
    path = np.stack((np.arange(-30., 131.), np.zeros(161)), axis=-1)
    world = LocalRoadWorld(observation, config, path, 6.)
    imagined = imagined_observations(world, sensor, cKDTree(path), np.zeros((1, 3)),
                                     observation["ego"][None], observation["previous_action"][None])
    assert set(imagined) == set(observation)
    for key in observation:
        np.testing.assert_array_equal(imagined[key][0], observation[key])
