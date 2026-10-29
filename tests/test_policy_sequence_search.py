"""Recurrent isolation, actual rollout work and committing an earlier-batch winner."""
from dataclasses import replace

import numpy as np
import pytest

from cars.env.protocol import DrivingEnv, ProtocolConfig
from cars.models.physics import PhysicsDynamics
from cars.planning.policy_search import PolicySearchConfig
from cars.planning.policy_sequence_search import PolicySequenceSearch
from cars.planning.sequence_optimizer import SequenceOptimizerConfig
from cars.research.driver_config import DriverTrainingConfig
from cars.research.driver_training import train_driver
from cars.research.recurrent import RecurrentDriver


@pytest.fixture(scope="module")
def actor(tmp_path_factory):
    config = DriverTrainingConfig(workers=1, vector_backend="dummy", total_native_frames=256,
        checkpoints=(), n_steps=16, batch_size=16, n_epochs=1, lstm_hidden_size=16, features_dim=32)
    return train_driver("sequence_actor", config, stop_at=64, runs_dir=tmp_path_factory.mktemp("sequence_actor"))


@pytest.mark.parametrize("mode", ["shooting", "icem"])
def test_real_memory_root_state_work_and_reset(actor, mode, monkeypatch):
    protocol = ProtocolConfig()
    config = PolicySearchConfig(candidates=21, horizon=3, observation_renderer="sparse")
    driver = PolicySequenceSearch(actor, protocol, config, SequenceOptimizerConfig(mode=mode))
    direct = RecurrentDriver(actor)
    env = DrivingEnv(protocol, "train", seed=857)
    roots = []
    begin = driver.model.begin

    def observed_begin(observation, world, count):
        roots.append(None if driver.model.estimate is None else driver.model.estimate.copy())
        return begin(observation, world, count)

    monkeypatch.setattr(driver.model, "begin", observed_begin)
    predict = driver.actor.model.predict

    def local_only(observation, *args, **kwargs):
        assert set(observation) == {"road", "visible", "ego", "previous_action"}
        return predict(observation, *args, **kwargs)

    monkeypatch.setattr(driver.actor.model, "predict", local_only)
    try:
        obs, _ = env.reset()
        original = {key: value.copy() for key, value in obs.items()}
        for step in range(3):
            direct.act(obs)
            before = None if driver.model.estimate is None else driver.model.estimate.copy()
            action = driver.act(obs)
            for a, b in zip(driver.actor.memory, direct.memory, strict=True):
                np.testing.assert_array_equal(a, b)
            for root in roots[-2:]:
                if before is None:
                    assert root is None
                else:
                    np.testing.assert_array_equal(root, before)
            assert env.action_space.contains(action)
            d = driver.diagnostics
            assert d["batch_sizes"] == [13, 8] and d["model_transitions"] == 63
            assert d["internal_physics_frames"] == 252 and d["actor_forward_examples"] == 43
            assert d["actor_forward_calls"] == 5
            assert d["shifted_elites_reevaluated"] == int(mode == "icem" and step > 0)
            np.testing.assert_array_equal(driver.model.estimate, driver.candidate_first_estimates[d["selected_candidate"]])
            if step == 0:
                first_action, first_bank = action.copy(), driver.candidate_actions.copy()
            obs, *_ = env.step(action)
        driver.reset()
        np.testing.assert_array_equal(driver.act(original), first_action)
        np.testing.assert_array_equal(driver.candidate_actions, first_bank)
        with pytest.raises(ValueError, match="same declared local"):
            PolicySequenceSearch(actor, replace(protocol, view="global"), config, SequenceOptimizerConfig())
    finally:
        driver.reset()
        env.close()


def test_earlier_winner_commits_owned_first_snapshot_not_last_batch(actor, monkeypatch):
    import cars.planning.policy_sequence_search as module
    driver = PolicySequenceSearch(actor, ProtocolConfig(),
        PolicySearchConfig(candidates=21, horizon=3, planning_seed=7, observation_renderer="sparse"),
        SequenceOptimizerConfig(mode="icem"))
    real_evaluate = module.evaluate_feedback
    calls = []

    def synthetic_costs(*args, **kwargs):
        batch = real_evaluate(*args, **kwargs)
        calls.append(batch)
        batch.costs[:] = 100. + np.arange(len(batch.costs))
        if len(calls) == 1:
            batch.costs[5] = -100.
        return batch

    monkeypatch.setattr(module, "evaluate_feedback", synthetic_costs)
    env = DrivingEnv(ProtocolConfig(), "train", seed=858)
    try:
        obs, _ = env.reset()
        action = driver.act(obs)
        assert driver.diagnostics["selected_candidate"] == 5 and driver.diagnostics["selected_batch"] == 0
        np.testing.assert_array_equal(action, calls[0].actions[5, 0])
        np.testing.assert_array_equal(driver.model.estimate, calls[0].first_estimates[5])
        assert not np.array_equal(driver.model.estimate, calls[1].first_estimates[5])
        preserved = driver.model.estimate.copy()
        calls[0].first_estimates[:] = -999
        np.testing.assert_array_equal(driver.model.estimate, preserved)
    finally:
        driver.reset()
        env.close()


def test_commit_estimate_owns_input_and_rejects_bad_shapes():
    model = PhysicsDynamics()
    estimate = np.ones((4, 4))
    model.commit_estimate(estimate)
    estimate[:] = 9
    np.testing.assert_array_equal(model.estimate, np.ones((4, 4)))
    for invalid in (np.zeros((3, 4)), np.full((4, 4), np.nan)):
        with pytest.raises(ValueError):
            model.commit_estimate(invalid)
