"""Meaningful checks for schedule continuity, checkpoint state and normalization."""
from dataclasses import replace
import json
import pickle

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from sb3_contrib import RecurrentPPO

from cars.env.protocol import DrivingEnv, ProtocolConfig
from cars.research.driver_config import DriverTrainingConfig, NativeSchedule
from cars.research.driver_training import train_driver
from cars.research.driver_encoder import DeterministicAveragePool
from cars.research.recurrent import RecurrentDriver


def test_native_schedule_does_not_restart_at_chunk_boundary():
    schedule = NativeSchedule(.001, 1000, 250)
    assert schedule(1.) == schedule(0.) == .00075
    schedule.frames = 1000
    assert schedule(.5) == 0
    with pytest.raises(ValueError, match="Batch size"):
        DriverTrainingConfig(workers=1, n_steps=8, batch_size=16)
    for value in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="target_kl"):
            DriverTrainingConfig(target_kl=value)
    for value in (0, -1, 1.1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="normalization_gamma"):
            DriverTrainingConfig(normalization_gamma=value)
    assert DriverTrainingConfig(gamma=.8).resolved_normalization_gamma == .8


def test_policy_discount_can_change_without_normalization_discount(tmp_path):
    config = DriverTrainingConfig(workers=1, vector_backend="dummy", total_native_frames=128,
        checkpoints=(), n_steps=8, batch_size=8, n_epochs=1, lstm_hidden_size=16, features_dim=32,
        gamma=.99**4, normalization_gamma=.99)
    root = train_driver("separate_discounts", config, stop_at=32, runs_dir=tmp_path)
    model = RecurrentPPO.load(root / "model.zip", device="cpu")
    assert model.gamma == .99**4
    with (root / "normalization.pkl").open("rb") as file:
        normalizer = pickle.load(file)
    assert normalizer.gamma == .99
    assert not normalizer.norm_obs and normalizer.norm_reward
    resumed = train_driver("continued_discounts", config, stop_at=64, resume=root, runs_dir=tmp_path)
    assert json.loads((resumed / "training_state.json").read_text())["decisions"] == 16


@pytest.mark.parametrize("size", [9, 15])
def test_deterministic_pool_matches_adaptive_values_and_gradients(size):
    rng = torch.Generator().manual_seed(91)
    x = torch.randn(2, 3, size, size, generator=rng, requires_grad=True)
    weights = torch.randn(2, 3, 4, 4, generator=rng)
    expected = F.adaptive_avg_pool2d(x, (4, 4))
    (expected * weights).sum().backward()
    prior = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            observed = x.detach().to(device).requires_grad_(True)
            actual = DeterministicAveragePool()(observed)
            (actual * weights.to(device)).sum().backward()
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(observed.grad.cpu(), x.grad, rtol=1e-5, atol=1e-6)
    finally:
        torch.use_deterministic_algorithms(prior)


def test_configured_training_checkpoints_and_resume(tmp_path):
    config = DriverTrainingConfig(workers=1, vector_backend="dummy", total_native_frames=512,
        checkpoints=(64, 128, 256), n_steps=16, batch_size=16, n_epochs=1, lstm_hidden_size=16,
        features_dim=32)
    root = train_driver("first", config, stop_at=128, runs_dir=tmp_path)
    state = json.loads((root / "training_state.json").read_text())
    assert state["native_frames"] == 128 and state["saved_after_optimizer_update"]
    assert state["learning_rate"] == config.learning_rate * .75
    assert len(list((root / "checkpoints").iterdir())) == 2
    checkpoint = root / "checkpoints/frames_000000128"
    assert (checkpoint / "config.json").exists() and (checkpoint / "source/uv.lock").exists()
    model = RecurrentPPO.load(root / "model.zip", device="cpu")
    saved_steps = max(float(item["step"]) for item in model.policy.optimizer.state.values() if "step" in item)
    normalizer_before = (root / "normalization.pkl").read_bytes()
    with (root / "normalization.pkl").open("rb") as file:
        normalization = pickle.load(file)
    assert not normalization.norm_obs and normalization.norm_reward and normalization.ret_rms.count > 1
    env = DrivingEnv(ProtocolConfig(), seed=41)
    try:
        obs, _ = env.reset()
        driver = RecurrentDriver(checkpoint)
        for _ in range(3):
            action = driver.act(obs)
            assert env.action_space.contains(action)
            obs, _, _, _, _ = env.step(action)
        assert driver.configuration()["evaluation_normalization_updates"] is False
    finally:
        env.close()
    assert (root / "normalization.pkl").read_bytes() == normalizer_before
    resumed = train_driver("continued", config, stop_at=256, resume=root, runs_dir=tmp_path)
    final = json.loads((resumed / "training_state.json").read_text())
    assert final["native_frames"] == 256 and final["learning_rate"] == config.learning_rate * .5
    assert final["reward_normalization_count"] > state["reward_normalization_count"]
    restored = RecurrentPPO.load(resumed / "model.zip", device="cpu")
    assert max(float(item["step"]) for item in restored.policy.optimizer.state.values() if "step" in item) > saved_steps
    before = set(json.loads((root / "training_tracks.json").read_text())[0])
    after = set(json.loads((resumed / "training_tracks.json").read_text())[0])
    assert before.isdisjoint(after)
    with pytest.raises(ValueError, match="identical"):
        train_driver("incompatible", replace(config, learning_rate=.003), stop_at=512, resume=root, runs_dir=tmp_path)
    with pytest.raises(ValueError, match="identical"):
        train_driver("changed_kl", replace(config, target_kl=.02), stop_at=512, resume=root, runs_dir=tmp_path)


def test_kl_stop_reduces_optimizer_work_without_changing_initialization(tmp_path):
    config = DriverTrainingConfig(workers=1, vector_backend="dummy", total_native_frames=1024,
        checkpoints=(), n_steps=32, batch_size=16, n_epochs=4, learning_rate=.001,
        lstm_hidden_size=16, features_dim=32)
    paths = [train_driver("unlimited", config, stop_at=128, runs_dir=tmp_path),
             train_driver("limited", replace(config, target_kl=1e-8), stop_at=128, runs_dir=tmp_path)]
    initial = [json.loads((path / "initial_policy.json").read_text()) for path in paths]
    assert initial[0]["parameters_sha256"] == initial[1]["parameters_sha256"]
    rows = [json.loads((path / "learning.jsonl").read_text()) for path in paths]
    assert rows[1]["optimizer_steps_cumulative"] < rows[0]["optimizer_steps_cumulative"]
    assert rows[1]["epochs_entered_this_rollout"] <= rows[0]["epochs_entered_this_rollout"]
    assert RecurrentPPO.load(paths[1] / "model.zip").target_kl == 1e-8
