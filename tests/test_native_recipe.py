"""Native recipe parity, information boundary, memory and checkpoint integration."""
from dataclasses import replace
import json
import pickle

import gymnasium as gym
import numpy as np
import pytest
from sb3_contrib import RecurrentPPO

from cars.env.native_recipe import NativeRecipeProtocol, native_environment
from cars.research.native_config import NativeTrainingConfig
from cars.research.native_training import train_native
from cars.research.native_vector import native_vector


def test_native_pixels_reward_and_action_cadence_match_official(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    actual = native_environment(seed=7)
    expected = gym.wrappers.GrayscaleObservation(gym.wrappers.ResizeObservation(
        gym.make("CarRacing-v3", continuous=True, lap_complete_percent=.95), (64, 64)), keep_dim=True)
    try:
        obs, info = actual.reset()
        seed = actual.get_wrapper_attr("track_seeds")[-1]  # evaluator/test only
        reference, _ = expected.reset(seed=seed)
        np.testing.assert_array_equal(obs, reference)
        assert info == {} and obs.shape == (64, 64, 1) and obs.dtype == np.uint8
        assert isinstance(actual.observation_space, gym.spaces.Box)
        for i in range(12):
            action = np.array([.1 if i > 5 else 0, .6, 0], dtype=np.float32)
            obs, reward, term, trunc, info = actual.step(action)
            reference, reference_reward, reference_term, reference_trunc, _ = expected.step(action)
            np.testing.assert_array_equal(obs, reference)
            assert (reward, term, trunc) == (reference_reward, reference_term, reference_trunc)
            assert info == {"native_frames": 1}
    finally:
        actual.close()
        expected.close()
    with pytest.raises(ValueError):
        NativeRecipeProtocol(action_repeat=4)
    with pytest.raises(ValueError, match="sealed"):
        native_environment("test")


@pytest.mark.parametrize("backend", ["dummy", "subproc"])
def test_native_vector_preserves_frames_workers_and_autoreset(backend, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    config = NativeTrainingConfig(workers=2, vector_backend=backend)
    raw, norm, vector = native_vector(config)
    try:
        first = vector.reset()
        assert first.shape == (2, 2, 64, 64) and first.dtype == np.uint8
        assert not first[:, 0].any() and first[:, 1].any()
        action = np.array([[0, .5, 0], [.1, .5, 0]], dtype=np.float32)
        second, _, _, _ = vector.step(action)
        np.testing.assert_array_equal(second[:, 0], first[:, 1])
        assert norm.ret_rms.count == pytest.approx(2.0001)
        seeds = raw.get_attr("track_seeds")
        assert seeds[0][0] != seeds[1][0]
        if backend == "dummy":
            # Explicit unit-test time limit only; production limit remains 1000.
            raw.envs[0].unwrapped._sim.env._max_episode_steps = 2
            third, _, done, info = vector.step(action)
            assert done.tolist() == [True, False]
            assert not third[0, 0].any()
            np.testing.assert_array_equal(third[1, 0], second[1, 1])
            np.testing.assert_array_equal(info[0]["terminal_observation"][0], second[0, 1])
            assert len(raw.get_attr("track_seeds")[0]) == 2
    finally:
        vector.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA unavailable"))])
def test_native_training_checkpoint_resume_and_frozen_eval(tmp_path, monkeypatch, device):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    config = NativeTrainingConfig(workers=1, total_decisions=32, checkpoints=(8, 16),
        n_steps=8, batch_size=8, n_epochs=1, lstm_hidden_size=16, device=device)
    first = train_native("first", config, stop_at=8, runs_dir=tmp_path)
    model = RecurrentPPO.load(first / "model.zip", device="cpu")
    assert model.policy.net_arch == [] and model.policy.features_dim == 512
    assert model.observation_space.shape == (2, 64, 64)
    assert not model.policy.enable_critic_lstm
    checkpoint = first / "checkpoints/frames_000000008"
    assert json.loads((checkpoint / "config.json").read_text())["method"] == "native_checkpoint"
    second = train_native("second", config, stop_at=16, resume=checkpoint, runs_dir=tmp_path)
    state = json.loads((second / "training_state.json").read_text())
    assert state["native_frames"] == state["decisions"] == 16
    assert state["learning_rate"] == config.learning_rate / 2
    assert state["saved_after_optimizer_update"] and state["updates"] == 2
    with (second / "normalization.pkl").open("rb") as file:
        stats = pickle.load(file).ret_rms
    assert stats.count == pytest.approx(16.0001)
    old_tracks = json.loads((first / "training_tracks.json").read_text())[0]
    new_tracks = json.loads((second / "training_tracks.json").read_text())[0]
    assert set(old_tracks).isdisjoint(new_tracks)
    raw, norm, vector = native_vector(replace(config, workers=1),
        normalization=second / "normalization.pkl", training=False)
    try:
        before = (norm.ret_rms.mean.copy(), norm.ret_rms.var.copy(), norm.ret_rms.count)
        vector.reset()
        _, reward, _, _ = vector.step(np.zeros((1, 3), np.float32))
        assert (norm.ret_rms.mean, norm.ret_rms.var, norm.ret_rms.count) == before
        assert not norm.norm_reward and not norm.training
        assert reward.dtype == np.float32
    finally:
        vector.close()
    cost = json.loads((second / "cost.json").read_text())
    assert cost["native_frames"] == 8 and cost["reload_check"]["native_frames"] == 2
    assert cost["reload_check"]["same_device_exact"]
    assert cost["reload_check"]["cpu_repeated_reload_exact"]
    # Exercise the full evaluator (including terminal metadata and the next
    # episode's recurrent/frame reset) on a declared test-only shortened episode.
    from cars.research import native_evaluation
    original_vector = native_evaluation.native_vector

    def short_vector(*args, **kwargs):
        raw, normalizer, vector = original_vector(*args, **kwargs)
        raw.envs[0].unwrapped._sim.env._max_episode_steps = 3
        return raw, normalizer, vector

    monkeypatch.setattr(native_evaluation, "native_vector", short_vector)
    result = native_evaluation.evaluate_native(second, "evaluation", episodes=2,
                                               runs_dir=tmp_path, device=device)
    evaluated = json.loads((result / "config.json").read_text())
    assert evaluated["settings"]["inference_device"] == device
    assert json.loads((result / "runtime.json").read_text())["device"].startswith(device)
    assert json.loads((result / "cost.json").read_text())["native_frames"] == 6
    assert json.loads((result / "cost.json").read_text())["reset_physics_ticks"] == 2
    assert json.loads((result / "summary.json").read_text())["normalization_frozen"]
    for i in range(2):
        with np.load(result / f"episodes/{i:04d}.npz") as trace:
            assert trace["observation"].shape == (3, 2, 64, 64)
            assert not trace["observation"][0, 0].any()
    with pytest.raises(FileExistsError):
        train_native("first", config, stop_at=8, runs_dir=tmp_path)
    with pytest.raises(ValueError, match="identical"):
        train_native("wrong", replace(config, gamma=.9), stop_at=24, resume=second, runs_dir=tmp_path)


def test_native_evaluation_device_and_partition_guards(tmp_path):
    from cars.research.native_evaluation import evaluate_native
    with pytest.raises(ValueError, match="device"):
        evaluate_native(tmp_path, "bad", device="auto")
    with pytest.raises(ValueError, match="episode count"):
        evaluate_native(tmp_path, "bad", episodes=201)


def test_native_evaluation_stops_without_consuming_an_extra_track(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    raw, norm, env = native_vector(NativeTrainingConfig(workers=1),
        partition="development", training=False, evaluation_episodes=1)
    try:
        raw.envs[0].unwrapped._sim.env._max_episode_steps = 1  # explicit test-only limit
        first = env.reset()
        _, _, done, info = env.step(np.zeros((1, 3), np.float32))
        assert done[0]
        assert len(raw.get_attr("track_seeds")[0]) == 1
        np.testing.assert_array_equal(info[0]["terminal_observation"][0], first[0, 1])
        assert norm.ret_rms.count == .0001
        with pytest.raises(RuntimeError, match="not actionable"):
            env.step(np.zeros((1, 3), np.float32))
    finally:
        env.close()
