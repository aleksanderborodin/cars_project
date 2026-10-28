"""New protocol checks use synthetic geometry and training/development only."""
import json
from pathlib import Path

import numpy as np
import pytest

from cars.env.partitions import Partitions, TrainingStream, WIDTH
from cars.env.protocol import DrivingEnv, LocalSensor, ProtocolConfig
from cars.env.state import CarState, TrackInfo
from cars.models.dynamics import Dynamics
from cars.planning.local import LocalPursuit, centreline
from cars.planning.search import BehaviorSearch


def test_partition_reservation_stream_resume_and_sealing(tmp_path):
    partitions = Partitions(development_count=8, calibration_count=9, test_count=11)
    manifest = partitions.manifest()
    assert "test" not in manifest and "test_seeds" not in manifest
    assert len(manifest["test_commitment_sha256"]) == 64
    assert set(manifest["development"]).isdisjoint(manifest["calibration"])
    with pytest.raises(ValueError, match="sealed"):
        partitions.seeds("test")
    with pytest.raises(ValueError, match="sealed"):
        DrivingEnv(ProtocolConfig(), "test")
    path = tmp_path / "manifest.json"
    partitions.write(path)
    before = path.read_bytes()
    partitions.write(path)
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        Partitions().write(path)
    a, b = TrainingStream(42, 1, 3), TrainingStream(999, 1, 3)
    history = [a.sample() for _ in range(100)]
    b.load_state_dict(json.loads(json.dumps(a.state_dict())))
    future = [a.sample() for _ in range(100)]
    assert future == [b.sample() for _ in range(100)]
    assert len(set(history + future)) == 200
    assert all(WIDTH <= seed < 2 * WIDTH and (seed - WIDTH) % 3 == 1 for seed in history)
    other = TrainingStream(42, 2, 3)
    assert set(history).isdisjoint(other.sample() for _ in range(100))


def straight_track() -> TrackInfo:
    # A closed rectangle whose far return segment stays outside the sensor.
    points = np.array([[-100, 0], [-30, 0], [0, 0], [30, 0], [100, 0],
                       [100, 100], [-100, 100]], dtype=float)
    return TrackInfo(points, np.array([0., 0., 0., 0., np.pi / 2, np.pi, -np.pi / 2]), np.ones(7) * 6)


def car() -> CarState:
    return CarState(0, 0, 0, 20, 0, 0, (0,) * 4, (0,) * 4, (True,) * 4, 123, .73)


def test_sensor_ignores_hidden_geometry_ids_and_truth_fields():
    config = ProtocolConfig()
    sensor = LocalSensor(config)
    track = straight_track()
    original = sensor.observe(track, car(), np.zeros(3, np.float32))
    changed = track.centre.copy()
    changed[5:, 1] += 100
    second = sensor.observe(TrackInfo(changed, track.tangent, track.half_width), car(), np.zeros(3, np.float32))
    assert set(original) == {"road", "visible", "ego", "previous_action"}
    for key in original:
        np.testing.assert_array_equal(original[key], second[key])
    assert not original["road"][original["visible"] == 0].any()
    assert original["road"][30, 30] == 1
    assert original["road"][30, 50] == 0
    # Cyclic reindexing the whole track must not reveal a hidden global index.
    rolled = TrackInfo(np.roll(track.centre, 3, axis=0), np.roll(track.tangent, 3), np.roll(track.half_width, 3))
    np.testing.assert_array_equal(sensor.road(track, car()), sensor.road(rolled, car()))


def test_local_repeat_matches_official_transitions_and_sanitizes_info():
    a = DrivingEnv(ProtocolConfig(action_repeat=4), seed=7)
    b = DrivingEnv(ProtocolConfig(action_repeat=1), seed=7)
    try:
        obs, info = a.reset()
        other, _ = b.reset()
        assert info == {} and a.observation_space.contains(obs)
        for key in obs:
            np.testing.assert_array_equal(obs[key], other[key])
        action = np.array([.05, .4, 0], np.float32)
        for _ in range(8):
            obs, reward, done, truncated, info = a.step(action)
            reward_b = 0
            for _ in range(4):
                other, value, done_b, truncated_b, _ = b.step(action)
                reward_b += value
            assert reward == reward_b and (done, truncated) == (done_b, truncated_b)
            assert set(info) == {"native_frames"} and info["native_frames"] == 4
            for key in obs:
                np.testing.assert_array_equal(obs[key], other[key])
        with pytest.raises(ValueError, match="Invalid action"):
            a.step(np.array([np.nan, 0, 0], np.float32))
        obs, info = a.reset()
        assert info == {} and not obs["previous_action"].any()
    finally:
        a.close()
        b.close()


def test_native_input_shape_and_no_ego_truth():
    env = DrivingEnv(ProtocolConfig(view="native"), seed=9)
    try:
        obs, info = env.reset()
        assert set(obs) == {"image", "previous_action"} and info == {}
        assert obs["image"].shape == (96, 96, 3) and obs["image"].dtype == np.uint8
        assert env.observation_space.contains(obs)
        obs, _, _, _, info = env.step(np.array([0, .2, 0], np.float32))
        assert set(info) == {"native_frames"}
    finally:
        env.close()


def test_dynamics_reload_and_search_have_no_environment_handle(tmp_path):
    rng = np.random.default_rng(3)
    ego = rng.normal(0, 1, (100, 3))
    action = rng.uniform(0, 1, (100, 3))
    previous = np.zeros_like(action)
    grip = np.ones(100)
    next_ego = ego + np.stack((action[:, 1], np.zeros(100), -.2 * action[:, 0]), axis=-1)
    model = Dynamics.fit(ego, action, previous, grip, next_ego, ridge=1e-6)
    model.save(tmp_path / "model.npz")
    restored = Dynamics.load(tmp_path / "model.npz")
    np.testing.assert_array_equal(model.predict(ego, action, previous, grip), restored.predict(ego, action, previous, grip))
    assert np.mean((model.predict(ego, action, previous, grip) - next_ego)**2) < 1e-8
    config = ProtocolConfig()
    observation = LocalSensor(config).observe(straight_track(), car(), np.zeros(3, np.float32))
    for policy in (LocalPursuit(config), BehaviorSearch(config, restored)):
        policy.reset()
        first = policy.act(observation)
        policy.reset()
        np.testing.assert_array_equal(first, policy.act(observation))
        assert first.shape == (3,) and np.isfinite(first).all()
        assert np.all(first >= [-1, 0, 0]) and np.all(first <= 1)


def test_reconstruction_does_not_mistake_visibility_edge_for_turn():
    config = ProtocolConfig()
    sensor = LocalSensor(config)
    x, y = sensor.points[..., 0], sensor.points[..., 1]
    road = (abs(y - .3 * x) < 6) & sensor.visible
    observation = {"road": road.astype(np.float32), "visible": sensor.visible.astype(np.float32)}
    path = centreline(observation, config)
    # Beyond the complete cross-sections, circle clipping would bend a straight
    # tilted road back toward y=0. The reconstruction must stop before that.
    assert path[-1, 0] < 28
    np.testing.assert_allclose(path[4:-4, 1], .3 * path[4:-4, 0], atol=.3)


def test_data_fit_and_evaluator_provenance_round_trip(tmp_path):
    from cars.research.data import collect, fit
    from cars.research.evaluation import evaluate_driver, paired_report
    from cars.research.artifacts import read_config
    training = collect("train", episodes=1, runs_dir=tmp_path)
    development = collect("development", partition="development", episodes=1, runs_dir=tmp_path)
    fitted = fit("model", training, development, runs_dir=tmp_path)
    assert json.loads((fitted / "diagnostics.json").read_text())["save_reload_equal"]
    config = ProtocolConfig(**read_config(fitted)["protocol"])
    reference = evaluate_driver(LocalPursuit(config), "reference", episodes=1, runs_dir=tmp_path)
    search = evaluate_driver(BehaviorSearch(config, Dynamics.load(fitted / "model.npz")),
                             "search", episodes=1, runs_dir=tmp_path)
    paired_report([reference, search], tmp_path / "report.json")
    report = json.loads((tmp_path / "report.json").read_text())
    assert len(report["paired_differences"]) == 1
    assert (search / "evaluated_dynamics.npz").exists()
    with np.load(search / "episodes/0000.npz") as data:
        assert len(data["observation_ego"]) == len(data["action"]) + 1
        assert data["frames"].sum() <= 1000
        import pandas as pd
        metrics = pd.read_csv(search / "metrics.csv").iloc[0]
        assert len(data["evaluation_offroad_frame_indices"]) == metrics.off_track_steps
    with pytest.raises(ValueError, match="separate"):
        fit("bad", development, training, runs_dir=tmp_path)
    with pytest.raises(FileExistsError):
        collect("train", episodes=1, runs_dir=tmp_path)


def test_recurrent_train_reload_and_episode_memory_reset(tmp_path):
    from cars.research.recurrent import RecurrentDriver, train
    root = train("tiny", frames=512, workers=1, runs_dir=tmp_path)
    assert json.loads((root / "reload_check.json").read_text())["initial_and_recurrent_action_equal"]
    driver = RecurrentDriver(root)
    env = DrivingEnv(ProtocolConfig(), seed=111)
    try:
        obs, _ = env.reset()
        first = driver.act(obs)
        driver.act(obs)
        driver.reset()
        np.testing.assert_array_equal(first, driver.act(obs))
    finally:
        env.close()
    resumed = train("continued", frames=512, workers=1, runs_dir=tmp_path, resume=root)
    cost = json.loads((resumed / "cost.json").read_text())
    assert cost["cumulative_native_frames"] >= 1024 and cost["partial_episode_restart"]
    before = set(json.loads((root / "training_tracks.json").read_text())[0])
    after = set(json.loads((resumed / "training_tracks.json").read_text())[0])
    assert before.isdisjoint(after)


def test_crossed_intervals_do_not_treat_tracks_as_independent_training_runs():
    from cars.eval.paired import difference_interval
    # Between-training variation persists even with many identical tracks.
    candidate = np.repeat(np.array([[0.], [100.], [200.]]), 1000, axis=1)
    reference = np.ones((1, 1000)) * 100
    result = difference_interval(candidate, reference, reps=1000)
    assert result["mean_difference"] == 0
    assert result["interval_95_percent"][0] < -50
    assert result["interval_95_percent"][1] > 50
    # Common track effects cancel against a deterministic paired reference.
    tracks = np.arange(20.)
    result = difference_interval(np.stack([tracks + 3] * 3), tracks[None], reps=1000)
    np.testing.assert_allclose(result["interval_95_percent"], [3, 3])
