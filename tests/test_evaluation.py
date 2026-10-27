import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest

from cars.env.make import make_env
from cars.env.splits import episode_seeds
from cars.env.state import CarState, TrackInfo
from cars.eval.episode_log import read_episode, write_episode
from cars.eval.report import aggregate
from cars.eval.runner import evaluate
from cars.policies.baseline import BaselinePolicy
from cars.policies.rust import RustPolicy, build_rust


@pytest.fixture(scope="module")
def baseline_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("base")
    frame = evaluate(BaselinePolicy(), "seen-1", 1, directory)
    return directory, frame


def test_baseline_episode_and_log_roundtrip(baseline_run, tmp_path):
    directory, frame = baseline_run
    row = frame.iloc[0]
    assert bool(row.success)
    assert row["return"] >= 900
    config = json.loads((directory / "config.json").read_text())
    assert config["seeds"] == [0]
    assert config["policy"] == BaselinePolicy().configuration()
    path = directory / row.log
    episode = read_episode(path)
    assert episode["meta"]["seed"] == 0
    assert episode["meta"]["method"] == "base"
    assert episode["summary"]["return"] == row["return"]
    assert sum(episode["steps"]["reward"]) == row["return"]
    assert len(episode["steps"]["t"]) == row.steps
    assert episode["events"]["finish_step"] == row.steps
    assert len(episode["events"]["off_track_steps"]) == row.off_track_steps
    copy = tmp_path / "roundtrip.json.gz"
    write_episode(copy, episode)
    assert read_episode(copy) == episode
    assert copy.read_bytes() == path.read_bytes()
    # Preserved geometry reconstructs raw observations and a policy reset.
    geometry = episode["track"]
    assert len(geometry["centre"]) == len(geometry["left"]) == len(geometry["right"])


def test_baseline_lap_is_bit_identical_to_official():
    official = gym.make("CarRacing-v3", lap_complete_percent=0.95)
    env = make_env(0, obs="raw")
    try:
        official.reset(seed=0)
        obs, _ = env.reset()
        policy = BaselinePolicy()
        policy.reset(env.unwrapped.track_info)
        for _ in range(1000):
            action = policy.act(CarState.from_env(env), obs)
            _, reward1, term1, trunc1, info1 = official.step(action)
            obs, reward2, term2, trunc2, info2 = env.step(action)
            assert (reward1, term1, trunc1) == (reward2, term2, trunc2)
            assert info1.get("lap_finished") == info2.get("lap_finished")
            if term2 or trunc2:
                assert term2 and info1["lap_finished"]
                assert info2["episode_metrics"]["success"]
                break
    finally:
        official.close()
        env.close()


def test_rust_build_load_and_episode(tmp_path):
    crate = Path(__file__).resolve().parents[1] / "rust/carctl"
    library = build_rust(crate, tmp_path / "target")
    policy = RustPolicy(library)
    env = make_env(0, obs="raw")
    try:
        obs, _ = env.reset()
        policy.reset(TrackInfo.from_env(env))
        action = policy.act(CarState.from_env(env), obs)
        assert action.shape == (3,) and action.dtype == np.float32
        assert env.action_space.contains(action)
    finally:
        env.close()
        policy.close()
    frame = evaluate(RustPolicy(library), "seen-1", 1, tmp_path / "rust_run")
    assert len(frame) == 1
    assert 0 < frame.iloc[0].steps <= 1000
    assert (tmp_path / "rust_run/candidate.so").is_file()


def test_multiprocessing_matches_serial(tmp_path):
    parallel = evaluate(BaselinePolicy(), "seen-1", 2, tmp_path / "parallel", workers=2)
    serial = evaluate(BaselinePolicy(), "seen-1", 2, tmp_path / "serial", workers=1)
    columns = ["seed", "success", "return", "lap_time_s", "off_track_events", "off_track_steps", "crash", "progress", "steps"]
    pd.testing.assert_frame_equal(parallel[columns], serial[columns])
    assert len(list((tmp_path / "parallel/eval/seen-1").glob("*.json.gz"))) == 2


def test_rliable_report_seed_bootstrap(tmp_path):
    paths = []
    for training_seed in range(5):
        path = tmp_path / f"{training_seed}.csv"
        pd.DataFrame([{"method": "example", "split": "seen-10", "training_seed": training_seed,
                       "seed": i, "return": 800 + training_seed * 20 + i, "success": True,
                       "lap_time_s": 15.0, "off_track_events": 0, "off_track_steps": 0,
                       "crash": False, "progress": 1.0} for i in range(10)]).to_csv(path, index=False)
        paths.append(path)
    report = aggregate(paths, reps=100, seed=5)
    repeated = aggregate(paths, reps=100, seed=5)
    pd.testing.assert_frame_equal(report, repeated)
    row = report.set_index("metric").loc["return"]
    assert row["mean"] == 844.5
    assert row.iqm == 844.5
    assert row.ci_low <= row.iqm <= row.ci_high
    assert row.bootstrap_unit == "training_seeds"
    assert row.n_runs == 5


def test_single_run_report_labels_episode_bootstrap(baseline_run):
    directory, _ = baseline_run
    result = aggregate([directory / "eval/seen-1/metrics.csv"], reps=10)
    assert set(result.bootstrap_unit) == {"episodes"}


def test_seed_sequence_and_factory_validation():
    assert episode_seeds("seen-10", 12) == list(range(10)) + [0, 1]
    fresh = episode_seeds("seen-inf", 20, seed=7)
    assert fresh == episode_seeds("seen-inf", 20, seed=7)
    assert len(set(fresh)) == 20 and min(fresh) >= 1_000_000
    with pytest.raises(ValueError):
        make_env(0, obs="pixels")
    with pytest.raises(ValueError):
        episode_seeds("seen-10", 0)


def test_factory_logging_records_config_and_actual_crash(tmp_path):
    env = make_env(0, log=tmp_path)
    try:
        env.reset()
        env.unwrapped.car.hull.position = (10000, 10000)
        _, reward, terminated, _, info = env.step(np.zeros(3, np.float32))
        assert terminated and reward == -100
        assert info["episode_metrics"]["crash"]
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["episodes"] == [{"episode_index": 0, "seed": 0}]
        episode = read_episode(info["episode_log"])
        assert episode["events"]["crash"] == 1
        assert episode["events"]["finish_step"] is None
        assert episode["summary"]["return"] == -100
    finally:
        env.close()
