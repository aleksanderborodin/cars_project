"""Tiny-budget integration tests; training and evaluation touch seen tracks only."""
import copy
import json
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from cars.env.make import make_env
from cars.env.state import CarState, OBS_SCHEMA
from cars.eval.runner import evaluate
from cars.il.bc import BCConfig, train_bc
from cars.il.dagger import DAggerConfig, train_dagger
from cars.policies.sb3 import load_learned_policy
from cars.policies.torch_mlp import MLPPolicy
from cars.rl.train import RLConfig, train_rl
from cars.training import SeedSampler, TrainingSeeds


@pytest.fixture(scope="module", params=["il_bc", "il_dagger", "rl_ppo", "rl_sac"])
def trained(request, tmp_path_factory):
    runs = tmp_path_factory.mktemp("learning")
    method = request.param
    if method == "il_bc":
        root = train_bc(BCConfig(steps=64, epochs=2, hidden_sizes=(16, 16), batch_size=16), "tiny", runs)
    elif method == "il_dagger":
        root = train_dagger(DAggerConfig(steps=96, episodes=3, bootstrap_episodes=1, iterations=2,
                                        epochs=2, hidden_sizes=(16, 16), batch_size=16,
                                        beta_start=0.0), "tiny", runs)
    else:
        root = train_rl(RLConfig(algo=method.removeprefix("rl_"), steps=32, n_envs=2, device="cpu",
                                n_steps=8, n_epochs=1, batch_size=8, learning_starts=4,
                                buffer_size=128, hidden_sizes=(16, 16), eval_freq=16,
                                eval_episodes=1), "tiny", runs)
    return method, root


def test_training_artifacts_and_cost(trained):
    method, root = trained
    config = json.loads((root / "config.json").read_text())
    cost = json.loads((root / "cost.json").read_text())
    assert config["method"] == method and config["seed"] == 0
    assert config["obs_schema"] == OBS_SCHEMA
    assert cost["env_steps"] > 0 and cost["wall_time_s"] > 0
    if method.startswith("il_"):
        data = np.load(root / "dataset.npz")
        assert len(data["actions"]) == cost["expert_labels"] == cost["env_steps"]
        assert set(data["seeds"]).issubset(range(10))
        history = json.loads((root / "history.json").read_text())
        if method == "il_dagger":
            assert [r["expert_labels"] for r in history["rounds"]] == [32, 64, 96]
            assert [r["beta"] for r in history["rounds"]] == [1.0, 0.0, 0.0]
            assert all(e["expert_actions"] == 0 for e in history["episodes"] if e["round"] > 0)
        assert np.isfinite(data["observations"]).all()
    else:
        assert cost["env_steps"] == 32
        assert cost["evaluation_env_steps"] > 0
        assert list((root / "tensorboard").rglob("events.out.tfevents.*"))
        assert (root / "best/best_model.zip").exists()
        assert (root / "best/vecnormalize.pkl").exists()
        with (root / "vecnormalize.pkl").open("rb") as stream:
            normalizer = pickle.load(stream)
        assert normalizer.norm_obs and not normalizer.norm_reward


def test_save_load_bounds_and_frozen_normalization(trained):
    method, root = trained
    policy = load_learned_policy(method, root)
    env = make_env(0)
    try:
        obs, _ = env.reset()
        state = CarState.from_env(env)
        policy.reset(env.unwrapped.track_info)
        first = policy.act(state, obs)
        second = copy.deepcopy(policy).act(state, obs)
        assert env.action_space.contains(first)
        np.testing.assert_array_equal(first, second)
        if isinstance(policy, MLPPolicy):
            assert torch.isfinite(policy.network.obs_std).all()
            assert torch.all(policy.network.obs_std >= policy.config.normalization_floor)
            with torch.inference_mode():
                extreme = policy.network(torch.randn(64, len(OBS_SCHEMA)) * 1e6).numpy()
            assert (extreme >= np.array([-1, 0, 0])).all()
            assert (extreme <= 1).all()
        else:
            assert policy._model._n_updates > 0
            mean = policy._normalizer.obs_rms.mean.copy()
            count = policy._normalizer.obs_rms.count
            policy.act(state, obs + 1)
            np.testing.assert_array_equal(mean, policy._normalizer.obs_rms.mean)
            assert count == policy._normalizer.obs_rms.count
            direct, _ = policy._model.predict(policy._normalizer.normalize_obs(obs), deterministic=True)
            np.testing.assert_array_equal(first, direct)
    finally:
        env.close()


def test_runner_and_saved_policy_episode_determinism(trained, tmp_path):
    method, root = trained
    original_config = (root / "config.json").read_bytes()
    a = evaluate(load_learned_policy(method, root), "seen-1", 1, root)
    b = evaluate(load_learned_policy(method, root), "seen-1", 1, tmp_path / "repeat")
    assert a.iloc[0].method == method
    assert a.iloc[0].training_seed == 0
    columns = ["seed", "return", "success", "steps", "off_track_steps", "progress"]
    pd.testing.assert_frame_equal(a[columns], b[columns])
    from cars.eval.episode_log import read_episode

    assert read_episode(root / a.iloc[0].log)["steps"] == read_episode(tmp_path / "repeat" / b.iloc[0].log)["steps"]
    assert (root / "config.json").read_bytes() == original_config
    assert (root / "eval/seen-1/config.json").exists()
    with pytest.raises(FileExistsError):
        evaluate(load_learned_policy(method, root), "seen-1", 1, root)


def test_learned_policy_spawn_evaluation(tmp_path):
    root = train_bc(BCConfig(steps=8, epochs=1, hidden_sizes=(8,)), "spawn", tmp_path)
    frame = evaluate(MLPPolicy.load(root), "seen-1", 2, tmp_path / "parallel", workers=2)
    assert frame["return"].iloc[0] == frame["return"].iloc[1]
    assert set(frame.method) == {"il_bc"}


def test_seen_sampler_and_training_reset():
    a, b = SeedSampler("seen-10", 12), SeedSampler("seen-10", 12)
    seeds = [a.sample() for _ in range(50)]
    assert seeds == [b.sample() for _ in seeds]
    assert set(seeds) == set(range(10))
    streams = [[SeedSampler("seen-inf", 7, rank, 3) for rank in range(3)] for _ in range(2)]
    values = [[s.sample() for s in group for _ in range(20)] for group in streams]
    assert values[0] == values[1]
    assert len(set(values[0])) == 60 and min(values[0]) >= 1_000_000
    env = TrainingSeeds("seen-1", 13)
    try:
        obs1, info1 = env.reset(seed=456)
        obs2, info2 = env.reset(seed=456)
        assert info1["seed"] == info2["seed"] == 0
        np.testing.assert_array_equal(obs1, obs2)
    finally:
        env.close()


@pytest.mark.parametrize("split", ["unseen", "hard"])
def test_reject_held_out_before_creating_run(split, tmp_path):
    # Test validation only: no held-out seeds, tracks, or results are accessed.
    with pytest.raises(ValueError):
        train_bc(BCConfig(split=split), "bad", tmp_path)
    with pytest.raises(ValueError):
        train_dagger(DAggerConfig(split=split), "bad", tmp_path)
    with pytest.raises(ValueError):
        train_rl(RLConfig(split=split), "bad", tmp_path)
    assert not list(tmp_path.iterdir())
