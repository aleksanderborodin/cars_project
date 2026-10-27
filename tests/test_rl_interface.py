"""Agent interface equivalence, including the runner's per-frame trajectory logs."""
import json
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest

from cars.env.make import make_env
from cars.env.state import CarState
from cars.eval.episode_log import read_episode
from cars.eval.runner import evaluate
from cars.policies.sb3 import load_learned_policy
from cars.rl.interface import ActionRepeat, ThrottleAxis, agent_interface, throttle_action
from cars.rl.train import RLConfig, _eval_env, train_rl


class CounterEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
    observation_space = gym.spaces.Box(0.0, 1000.0, (1,), dtype=np.float32)

    def __init__(self, terminate_at=None):
        self.terminate_at = terminate_at

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.frames = 0
        return np.array([0], dtype=np.float32), {}

    def step(self, action):
        self.frames += 1
        return (np.array([self.frames], dtype=np.float32), float(self.frames),
                self.frames == self.terminate_at, False, {"frame": self.frames})


@pytest.mark.parametrize("terminate_at,limit", [(None, 1000), (None, 10), (7, 1000)])
def test_repeat_rewards_and_episode_boundary(terminate_at, limit):
    env = ActionRepeat(gym.wrappers.TimeLimit(CounterEnv(terminate_at), limit), 4)
    env.reset()
    frames = 0
    rewards = []
    while True:
        obs, reward, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
        count = info["action_repeat_frames"]
        assert reward == sum(range(frames + 1, frames + count + 1))
        frames += count
        rewards.append(reward)
        assert obs[0] == info["frame"] == frames
        if terminated or truncated:
            break
    assert frames == (terminate_at or limit)
    assert len(rewards) == (frames + 3) // 4
    assert terminated == (terminate_at is not None)
    assert truncated == (terminate_at is None)
    assert sum(rewards) == frames * (frames + 1) / 2


def test_throttle_bounds_and_mapping():
    env = ThrottleAxis(make_env(0))
    try:
        assert env.action_space.shape == (2,)
        for steer in (-1.0, 0.0, 1.0):
            for throttle in np.linspace(-1, 1, 101):
                action = env.action(np.array([steer, throttle]))
                assert env.env.action_space.contains(action)
                assert action[1] * action[2] == 0
                assert action[0] == steer
                assert action[1] - action[2] == pytest.approx(throttle)
        np.testing.assert_array_equal(throttle_action(np.array([3, -2])), [1, 0, 1])
    finally:
        env.close()


@pytest.fixture(scope="module", params=["ppo", "sac"])
def repeated_model(request, tmp_path_factory):
    root = train_rl(RLConfig(algo=request.param, split="seen-1", steps=32, n_envs=2,
                            action_repeat=4, throttle_axis=True, device="cpu", n_steps=8,
                            n_epochs=1, batch_size=8, learning_starts=4, buffer_size=128,
                            hidden_sizes=(16,), eval_freq=16, eval_episodes=1),
                    "repeated", tmp_path_factory.mktemp(request.param))
    return f"rl_{request.param}", root


def test_saved_adapter_matches_training_interface_in_runner(repeated_model, tmp_path):
    method, root = repeated_model
    policy = load_learned_policy(method, root)
    settings = json.loads((root / "config.json").read_text())["hyperparameters"]
    assert settings["action_repeat"] == 4 and settings["throttle_axis"] is True
    cost = json.loads((root / "cost.json").read_text())
    assert cost["decision_steps"] == 32
    assert cost["env_steps"] == 128
    assert cost["evaluation_env_steps"] == 2000
    best = load_learned_policy(method, root / "best/best_model.zip")
    assert best.action_repeat == 4 and best.throttle_axis is True
    assert best.configuration()["action_repeat"] == 4
    # All actual evaluation uses the same interface, including the periodic env.
    periodic = _eval_env("seen-1", 4, True)
    assert periodic.repeat == 4 and periodic.action_space.shape == (2,)
    periodic.close()
    training = agent_interface(make_env(0, log=tmp_path / "training"), 4, True)
    obs, _ = training.reset()
    policy.reset(training.unwrapped.track_info)
    model = policy._model
    normalizer = policy._normalizer
    decisions = 0
    total_reward = 0.0
    while True:
        action, _ = model.predict(normalizer.normalize_obs(obs), deterministic=True)
        obs, reward, terminated, truncated, info = training.step(action)
        total_reward += reward
        decisions += 1
        if terminated or truncated:
            break
    expected = read_episode(info["episode_log"])
    assert total_reward == pytest.approx(expected["summary"]["return"])
    training.close()
    # The runner deep-copies policies; replace that with our instance to count calls.
    with patch("cars.eval.runner.copy.deepcopy", return_value=policy), \
            patch.object(model, "predict", wraps=model.predict) as predict, \
            patch.object(normalizer, "normalize_obs", wraps=normalizer.normalize_obs) as normalize:
        frame = evaluate(policy, "seen-1", 1, tmp_path / "runner")
        assert predict.call_count == normalize.call_count == decisions
    actual = read_episode(tmp_path / "runner" / frame.iloc[0].log)
    assert actual["steps"] == expected["steps"]
    assert actual["summary"] == expected["summary"]
    assert len(actual["steps"]["t"]) == frame.iloc[0].steps
    # Reset starts a new decision even if an episode ended within a hold.
    env = make_env(0)
    obs, _ = env.reset()
    policy._remaining = 3
    policy.reset(env.unwrapped.track_info)
    with patch.object(model, "predict", wraps=model.predict) as predict:
        policy.act(CarState.from_env(env), obs)
        assert predict.call_count == 1
    env.close()


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_old_config_defaults(algo, tmp_path):
    root = train_rl(RLConfig(algo=algo, steps=8, n_envs=1, device="cpu", n_steps=8,
                            n_epochs=1, batch_size=8, learning_starts=0, buffer_size=32,
                            hidden_sizes=(8,), eval_freq=16, eval_episodes=1), "old", tmp_path)
    config = json.loads((root / "config.json").read_text())
    del config["hyperparameters"]["action_repeat"]
    del config["hyperparameters"]["throttle_axis"]
    (root / "config.json").write_text(json.dumps(config))
    policy = load_learned_policy(f"rl_{algo}", root)
    assert policy.action_repeat == 1 and policy.throttle_axis is False
    env = make_env(0)
    obs, _ = env.reset()
    policy.reset(env.unwrapped.track_info)
    with patch.object(policy._model, "predict", wraps=policy._model.predict) as predict:
        for _ in range(2):
            assert env.action_space.contains(policy.act(CarState.from_env(env), obs))
        assert predict.call_count == 2
    env.close()


@pytest.mark.parametrize("repeat", [0, -1, 1.5, True])
def test_invalid_repeat_rejected_before_run(repeat, tmp_path):
    with pytest.raises(ValueError, match="action_repeat"):
        train_rl(RLConfig(action_repeat=repeat), "invalid", tmp_path)
    assert not list(tmp_path.iterdir())
