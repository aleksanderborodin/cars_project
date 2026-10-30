"""Independent numerical checks for clipped marginals and recurrent API parity."""
import gymnasium as gym
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from sb3_contrib import RecurrentPPO
import torch

from cars.research.action_distribution import clipped_normal, replay_distribution


@pytest.mark.parametrize("mean,std,low,high", [(0, 1, -1, 1), (-.3, .7, 0, 1),
                                              (1.4, .5, 0, 1), (9, .1, 0, 1)])
def test_clipped_normal_matches_numerical_integration(mean, std, low, high):
    r = clipped_normal(np.array(mean), np.array(std), np.array(low), np.array(high))
    left, right = norm.cdf(low, mean, std), norm.sf(high, mean, std)
    first = low * left + high * right + quad(lambda x: x * norm.pdf(x, mean, std), low, high)[0]
    second = low**2 * left + high**2 * right + quad(lambda x: x*x * norm.pdf(x, mean, std), low, high)[0]
    assert float(r["expected_clipped_action"]) == pytest.approx(first, abs=1e-10)
    assert float(r["clipped_action_std"]) == pytest.approx(np.sqrt(max(0, second-first**2)), abs=1e-8)
    assert float(r["lower_clip_probability"]) == pytest.approx(left)
    assert float(r["upper_clip_probability"]) == pytest.approx(right)
    with pytest.raises(ValueError):
        clipped_normal(np.array(0), np.array(0), np.array(0), np.array(1))


def test_recurrent_distribution_parity_and_reset_without_environment_steps():
    torch.set_num_threads(1)

    class Synthetic(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (4,), np.float32)
        action_space = gym.spaces.Box(np.array([-1, 0, 0], np.float32), 1)

        def reset(self, **kwargs):
            raise AssertionError("The offline audit must not reset or step an environment")

        def step(self, action):
            raise AssertionError("No environment calls allowed")

    model = RecurrentPPO("MlpLstmPolicy", Synthetic(), use_sde=True, seed=42, device="cpu",
                         n_steps=8, batch_size=8, policy_kwargs={"lstm_hidden_size": 16})
    observations = np.random.default_rng(7).uniform(-1, 1, (12, 4)).astype(np.float32)
    actions, state = [], None
    for i, obs in enumerate(observations):
        action, state = model.predict(obs, state=state, episode_start=np.array([i == 0]), deterministic=True)
        actions.append(action)
    first = replay_distribution(model, observations, np.stack(actions))
    second = replay_distribution(model, observations, np.stack(actions))
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert first["recorded_action_max_error"] == 0
    assert first["actor_forward_examples"] == 24
    # A wrong recorded control must fail instead of silently changing policy memory.
    corrupted = np.stack(actions)
    corrupted[2, 0] += .1
    with pytest.raises(AssertionError):
        replay_distribution(model, observations, corrupted)
