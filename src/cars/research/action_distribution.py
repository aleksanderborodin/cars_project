"""Offline actor diagnostics on permitted histories; no environment or oracle calls."""
from typing import TypeAlias

import numpy as np
from scipy.special import ndtr
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.distributions import StateDependentNoiseDistribution
import torch

Observations: TypeAlias = dict[str, np.ndarray] | np.ndarray


def clipped_normal(mean: np.ndarray, std: np.ndarray,
                   low: np.ndarray, high: np.ndarray) -> dict[str, np.ndarray]:
    """Exact one-step marginal moments after clipping an unsquashed Gaussian.

    This does not model gSDE's temporal correlation or closed-loop driving.
    """
    mean, std = np.asarray(mean, np.float64), np.asarray(std, np.float64)
    low, high = np.asarray(low, np.float64), np.asarray(high, np.float64)
    if (not all(np.isfinite(a).all() for a in (mean, std, low, high)) or
            np.any(std <= 0) or np.any(low >= high)):
        raise ValueError("Finite Gaussian parameters, positive std and ordered finite bounds required")
    alpha, beta = (low - mean) / std, (high - mean) / std
    lower, upper = ndtr(alpha), ndtr(-beta)
    middle = ndtr(beta) - ndtr(alpha)
    pa = np.exp(-.5 * alpha**2) / np.sqrt(2 * np.pi)
    pb = np.exp(-.5 * beta**2) / np.sqrt(2 * np.pi)
    expected = low * lower + mean * middle + std * (pa - pb) + high * upper
    second = (low**2 * lower + (mean**2 + std**2) * middle +
              2 * mean * std * (pa - pb) + std**2 * (alpha * pa - beta * pb) + high**2 * upper)
    return {"lower_clip_probability": lower, "upper_clip_probability": upper,
            "expected_clipped_action": expected,
            "clipped_action_std": np.sqrt(np.maximum(0, second - expected**2))}


def replay_distribution(model: RecurrentPPO, observations: Observations,
                        recorded_actions: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Reset actor memory, replay one whole episode, and verify predictor parity."""
    if model.device.type != "cpu" or model.policy.squash_output or not model.use_sde:
        raise ValueError("Audit requires the declared CPU, unsquashed gSDE policy")
    model.policy.set_training_mode(False)
    n = len(next(iter(observations.values()))) if isinstance(observations, dict) else len(observations)
    if n < 1 or (isinstance(observations, dict) and any(len(v) != n for v in observations.values())):
        raise ValueError("Nonempty aligned observation history required")
    if recorded_actions is not None and recorded_actions.shape != (n, 3):
        raise ValueError("Recorded actions must align with the actor inputs")
    state = tuple(torch.zeros(model.policy.lstm_hidden_state_shape, dtype=torch.float32) for _ in range(2))
    prediction_state = None
    means, stds, actions = [], [], []
    prediction_error = recorded_error = 0.
    with torch.no_grad():
        for i in range(n):
            observation = ({k: v[i] for k, v in observations.items()}
                           if isinstance(observations, dict) else observations[i])
            tensor, _ = model.policy.obs_to_tensor(observation)
            distribution, state = model.policy.get_distribution(tensor, state, torch.tensor([float(i == 0)]))
            if not isinstance(distribution, StateDependentNoiseDistribution) or distribution.bijector is not None:
                raise ValueError("Expected unsquashed state-dependent Gaussian")
            mean = distribution.distribution.mean[0].cpu().numpy().copy()
            std = distribution.distribution.stddev[0].cpu().numpy().copy()
            # Verify the effective gSDE variance independently from Normal.stddev.
            scale = distribution.get_std(model.policy.log_std)
            variance = distribution._latent_sde.square() @ scale.square() + distribution.epsilon
            np.testing.assert_allclose(std**2, variance[0].cpu().numpy(), rtol=1e-5, atol=1e-7)
            action = np.clip(mean, model.action_space.low, model.action_space.high)
            if i < 16:
                predicted, prediction_state = model.predict(observation, state=prediction_state,
                    episode_start=np.array([i == 0]), deterministic=True)
                prediction_error = max(prediction_error, float(np.abs(predicted - action).max()))
                np.testing.assert_array_equal(predicted, action)
            if recorded_actions is not None:
                recorded_error = max(recorded_error, float(np.abs(recorded_actions[i] - action).max()))
                np.testing.assert_allclose(action, recorded_actions[i], rtol=0, atol=1e-6)
            means.append(mean)
            stds.append(std)
            actions.append(action)
    mean, std, action = np.stack(means), np.stack(stds), np.stack(actions)
    moments = clipped_normal(mean, std, model.action_space.low, model.action_space.high)
    # Independent action coordinates at one fixed history; not independent steps.
    active = ndtr((mean - .05) / std)
    return {"raw_mean": mean, "marginal_std": std, "deterministic_action": action, **moments,
            "gas_brake_overlap_probability": active[:, 1] * active[:, 2],
            "predictor_parity_max_error": np.array(prediction_error),
            "recorded_action_max_error": np.array(recorded_error),
            "actor_forward_examples": np.array(n + min(16, n))}


def summarize_distribution(arrays: dict[str, np.ndarray], low: np.ndarray, high: np.ndarray) -> dict:
    mean, std, action = (arrays[key] for key in ("raw_mean", "marginal_std", "deterministic_action"))
    channels = {}
    for i, label in enumerate(("steer", "gas", "brake")):
        channels[label] = {"raw_mean_mean": float(mean[:, i].mean()),
            "marginal_std_median": float(np.median(std[:, i])),
            "marginal_std_p90": float(np.quantile(std[:, i], .9)),
            "clipped_std_median": float(np.median(arrays["clipped_action_std"][:, i])),
            "deterministic_action_mean": float(action[:, i].mean()),
            "deterministic_bound_fraction": float(np.mean((mean[:, i] <= low[i]) | (mean[:, i] >= high[i]))),
            "expected_sample_clip_fraction": float(np.mean(arrays["lower_clip_probability"][:, i] +
                                                         arrays["upper_clip_probability"][:, i])),
            "expected_sample_action_mean": float(arrays["expected_clipped_action"][:, i].mean()),
            "expected_sample_vs_mode_mean_absolute_gap": float(np.mean(np.abs(
                arrays["expected_clipped_action"][:, i] - action[:, i])))}
    return {"decisions": len(mean), "channels": channels,
            "gas_brake_overlap_probability_mean": float(arrays["gas_brake_overlap_probability"].mean()),
            "deterministic_gas_brake_overlap_fraction": float(np.mean((action[:, 1] > .05) & (action[:, 2] > .05)))}
