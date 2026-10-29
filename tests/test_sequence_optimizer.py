"""Synthetic search objectives only; no privileged candidate oracle is used."""
import numpy as np
import pytest

from cars.planning.sequence_optimizer import ResidualOptimizer, SequenceOptimizerConfig, colored_noise


def make_optimizer(mode="icem"):
    return ResidualOptimizer(SequenceOptimizerConfig(mode=mode), 20, np.array([.1, .15, .15]), 19)


def test_spectral_noise_marginal_scale_temporal_correlation_and_bounds():
    white = colored_noise(np.random.default_rng(1), 10000, 20, 0.)
    colored = colored_noise(np.random.default_rng(1), 10000, 20, 1.)
    np.testing.assert_allclose(colored.var(axis=0), 1., atol=.05)
    assert abs(colored.mean()) < .02
    assert abs(np.mean(white[:, :-1] * white[:, 1:])) < .02
    assert np.mean(colored[:, :-1] * colored[:, 1:]) > .4
    # Odd horizons need the full-spectrum normalization too.
    odd = colored_noise(np.random.default_rng(2), 10000, 19, 1.)
    np.testing.assert_allclose(odd.var(axis=0), 1., atol=.05)
    result = make_optimizer().optimize(lambda x: np.sum(x**2, axis=(1, 2)))
    assert np.all(abs(result.residuals) <= np.array([.3, .45, .45]) + 1e-15)


def test_matched_initial_bank_work_and_mean_candidate_is_evaluated():
    banks = {}
    for mode in ("shooting", "icem"):
        banks[mode] = []

        def evaluate(x):
            banks[mode].append(x.copy())
            return np.sum((x - np.array([.1, .12, 0.]))**2, axis=(1, 2))

        result = make_optimizer(mode).optimize(evaluate)
        assert result.batch_sizes == [13, 8] and len(result.costs) == 21
        assert result.selected == np.argmin(result.costs)
        assert result.cached_reused == (1 if mode == "icem" else 0)
        assert result.shifted_reevaluated == 0
        assert not result.residuals[0].any()
    np.testing.assert_array_equal(banks["shooting"][0], banks["icem"][0])
    first = banks["icem"][0]
    costs = np.sum((first - np.array([.1, .12, 0.]))**2, axis=(1, 2))
    fitted = .9 * first[np.argsort(costs, kind="stable")[:4]].mean(axis=0)
    np.testing.assert_allclose(banks["icem"][1][0], fitted)
    assert not np.array_equal(banks["shooting"][1], banks["icem"][1])


def test_cached_first_batch_winner_shift_reevaluation_and_reset():
    optimizer = make_optimizer()
    calls = []

    def first_root(x):
        calls.append(x.copy())
        costs = np.arange(len(x), dtype=float) + 10
        if len(calls) == 1:
            costs[5] = -100.
        return costs

    first = optimizer.optimize(first_root)
    assert first.selected == 5 and len(calls) == 2
    elite = first.residuals[5].copy()
    next_calls = []

    def changed_root(x):
        next_calls.append(x.copy())
        return np.arange(len(x), dtype=float) + (0 if len(next_calls) == 1 else 20)

    second = optimizer.optimize(changed_root)
    assert second.shifted_reevaluated == 1 and second.selected == 0
    assert second.costs.min() == 0  # previous root's -100 cost must never survive
    np.testing.assert_array_equal(next_calls[0][1, :-1], elite[1:])
    optimizer.reset()
    calls.clear()
    replay = optimizer.optimize(first_root)
    np.testing.assert_array_equal(first.residuals, replay.residuals)
    np.testing.assert_array_equal(first.costs, replay.costs)


def test_reject_invalid_costs_and_configuration():
    with pytest.raises(ValueError, match="finite costs"):
        make_optimizer().optimize(lambda x: np.full(len(x), np.nan))
    with pytest.raises(ValueError, match="finite costs"):
        make_optimizer().optimize(lambda x: np.zeros(len(x) - 1))
    with pytest.raises(ValueError, match="exactly one"):
        SequenceOptimizerConfig(reuse_fraction=0.)
    with pytest.raises(ValueError, match="twice"):
        SequenceOptimizerConfig(batches=(4, 3))
