"""Budgeted colored residual shooting and a policy-guided iCEM adaptation.

Algorithm source: Pinneri et al., iCEM, CoRL 2020/PMLR 2021;
martius-lab/iCEM revision 98c1c1fe2bfc94a87e4658b6c793f4e81bc30203.
This is a local reimplementation, not an original-protocol reproduction.
"""
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SequenceOptimizerConfig:
    mode: str = "shooting"
    batches: tuple[int, ...] = (12, 8)  # excludes the separately counted base in batch one
    noise_beta: float = 1.
    residual_bound_sigma: float = 3.
    elites: int = 4
    smoothing: float = .1
    reuse_fraction: float = .3
    version: int = 1

    def __post_init__(self) -> None:
        if self.mode not in ("shooting", "icem") or self.version != 1:
            raise ValueError("Unsupported sequence optimizer")
        if (len(self.batches) != 2 or self.elites < 2
                or any(n < 2 * self.elites for n in self.batches)):
            raise ValueError("Two batches, each at least twice the elite count, required")
        if (not np.isfinite([self.noise_beta, self.residual_bound_sigma, self.smoothing, self.reuse_fraction]).all()
                or not 0 <= self.noise_beta <= 4 or self.residual_bound_sigma <= 0
                or not 0 <= self.smoothing < 1 or not 0 <= self.reuse_fraction < 1):
            raise ValueError("Invalid sampling/distribution settings")
        if int(self.elites * self.reuse_fraction) != 1:
            raise ValueError("This bounded adaptation keeps exactly one cached/shifted elite")

    @property
    def evaluations(self) -> int:
        return 1 + sum(self.batches)


def colored_noise(generator: np.random.Generator, count: int, horizon: int, beta: float) -> np.ndarray:
    """Gaussian DFT filter with theoretical unit marginal variance, including DC.

    Frequencies are cycles/sample, cutoff max(f, 1/H). Filter amplitude f**(-beta/2).
    Normalize by sqrt(mean(filter**2)) over the full (positive/negative) spectrum.
    No per-trajectory centering/rescaling, which would alter the Gaussian draws.
    This explicit finite-length convention differs from the upstream noise package.
    """
    if count < 1 or horizon < 2 or not np.isfinite(beta) or not 0 <= beta <= 4:
        raise ValueError("Positive count, horizon >= 2, and beta in [0,4] required")
    white = generator.normal(size=(count, 3, horizon))
    frequency = np.maximum(abs(np.fft.fftfreq(horizon)), 1. / horizon)
    amplitude = frequency ** (-beta / 2.)
    normalization = np.sqrt(np.mean(amplitude**2))
    filtered = np.fft.irfft(np.fft.rfft(white, axis=-1) * amplitude[:horizon // 2 + 1], n=horizon, axis=-1)
    return (filtered / normalization).transpose(0, 2, 1)


@dataclass
class SequenceResult:
    residuals: np.ndarray
    costs: np.ndarray
    selected: int
    batch_sizes: list[int]
    cached_reused: int
    shifted_reevaluated: int


class ResidualOptimizer:
    def __init__(self, config: SequenceOptimizerConfig, horizon: int, scales: np.ndarray, seed: int):
        self.config, self.horizon, self.scales, self.seed = config, horizon, np.asarray(scales, float), seed
        if horizon < 2 or seed < 0 or self.scales.shape != (3,) or not np.isfinite(self.scales).all() or (self.scales < 0).any():
            raise ValueError("Invalid horizon/scales/seed")
        self.reset()

    def reset(self) -> None:
        self.mean = np.zeros((self.horizon, 3))
        self.elite: np.ndarray | None = None
        self.decision = 0

    def configuration(self) -> dict:
        return {**asdict(self.config), "batches": list(self.config.batches),
            "evaluations": self.config.evaluations, "noise": "gaussian_dft; cutoff_1/H; unit_marginal_variance_including_DC",
            "bounds": "residuals clipped at +/- bound_sigma * original channel sigma; final actions clipped separately",
            "mean_tail": "hold_last_fitted_mean", "shifted_elite_tail": "fresh_colored_sequence_last_step; separate_rng_stream",
            "standard_deviation": "reset_original_scales_each_real_decision; no_floor",
            "selection": "best_evaluated_across_all_batches; earliest_index_breaks_ties",
            "distribution_fit": f"{self.config.elites}_elites; new_batch_plus_best_cached_prior; base_eligible_in_first_fit",
            "adaptation": "feedback_policy_residuals; two_fixed_batches; within_budget_shift_and_mean_slots; global_incumbent"}

    def optimize(self, evaluate: Callable[[np.ndarray], np.ndarray]) -> SequenceResult:
        cfg = self.config
        adaptive = cfg.mode == "icem"
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, self.decision, 0]))
        tail_generator = np.random.default_rng(np.random.SeedSequence([self.seed, self.decision, 2]))
        mean = self.mean.copy() if adaptive else np.zeros_like(self.mean)
        std = np.broadcast_to(self.scales, mean.shape).copy()
        bound = cfg.residual_bound_sigma * self.scales
        residual_parts, cost_parts, sizes = [], [], []
        elite_ids = np.empty(0, int)
        shifted = cached = 0
        for iteration, count in enumerate(cfg.batches):
            residuals = np.clip(mean + std * colored_noise(generator, count, self.horizon, cfg.noise_beta), -bound, bound)
            if adaptive and iteration == 0 and self.elite is not None:
                tail = np.clip(mean[-1] + std[-1] * colored_noise(tail_generator, 1, self.horizon, cfg.noise_beta)[0, -1], -bound, bound)
                residuals[0] = np.concatenate((self.elite[1:], tail[None]), axis=0)
                shifted = 1
            if adaptive and iteration == len(cfg.batches) - 1:
                residuals[0] = mean  # this mean candidate is evaluated, never executed without scoring
            if iteration == 0:
                residuals = np.concatenate((np.zeros((1, self.horizon, 3)), residuals), axis=0)
            costs = np.asarray(evaluate(residuals.copy()), float)
            if costs.shape != (len(residuals),) or not np.isfinite(costs).all():
                raise ValueError("Evaluator must return finite costs for every candidate")
            offset = sum(sizes)
            sizes.append(len(residuals))
            residual_parts.append(residuals.copy())
            cost_parts.append(costs.copy())
            if adaptive:
                # Reuse one prior rollout only at this unchanged root/world/objective.
                eligible = np.arange(offset, offset + len(residuals))
                if iteration:
                    eligible = np.r_[elite_ids[:1], eligible]
                    cached += 1
                all_costs = np.concatenate(cost_parts)
                elite_ids = eligible[np.lexsort((eligible, all_costs[eligible]))[:cfg.elites]]
                elite = np.concatenate(residual_parts)[elite_ids]
                mean = cfg.smoothing * mean + (1. - cfg.smoothing) * elite.mean(axis=0)
                std = cfg.smoothing * std + (1. - cfg.smoothing) * elite.std(axis=0)
        all_residuals, all_costs = np.concatenate(residual_parts), np.concatenate(cost_parts)
        selected = int(np.argmin(all_costs))
        if adaptive:
            self.mean = np.concatenate((mean[1:], mean[-1:]), axis=0)
            self.elite = all_residuals[selected].copy()
        self.decision += 1
        return SequenceResult(all_residuals, all_costs, selected, sizes, cached, shifted)
