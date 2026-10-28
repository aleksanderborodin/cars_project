"""Crossed model/track bootstrap; common tracks, independently sampled models."""
import numpy as np


def paired_checkpoint_interval(candidate: np.ndarray, reference: np.ndarray, *, reps: int = 5000,
                               seed: int = 0) -> dict:
    """Same frozen checkpoints x common tracks: jointly resample both axes.

    The caller must verify row pairing with checkpoint hashes. This is not for
    independently trained methods whose initialization seeds happen to match.
    Stochastic repeats must have a declared aggregation/resampling protocol.
    """
    candidate, reference = np.asarray(candidate, float), np.asarray(reference, float)
    if (candidate.ndim != 2 or candidate.shape != reference.shape or not candidate.size or reps < 1
            or not np.isfinite(candidate).all() or not np.isfinite(reference).all()):
        raise ValueError("Finite matched checkpoint-by-track matrices required")
    difference = candidate - reference
    rng = np.random.default_rng(seed)
    checkpoints = rng.integers(len(difference), size=(reps, len(difference)))
    tracks = rng.integers(difference.shape[1], size=(reps, difference.shape[1]))
    draws = difference[checkpoints[:, :, None], tracks[:, None, :]].mean(axis=(1, 2))
    return {"mean_difference": float(difference.mean()),
        "interval_95_percent": np.quantile(draws, [.025, .975]).tolist(),
        "paired_checkpoints": len(difference), "common_tracks": difference.shape[1],
        "reps": reps, "seed": seed, "selection_adjusted": False,
        "units": "joint frozen-checkpoint draws and joint track draws; not independent algorithm models"}


def difference_interval(candidate: np.ndarray, reference: np.ndarray, *, reps: int = 5000,
                        seed: int = 0) -> dict:
    """Matrices are independently trained models x identical ordered tracks.

    A deterministic reference has one row. Matching integer run seeds does not
    create model pairing. Intervals are exploratory with very few trained models.
    """
    candidate, reference = np.asarray(candidate, float), np.asarray(reference, float)
    if (candidate.ndim != 2 or reference.ndim != 2 or not candidate.size or not reference.size
            or candidate.shape[1] != reference.shape[1] or reps < 1
            or not np.isfinite(candidate).all() or not np.isfinite(reference).all()):
        raise ValueError("Finite nonempty model-by-common-track matrices required")
    rng = np.random.default_rng(seed)
    tracks = rng.integers(candidate.shape[1], size=(reps, candidate.shape[1]))
    candidate_models = rng.integers(len(candidate), size=(reps, len(candidate)))
    reference_models = rng.integers(len(reference), size=(reps, len(reference)))
    a = candidate[candidate_models[:, :, None], tracks[:, None, :]].mean(axis=(1, 2))
    b = reference[reference_models[:, :, None], tracks[:, None, :]].mean(axis=(1, 2))
    return {"mean_difference": float(candidate.mean() - reference.mean()),
            "interval_95_percent": np.quantile(a - b, [.025, .975]).tolist(),
            "candidate_models": len(candidate), "reference_models": len(reference),
            "common_tracks": candidate.shape[1], "reps": reps, "seed": seed,
            "units": "independent model draws; paired track draws shared across models and methods",
            "selection_adjusted": False}
