"""Split resolution. Never materialise held-out results during method design."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    name: str
    seeds: tuple[int, ...] | None
    hard: bool = False


SPLITS = {
    **{f"seen-{k}": Split(f"seen-{k}", tuple(range(k))) for k in (1, 10, 100)},
    "seen-inf": Split("seen-inf", None),
    "unseen": Split("unseen", tuple(range(10000, 10100))),
    "hard": Split("hard", tuple(range(20000, 20100)), True),
}


def episode_seeds(split: str, n: int, *, seed: int = 0) -> list[int]:
    if n < 1:
        raise ValueError("n_episodes must be positive")
    spec = SPLITS[split]
    if spec.seeds is None:
        # A reproducible fresh-seed sequence, unique within each evaluation.
        rng = np.random.default_rng(seed)
        result: list[int] = []
        used: set[int] = set()
        while len(result) < n:
            value = int(rng.integers(1_000_000, 2**31))
            if value not in used:
                used.add(value)
                result.append(value)
        return result
    return [spec.seeds[i % len(spec.seeds)] for i in range(n)]
