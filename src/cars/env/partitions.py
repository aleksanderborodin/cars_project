"""Versioned seed namespaces. Final-test identities are not exposed by this API.

Legacy collection was bounded below 2**31. New paper seeds occupy disjoint
2**40-wide namespaces; partition selection never depends on track outcomes.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np

Partition = Literal["train", "development", "calibration"]
WIDTH = 2**40
OFFSETS = {"train": WIDTH, "development": 2 * WIDTH,
           "calibration": 3 * WIDTH, "test": 4 * WIDTH}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def random_stream(seed: int, purpose: str) -> np.random.Generator:
    tag = int.from_bytes(hashlib.sha256(purpose.encode()).digest()[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([seed, tag]))


@dataclass(frozen=True)
class Partitions:
    version: str = "standard-v1"
    reservation_seed: int = 20261028
    development_count: int = 200
    calibration_count: int = 200
    test_count: int = 500

    def __post_init__(self) -> None:
        if min(self.development_count, self.calibration_count, self.test_count) < 1:
            raise ValueError("Partition capacities must be positive")
        if self.reservation_seed < 0 or self.version != "standard-v1":
            raise ValueError("Unsupported partition recipe")

    def _reserved(self, partition: str, count: int) -> list[int]:
        rng = random_stream(self.reservation_seed, f"{self.version}/{partition}")
        values: list[int] = []
        used: set[int] = set()
        while len(values) < count:
            value = OFFSETS[partition] + int(rng.integers(WIDTH))
            if value not in used:
                used.add(value)
                values.append(value)
        return values

    def seeds(self, partition: str, count: int | None = None) -> list[int]:
        if partition not in ("development", "calibration"):
            raise ValueError("Only development/calibration manifests are available; final test is sealed")
        capacity = getattr(self, f"{partition}_count")
        if count is not None and not 0 < count <= capacity:
            raise ValueError("Requested count exceeds the reserved partition")
        return self._reserved(partition, capacity)[:count]

    def contains(self, partition: str, seed: int) -> bool:
        if partition == "train":
            return WIDTH <= seed < 2 * WIDTH
        return seed in self.seeds(partition)

    def manifest(self) -> dict:
        # Generate only integer identities for a commitment: no environment or
        # policy is instantiated, and no held-out identities leave this function.
        return {"recipe": asdict(self), "train_namespace": [WIDTH, 2 * WIDTH],
                "development": self.seeds("development"),
                "calibration": self.seeds("calibration"),
                "test_commitment_sha256": digest(self._reserved("test", self.test_count)),
                "legacy_sampler_disjoint_below": 2**31}

    @property
    def fingerprint(self) -> str:
        return digest(self.manifest())

    def write(self, path: Path) -> None:
        value = {**self.manifest(), "sha256": self.fingerprint}
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise FileExistsError("Refusing to replace a different partition manifest")
            return
        path.write_text(json.dumps(value, indent=2) + "\n")


class TrainingStream:
    """Fresh tracks; stream state is serializable and independent of actions."""

    def __init__(self, seed: int, worker: int = 0, workers: int = 1):
        if seed < 0 or not 0 <= worker < workers:
            raise ValueError("Invalid seed or worker configuration")
        self.worker, self.workers = worker, workers
        self.rng = random_stream(seed, f"training_tracks/{worker}/{workers}")
        self.used: set[int] = set()

    def sample(self) -> int:
        limit = (WIDTH - 1 - self.worker) // self.workers + 1
        if len(self.used) >= limit:
            raise RuntimeError("Training namespace exhausted")
        while True:
            seed = WIDTH + self.worker + self.workers * int(self.rng.integers(limit))
            if seed not in self.used:
                self.used.add(seed)
                return seed

    def state_dict(self) -> dict:
        return {"worker": self.worker, "workers": self.workers,
                "rng": self.rng.bit_generator.state, "used": sorted(self.used)}

    def load_state_dict(self, state: dict) -> None:
        if (state["worker"], state["workers"]) != (self.worker, self.workers):
            raise ValueError("Worker layout cannot change during exact continuation")
        if any(not WIDTH <= value < 2 * WIDTH or (value - WIDTH) % self.workers != self.worker
               for value in state["used"]):
            raise ValueError("Invalid training-stream state")
        self.rng.bit_generator.state = state["rng"]
        self.used = set(state["used"])
