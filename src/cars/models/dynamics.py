"""Compact supervised dynamics model fitted only to observed ego transitions."""
from pathlib import Path

import numpy as np

SCALE = np.array([50., 10., 3.])


def features(ego: np.ndarray, action: np.ndarray, previous: np.ndarray, grip: np.ndarray) -> np.ndarray:
    v, lateral, yaw = np.moveaxis(np.asarray(ego) / SCALE, -1, 0)
    steer, gas, brake = np.moveaxis(np.asarray(action), -1, 0)
    old_steer, old_gas, old_brake = np.moveaxis(np.asarray(previous), -1, 0)
    return np.stack([np.ones_like(v), v, lateral, yaw, steer, gas, brake,
                     old_steer, old_gas, old_brake, steer * v, steer * abs(v),
                     v * yaw, lateral * yaw, gas / (1 + abs(v)), brake * v,
                     grip, grip * gas, grip * brake, grip * steer * v,
                     v * abs(v), steer - old_steer], axis=-1)


class Dynamics:
    def __init__(self, coefficients: np.ndarray, repeat: int = 4):
        if coefficients.shape != (22, 3) or not np.isfinite(coefficients).all():
            raise ValueError("Invalid dynamics coefficients")
        self.coefficients, self.repeat = coefficients, repeat

    @classmethod
    def fit(cls, ego: np.ndarray, action: np.ndarray, previous: np.ndarray,
            grip: np.ndarray, next_ego: np.ndarray, *, repeat: int = 4, ridge: float = .01) -> "Dynamics":
        x = features(ego, action, previous, grip)
        target = (next_ego - ego) / SCALE
        penalty = np.eye(x.shape[1]) * ridge
        penalty[0, 0] = 0
        return cls(np.linalg.solve(x.T @ x + penalty, x.T @ target), repeat)

    def predict(self, ego: np.ndarray, action: np.ndarray, previous: np.ndarray,
                grip: np.ndarray) -> np.ndarray:
        result = ego + features(ego, action, previous, grip) @ self.coefficients * SCALE
        return np.clip(result, [-10, -35, -6], [120, 35, 6])

    def advance(self, pose: np.ndarray, ego: np.ndarray, action: np.ndarray,
                previous: np.ndarray, grip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        next_ego = self.predict(ego, action, previous, grip)
        average = (ego + next_ego) / 2
        dt = self.repeat / 50
        heading = pose[..., 2] + average[..., 2] * dt / 2
        forward, left = average[..., 0], average[..., 1]
        displacement = np.stack([np.cos(heading) * forward - np.sin(heading) * left,
                                 np.sin(heading) * forward + np.cos(heading) * left,
                                 average[..., 2]], axis=-1) * dt
        return pose + displacement, next_ego

    def save(self, path: Path) -> None:
        np.savez(path, coefficients=self.coefficients, repeat=np.array(self.repeat))

    @classmethod
    def load(cls, path: Path) -> "Dynamics":
        with np.load(path, allow_pickle=False) as data:
            return cls(data["coefficients"], int(data["repeat"]))
