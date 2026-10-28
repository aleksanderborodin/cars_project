"""Cheap frozen-policy controls for attributing simulation-search improvements."""
from pathlib import Path

import numpy as np

from cars.env.protocol import Observation
from cars.research.recurrent import RecurrentDriver


class FixedPolicyBias(RecurrentDriver):
    method = "policy_bias"

    def __init__(self, checkpoint: Path, bias: tuple[float, float, float]):
        self.bias = np.asarray(bias, dtype=np.float32)
        if self.bias.shape != (3,) or not np.isfinite(self.bias).all():
            raise ValueError("A finite three-control bias is required")
        super().__init__(checkpoint)

    def act(self, observation: Observation) -> np.ndarray:
        base = super().act(observation)
        action = np.clip(base + self.bias, [-1., 0., 0.], 1.).astype(np.float32)
        self.diagnostics = {"model_transitions": 0, "actor_forward_examples": 1,
            "base_action": base.tolist(), "overrode_base": bool(np.max(abs(action - base)) > 1e-6)}
        return action

    def configuration(self) -> dict:
        return {**super().configuration(), "constant_action_bias": self.bias.tolist(),
            "selection": "fixed control offset; no candidate simulation", "training_at_inference": False}
