"""Explicit score-only ablation; the score remains a local proxy, not official return."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cars.env.protocol import ProtocolConfig
from cars.planning.policy_search import PolicyPhysicsSearch, PolicySearchConfig


@dataclass(frozen=True)
class ObjectiveSearchConfig(PolicySearchConfig):
    penalty_scale: float = 1.

    def __post_init__(self) -> None:
        super().__post_init__()
        value = self.penalty_scale
        if (isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value < 0):
            raise ValueError("Finite nonnegative penalty scale required")


class PolicyObjectiveSearch(PolicyPhysicsSearch):
    method = "policy_objective_search"

    def __init__(self, checkpoint: Path, protocol: ProtocolConfig, config: ObjectiveSearchConfig):
        super().__init__(checkpoint, protocol, config)

    def configuration(self) -> dict:
        return {**super().configuration(),
            "selection": "minimum_scaled_proxy_cost; unchanged candidates/model/horizon",
            "objective": {"formula": "penalty_scale * original_nonprogress_penalties - original_terminal_arc_progress",
                "penalty_scale": self.config.penalty_scale,
                "official_reward_modified": False, "official_reward_estimator": False,
                "learning_or_terminal_value_added": False,
                "interpretation": "score-only development ablation; no improvement guarantee"}}
