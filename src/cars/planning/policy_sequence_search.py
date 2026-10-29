"""Local, frozen-policy sequence search with matched batched rollout accounting."""
from pathlib import Path

import numpy as np

from cars.env.protocol import Observation, ProtocolConfig
from cars.planning.policy_rollouts import RolloutBatch, evaluate_feedback
from cars.planning.policy_search import PolicyPhysicsSearch, PolicySearchConfig
from cars.planning.sequence_optimizer import ResidualOptimizer, SequenceOptimizerConfig
from cars.planning.world import LocalRoadWorld


class PolicySequenceSearch(PolicyPhysicsSearch):
    def __init__(self, checkpoint: Path, protocol: ProtocolConfig, config: PolicySearchConfig,
                 optimizer: SequenceOptimizerConfig):
        if config.candidates != optimizer.evaluations:
            raise ValueError("Candidate count must equal actual base-plus-batch evaluations")
        self.optimizer = ResidualOptimizer(optimizer, config.horizon,
            np.array([config.steer_sigma, config.gas_sigma, config.brake_sigma]), config.planning_seed)
        self.method = "policy_colored_shooting" if optimizer.mode == "shooting" else "policy_icem"
        super().__init__(checkpoint, protocol, config)

    def reset(self) -> None:
        super().reset()
        self.optimizer.reset()
        self.candidate_first_estimates = self.candidate_first_ego = None

    def configuration(self) -> dict:
        return {**super().configuration(), "proposals": "full_horizon_colored_residuals_around_frozen_feedback_policy; base_zero",
            "optimizer": self.optimizer.configuration(),
            "online_distribution_optimization": self.optimizer.config.mode == "icem",
            "planning_rng": "seed, decision counter, stream0 for samples/stream2 for elite tail; reset per episode",
            "first_transition_commit": "owned snapshot from actual winning batch, including earlier cached winner",
            "real_memory": "one root actor call; all imagined batches use copied root memory"}

    def _search(self, observation: Observation, world: LocalRoadWorld, path: np.ndarray) -> np.ndarray:
        base = self.actor.act(observation)
        batches: list[RolloutBatch] = []

        def evaluate(residuals: np.ndarray) -> np.ndarray:
            batch = evaluate_feedback(self.model, self.actor.model, self.sensor, observation,
                world, path, base, self.actor.memory, residuals, self.config)
            batches.append(batch)
            return batch.costs

        result = self.optimizer.optimize(evaluate)
        self.decision += 1
        self.candidate_actions = np.concatenate([batch.actions for batch in batches])
        self.candidate_costs = result.costs.copy()
        self.candidate_first_estimates = np.concatenate([batch.first_estimates for batch in batches])
        self.candidate_first_ego = np.concatenate([batch.first_ego for batch in batches])
        selected = result.selected
        self.model.commit_estimate(self.candidate_first_estimates[selected])
        action = self.candidate_actions[selected, 0].copy()
        count, horizon = len(result.costs), self.config.horizon
        self.diagnostics = {"model_transitions": count * horizon,
            "internal_physics_frames": count * horizon * self.protocol.action_repeat,
            "actor_forward_examples": 1 + count * (horizon - 1),
            "actor_forward_calls": 1 + len(batches) * (horizon - 1),
            "candidates": count, "batch_sizes": result.batch_sizes,
            "cached_rollouts_reused": result.cached_reused, "shifted_elites_reevaluated": result.shifted_reevaluated,
            "selected_candidate": selected, "selected_batch": int(np.searchsorted(np.cumsum(result.batch_sizes), selected, side="right")),
            "predicted_cost": float(result.costs[selected]), "base_action": base.tolist(),
            "overrode_base": bool(np.max(abs(action - base)) > 1e-6),
            "predicted_next_ego": self.candidate_first_ego[selected].tolist()}
        return action
