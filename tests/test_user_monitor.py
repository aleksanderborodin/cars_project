"""Synthetic ranking checks only. Never open actual user-only monitor outputs."""
import numpy as np

from cars.monitoring.oracle import ranking_statistics
from cars.env.protocol import LocalSensor, ProtocolConfig
from cars.models.dynamics import Dynamics
from cars.planning.search import BehaviorSearch


def test_tie_aware_ranking_synthetic_values():
    result = ranking_statistics(np.array([4., 7., 7., 1.]), np.array([0., 1., 2., 3.]), 0)
    assert result["rank_mid"] == 3 and result["hindsight_regret"] == 3
    assert result["top3_by_best_rank"] and not result["top1_including_ties"]
    tied = ranking_statistics(np.array([4., 7., 7., 1.]), np.array([1., 0., 2., 3.]), 1)
    assert (tied["rank_best"], tied["rank_worst"], tied["rank_mid"]) == (1, 2, 1.5)
    same = ranking_statistics(np.ones(21), np.arange(21.), 0)
    assert same["uninformative_all_tied"] and same["spearman"] is None
    assert same["rank_mid"] == 11 and same["hindsight_regret"] == 0


def test_frozen_candidates_contain_exact_chosen_first_action():
    config = ProtocolConfig()
    sensor = LocalSensor(config)
    road = (abs(sensor.points[..., 1]) < 6) & sensor.visible
    observation = {"road": road.astype(np.float32), "visible": sensor.visible.astype(np.float32),
                   "ego": np.array([25., 0., 0.], np.float32), "previous_action": np.zeros(3, np.float32)}
    policy = BehaviorSearch(config, Dynamics(np.zeros((22, 3))))
    action = policy.act(observation)
    assert policy.candidate_actions.shape == (21, 20, 3)
    selected = policy.diagnostics["selected_candidate"]
    np.testing.assert_array_equal(action, policy.candidate_actions[selected, 0])
    assert selected == int(np.argmin(policy.candidate_costs))
