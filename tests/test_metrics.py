from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from cars.env.metrics import EpisodeMetrics
from cars.env.state import CarState


class ScriptedEnv(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (3,), np.float32)
    observation_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def __init__(self, counts, *, finish=False, crash=False):
        self.counts = counts
        self.finish = finish
        self.crash = crash
        self.track = list(range(10))
        self.tile_visited_count = 0

    def reset(self, **kwargs):
        self.i = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        self.i += 1
        self.tile_visited_count = 9
        done = self.i == len(self.counts) - 1
        return np.zeros(1, np.float32), -100.0 if self.crash and done else 2.0, done, False, {"lap_finished": self.finish and done}


@pytest.mark.parametrize("counts,events,steps", [([4, 0, 0, 1, 0, 2, 0], 3, 4), ([0, 0, 0, 4, 0], 1, 3), ([4, 4, 1, 2], 0, 0)])
def test_off_track_transitions(monkeypatch, counts, events, steps):
    raw = ScriptedEnv(counts)
    monkeypatch.setattr(CarState, "from_env", classmethod(lambda cls, env: SimpleNamespace(wheels_on_road=raw.counts[raw.i], x=0, y=0)))
    env = EpisodeMetrics(raw)
    env.reset()
    for _ in counts[1:]:
        *_, info = env.step(np.zeros(3, np.float32))
    summary = info["episode_metrics"]
    assert summary["off_track_events"] == events
    assert summary["off_track_steps"] == steps
    assert summary["return"] == 2 * (len(counts) - 1)
    assert summary["progress"] == 0.9


@pytest.mark.parametrize("finish_step,crash,success", [(999, False, True), (1000, False, False), (10, True, False)])
def test_finish_and_crash(monkeypatch, finish_step, crash, success):
    raw = ScriptedEnv([4] * (finish_step + 1), finish=True, crash=crash)
    monkeypatch.setattr(CarState, "from_env", classmethod(lambda cls, env: SimpleNamespace(wheels_on_road=4, x=10000 if crash and raw.i == finish_step else 0, y=0)))
    env = EpisodeMetrics(raw)
    env.reset()
    for _ in range(finish_step):
        *_, info = env.step(np.zeros(3, np.float32))
    summary = info["episode_metrics"]
    assert summary["success"] is success
    assert summary["crash"] is crash
    assert summary["lap_time_s"] == (None if crash else finish_step / 50)
