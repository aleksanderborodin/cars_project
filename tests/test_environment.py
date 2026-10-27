import gymnasium as gym
from gymnasium.envs.box2d import car_racing
import numpy as np
import pytest

from cars.env.carracing import StateCarRacing
from cars.env.make import make_env
from cars.env.state import CarState, OBS_SCHEMA, STATE_FIELDS, TrackInfo


@pytest.mark.parametrize("seed", [0, 9])
def test_bit_identity_with_official(seed):
    official = gym.make("CarRacing-v3", lap_complete_percent=0.95)
    state_env = make_env(seed, obs="raw")
    try:
        official.reset(seed=seed)
        state_env.reset()
        np.testing.assert_array_equal(official.unwrapped.track, state_env.unwrapped.track)
        assert official.unwrapped.reward == state_env.unwrapped.reward
        # Complete action sequences exercise time limits or official playfield exit.
        rng = np.random.default_rng(42)
        for step in range(1000):
            action = np.array([0.3 * np.sin(step / 50), rng.uniform(0.2, 0.8), 0], np.float32)
            _, r1, t1, l1, i1 = official.step(action)
            _, r2, t2, l2, i2 = state_env.step(action)
            assert (r1, t1, l1) == (r2, t2, l2)
            assert i1.get("lap_finished") == i2.get("lap_finished")
            c1, c2 = official.unwrapped.car, state_env.unwrapped.car
            assert tuple(c1.hull.position) == tuple(c2.hull.position)
            assert tuple(c1.hull.linearVelocity) == tuple(c2.hull.linearVelocity)
            assert c1.hull.angle == c2.hull.angle
            assert c1.hull.angularVelocity == c2.hull.angularVelocity
            assert [w.omega for w in c1.wheels] == [w.omega for w in c2.wheels]
            assert official.unwrapped.tile_visited_count == state_env.unwrapped.tile_visited_count
            if t1 or l1:
                break
    finally:
        official.close()
        state_env.close()


def test_state_schema_and_determinism():
    envs = [make_env(3) for _ in range(2)]
    try:
        obs = [env.reset()[0] for env in envs]
        np.testing.assert_array_equal(*obs)
        assert obs[0].dtype == np.float32
        assert obs[0].shape == (len(OBS_SCHEMA),)
        assert len(OBS_SCHEMA) == len(set(OBS_SCHEMA))
        assert envs[0].observation_space.contains(obs[0])
        assert len(CarState.from_env(envs[0]).as_array()) == len(STATE_FIELDS) == 20
        for _ in range(80):
            action = np.array([0.1, 0.3, 0.0], np.float32)
            outcomes = [env.step(action) for env in envs]
            np.testing.assert_array_equal(outcomes[0][0], outcomes[1][0])
            assert outcomes[0][1:] == outcomes[1][1:]
        np.testing.assert_array_equal(outcomes[0][0][-3:], action)
        np.testing.assert_array_equal(envs[0].reset()[0][-3:], np.zeros(3))
        track = TrackInfo.from_env(envs[0])
        np.testing.assert_allclose(np.linalg.norm(track.left - track.centre, axis=1), track.half_width)
        np.testing.assert_allclose(track.left + track.right, 2 * track.centre)
    finally:
        for env in envs:
            env.close()


def test_state_only_does_not_call_upstream_renderer(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Pixel renderer called")
    monkeypatch.setattr(car_racing.CarRacing, "_render", fail)
    env = make_env(0)
    try:
        env.reset()
        env.step(np.zeros(3, np.float32))
        assert env.unwrapped.screen is None and env.unwrapped.surf is None
    finally:
        env.close()


def test_hard_constant_scope_without_generating_heldout_tracks(monkeypatch):
    turn, width = car_racing.TRACK_TURN_RATE, car_racing.TRACK_WIDTH
    env = StateCarRacing(hard=True)
    def failed_creation(self):
        assert car_racing.TRACK_TURN_RATE > turn
        assert car_racing.TRACK_WIDTH < width
        raise RuntimeError("Simulated generator failure")
    monkeypatch.setattr(car_racing.CarRacing, "_create_track", failed_creation)
    with pytest.raises(RuntimeError):
        env._create_track()
    assert (car_racing.TRACK_TURN_RATE, car_racing.TRACK_WIDTH) == (turn, width)
    env.close()


def test_hard_geometry_and_grip_with_synthetic_track(monkeypatch):
    from types import SimpleNamespace
    turn, width = car_racing.TRACK_TURN_RATE, car_racing.TRACK_WIDTH
    def synthetic_creation(self):
        assert car_racing.TRACK_TURN_RATE == turn * 1.5
        assert car_racing.TRACK_WIDTH == width * 0.75
        self.track = [(0, 0, 0, 0), (1, 0.2, 1, 1)]
        self.road = [SimpleNamespace(road_friction=1.0) for _ in self.track]
        return True
    monkeypatch.setattr(car_racing.CarRacing, "_create_track", synthetic_creation)
    env = StateCarRacing(hard=True)
    assert env._create_track()
    assert all(tile.road_friction == 0.75 for tile in env.road)
    np.testing.assert_array_equal(env.track_info.half_width, [width * 0.75] * 2)
    assert (car_racing.TRACK_TURN_RATE, car_racing.TRACK_WIDTH) == (turn, width)
    env.close()
