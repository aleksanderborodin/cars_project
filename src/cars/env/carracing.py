"""Official CarRacing dynamics with the internal pixel-render hook replaced.

Do not copy upstream step/reset: inheriting them preserves upstream arithmetic,
contact handling, rewards and termination, including reset's initial physics tick.
"""
from threading import RLock

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d import car_racing

from cars.env.state import CarState, STATE_FIELDS, TrackInfo

_TRACK_LOCK = RLock()
HARD_TURN_MULTIPLIER = 1.5
HARD_WIDTH_MULTIPLIER = 0.75
HARD_ROAD_FRICTION = 0.75


class StateCarRacing(car_racing.CarRacing):
    def __init__(self, *, hard: bool = False):
        super().__init__(lap_complete_percent=0.95, continuous=True, render_mode=None)
        self.hard = hard
        self.track_half_width = car_racing.TRACK_WIDTH
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (len(STATE_FIELDS),), np.float32)

    def _create_track(self) -> bool:
        # Serialise constant overrides across our instances; process workers are
        # isolated. Concurrent track creation through *upstream* instances is not
        # supported while creating a hard track in the same process.
        with _TRACK_LOCK:
            turn, width = car_racing.TRACK_TURN_RATE, car_racing.TRACK_WIDTH
            try:
                if self.hard:
                    car_racing.TRACK_TURN_RATE = turn * HARD_TURN_MULTIPLIER
                    car_racing.TRACK_WIDTH = width * HARD_WIDTH_MULTIPLIER
                self.track_half_width = car_racing.TRACK_WIDTH
                success = super()._create_track()
                if success:
                    if self.hard:
                        for tile in self.road:
                            tile.road_friction = HARD_ROAD_FRICTION
                    self.track_info = TrackInfo.from_env(self)
                return success
            finally:
                car_racing.TRACK_TURN_RATE, car_racing.TRACK_WIDTH = turn, width

    def _render(self, mode: str) -> np.ndarray:
        if mode != "state_pixels":
            raise ValueError("Setting A exposes state only")
        return CarState.from_env(self).as_array()
