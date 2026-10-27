"""Episode metrics; reset's initial physics tick is not an agent step."""
from typing import Any

import gymnasium as gym
from gymnasium.envs.box2d.car_racing import FPS, PLAYFIELD

from cars.env.state import Array, CarState


class EpisodeMetrics(gym.Wrapper):
    def reset(self, **kwargs: Any) -> tuple[Array, dict]:
        obs, info = self.env.reset(**kwargs)
        self.steps = 0
        self.episode_return = 0.0
        self.off_track_events = 0
        self.off_track_steps = 0
        self.off_track_step_indices: list[int] = []
        self.was_on_road = CarState.from_env(self).wheels_on_road > 0
        self.finish_step: int | None = None
        self.crash = False
        return obs, info

    def step(self, action: Array) -> tuple[Array, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        self.episode_return += float(reward)
        state = CarState.from_env(self)
        on_road = state.wheels_on_road > 0
        self.off_track_events += int(self.was_on_road and not on_road)
        if not on_road:
            self.off_track_steps += 1
            self.off_track_step_indices.append(self.steps)
        self.was_on_road = on_road
        self.crash |= abs(state.x) > PLAYFIELD or abs(state.y) > PLAYFIELD
        if info.get("lap_finished", False) and not self.crash:
            self.finish_step = self.steps
        if terminated or truncated:
            info["episode_metrics"] = self.summary()
        return obs, reward, terminated, truncated, info

    def summary(self) -> dict:
        return {"success": self.finish_step is not None and self.finish_step < 1000 and not self.crash,
                "lap_time_s": self.finish_step / FPS if self.finish_step is not None else None,
                "off_track_events": self.off_track_events, "off_track_steps": self.off_track_steps,
                "crash": self.crash, "return": self.episode_return,
                "progress": self.unwrapped.tile_visited_count / len(self.unwrapped.track),
                "steps": self.steps}
