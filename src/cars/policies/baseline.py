"""Full-information pure pursuit expert. Parameters tuned on seen-10 only."""
from dataclasses import dataclass, asdict

import numpy as np

from cars.env.state import Array, CarState, TrackInfo


@dataclass(frozen=True)
class BaselineConfig:
    max_speed: float = 100.0
    lateral_accel: float = 180.0
    braking_accel: float = 50.0
    lookahead_base: float = 4.0
    lookahead_speed: float = 0.22
    wheelbase: float = 4.0
    steering_gain: float = 1.0


class BaselinePolicy:
    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig()

    def configuration(self) -> dict:
        return asdict(self.config)

    def reset(self, track: TrackInfo) -> None:
        self.track = track
        cfg = self.config
        self.segment_lengths = np.linalg.norm(np.roll(track.centre, -1, axis=0) - track.centre, axis=1)
        curvature = np.abs(track.curvature)
        self.speed_profile = np.minimum(cfg.max_speed, np.sqrt(cfg.lateral_accel / np.maximum(curvature, 1e-5)))
        # Backward braking envelope around the entire closed circuit.
        for _ in range(3):
            for idx in range(len(track.centre) - 1, -1, -1):
                nxt = (idx + 1) % len(track.centre)
                self.speed_profile[idx] = min(self.speed_profile[idx],
                    np.sqrt(self.speed_profile[nxt] ** 2 + 2 * cfg.braking_accel * self.segment_lengths[idx]))

    def act(self, state: CarState, obs: Array) -> Array:
        cfg = self.config
        idx = state.tile_idx
        lookahead = cfg.lookahead_base + cfg.lookahead_speed * state.speed
        # Interpolate by arc length from the nearest centre point.
        target_idx = idx
        remaining = lookahead
        while remaining > self.segment_lengths[target_idx]:
            remaining -= self.segment_lengths[target_idx]
            target_idx = (target_idx + 1) % len(self.track.centre)
        nxt = (target_idx + 1) % len(self.track.centre)
        fraction = remaining / self.segment_lengths[target_idx]
        target = self.track.centre[target_idx] * (1 - fraction) + self.track.centre[nxt] * fraction
        dx, dy = target - (state.x, state.y)
        left = -np.sin(state.heading) * dx + np.cos(state.heading) * dy
        curvature = 2 * left / max(dx * dx + dy * dy, 1.0)
        # Positive action steers right; our curvature is positive to the left.
        steer = np.clip(-cfg.steering_gain * np.arctan(cfg.wheelbase * curvature), -1, 1)
        target_speed = self.speed_profile[(idx + 1) % len(self.speed_profile)]
        forward, _ = state.body_velocity
        error = target_speed - forward
        gas = np.clip(0.2 * error, 0, 1)
        brake = np.clip(-0.08 * error, 0, 0.8)
        # Reduce rear-wheel spin and lateral slip without altering physics.
        rear_speed = np.mean(state.wheel_omega[2:]) * 0.54
        spin = max(0.0, rear_speed - max(forward, 0))
        gas *= np.clip(1 - spin / 30, 0.15, 1) * np.clip(1 - abs(state.slip_angle) / 0.6, 0.1, 1)
        return np.array([steer, gas, brake], dtype=np.float32)
