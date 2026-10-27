"""Agent-facing controls, outside the unmodified per-frame environment stack."""
import gymnasium as gym
import numpy as np

from cars.env.state import Array


def throttle_action(action: Array) -> Array:
    """Convert bounded [steer, throttle] into the official three controls."""
    steer, throttle = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    return np.array([steer, max(throttle, 0.0), max(-throttle, 0.0)], dtype=np.float32)


class ThrottleAxis(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def action(self, action: Array) -> Array:
        return throttle_action(action)


class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, repeat: int):
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise ValueError("action_repeat must be a positive integer")
        super().__init__(env)
        self.repeat = repeat

    def step(self, action: Array) -> tuple[Array, float, bool, bool, dict]:
        total_reward = 0.0
        for frame in range(1, self.repeat + 1):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, {**info, "action_repeat_frames": frame}


def agent_interface(env: gym.Env, action_repeat: int = 1, throttle_axis: bool = False) -> gym.Env:
    if throttle_axis:
        env = ThrottleAxis(env)
    return ActionRepeat(env, action_repeat)
