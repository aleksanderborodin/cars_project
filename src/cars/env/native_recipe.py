"""Recipe-aligned native pixels; separate from the existing dictionary protocol."""
from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor

from cars.env.partitions import digest
from cars.env.protocol import DrivingEnv, ProtocolConfig


@dataclass(frozen=True)
class NativeRecipeProtocol:
    view: Literal["native_recipe"] = "native_recipe"
    version: int = 1
    action_repeat: int = 1
    image_size: int = 64
    frame_stack: int = 2

    def __post_init__(self) -> None:
        if (self.view, self.version, self.action_repeat, self.image_size, self.frame_stack) != (
                "native_recipe", 1, 1, 64, 2):
            raise ValueError("The audited native recipe has a fixed versioned interface")

    @property
    def identifier(self) -> str:
        return f"native-recipe-v{self.version}-{digest(asdict(self))[:12]}"


class NativeImage(gym.ObservationWrapper):
    """Drop previous-action metadata before any learning-side preprocessing."""

    def __init__(self, env: DrivingEnv):
        super().__init__(env)
        self.observation_space = env.observation_space["image"]

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return observation["image"]


class EndOfEvaluationPass(gym.Wrapper):
    """Avoid VecEnv's unused final autoreset consuming another reserved track.

    After the last terminal, VecEnv receives a placeholder terminal image on its
    mandatory reset. That placeholder must never be acted on; stepping raises.
    Actual episode observations, rewards and termination are untouched.
    """

    def __init__(self, env: gym.Env, episodes: int):
        super().__init__(env)
        if episodes < 1:
            raise ValueError("Positive evaluation pass length required")
        self.remaining = episodes
        self.terminal_image: np.ndarray | None = None

    def step(self, action: np.ndarray) -> tuple:
        if self.remaining == 0:
            raise RuntimeError("Evaluation pass complete; final reset image is not actionable")
        result = self.env.step(action)
        if result[2] or result[3]:
            self.remaining -= 1
            self.terminal_image = result[0]
        return result

    def reset(self, **kwargs) -> tuple:
        if self.remaining == 0:
            return self.terminal_image.copy(), {}
        return self.env.reset(**kwargs)


def native_environment(partition: str = "train", seed: int = 0,
                       worker: int = 0, workers: int = 1,
                       evaluation_episodes: int | None = None) -> gym.Env:
    if evaluation_episodes is not None and partition != "development":
        raise ValueError("Bounded passes are development-only")
    cv2.setNumThreads(1)
    env = NativeImage(DrivingEnv(ProtocolConfig(view="native", action_repeat=1),
                                partition, seed=seed, worker=worker, workers=workers))
    result = gym.wrappers.GrayscaleObservation(
        gym.wrappers.ResizeObservation(Monitor(env), (64, 64)), keep_dim=True)
    return EndOfEvaluationPass(result, evaluation_episodes) if evaluation_episodes is not None else result
