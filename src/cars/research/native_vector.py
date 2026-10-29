"""Audited vector-wrapper order, shared by native training and evaluation."""
from functools import partial
from pathlib import Path

from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize, VecTransposeImage,
)

from cars.env.native_recipe import native_environment
from cars.research.native_config import NativeTrainingConfig


def native_vector(config: NativeTrainingConfig, *, partition: str = "train",
                  normalization: Path | None = None,
                  training: bool = True,
                  evaluation_episodes: int | None = None) -> tuple[VecEnv, VecNormalize, VecEnv]:
    if not training and config.workers != 1:
        raise ValueError("Ordered development evaluation requires exactly one worker")
    if training and partition != "train":
        raise ValueError("Only the train partition may update normalization")
    if evaluation_episodes is not None and training:
        raise ValueError("Evaluation pass limits are not training wrappers")
    factories = [partial(native_environment, partition, config.seed, i, config.workers,
                         evaluation_episodes=evaluation_episodes)
                 for i in range(config.workers)]
    raw = (SubprocVecEnv(factories, start_method="spawn")
           if config.vector_backend == "subproc" else DummyVecEnv(factories))
    try:
        normalizer = (VecNormalize.load(normalization, raw) if normalization else
                      VecNormalize(raw, norm_obs=False, norm_reward=True,
                                   gamma=config.gamma, clip_reward=config.reward_clip))
        if normalizer.norm_obs or not normalizer.norm_reward:
            raise ValueError("Checkpoint must contain reward-only training normalization")
        normalizer.training = training
        normalizer.norm_reward = training
        vector = VecTransposeImage(VecFrameStack(normalizer, n_stack=2, channels_order="last"))
        return raw, normalizer, vector
    except Exception:
        raw.close()
        raise
