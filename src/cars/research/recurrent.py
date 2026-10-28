"""Small recurrent PPO development adapter with the restricted observation API."""
from pathlib import Path
from dataclasses import asdict
import json
import random
import time

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from torch import nn

from cars.env.protocol import DrivingEnv, Observation, ProtocolConfig
from cars.research.artifacts import begin_run, read_config
from cars.training import seed_everything, sha256, write_json


class SensorEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.local = "road" in observation_space.spaces
        channels = 2 if self.local else 3
        extra = 6 if self.local else 3
        self.image = nn.Sequential(nn.Conv2d(channels, 16, 5, stride=3), nn.ReLU(),
                                   nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
                                   nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.output = nn.Sequential(nn.Linear(512 + extra, features_dim), nn.ReLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.local:
            pixels = torch.stack([observations["road"], observations["visible"]], dim=1)
            scale = observations["ego"].new_tensor([50., 10., 3.])
            extra = torch.cat([observations["ego"] / scale, observations["previous_action"]], dim=1)
        else:
            pixels = observations["image"]
            if pixels.shape[1] != 3:
                pixels = pixels.permute(0, 3, 1, 2)
            extra = observations["previous_action"]
        return self.output(torch.cat([self.image(pixels), extra], dim=1))


class FrameBudget(BaseCallback):
    def __init__(self, root: Path, budget: int):
        super().__init__()
        self.root, self.budget = root, budget
        self.frames = 0
        self.last_report = 0

    def _on_step(self) -> bool:
        self.frames += sum(info["native_frames"] for info in self.locals["infos"])
        if self.frames - self.last_report >= 10000:
            self.last_report = self.frames
            write_json(self.root / "progress.json", {"native_frames": self.frames, "decisions": self.num_timesteps})
            print(f"recurrent PPO: native frames={self.frames}/{self.budget}", flush=True)
        # Finish complete PPO rollouts; actual overshoot is counted and reported.
        return True


class RecurrentDriver:
    method = "recurrent_ppo"

    def __init__(self, root: Path):
        config = read_config(root)
        self.training_config = config["settings"]
        self.root = root.resolve()
        self.protocol = ProtocolConfig(**config["protocol"])
        self.model = RecurrentPPO.load(root / "model.zip", device="cpu")
        torch.set_num_threads(1)
        self.reset()

    def configuration(self) -> dict:
        return {"checkpoint": str(self.root), "deterministic_actions": True,
                "checkpoint_sha256": sha256(self.root / "model.zip"),
                "normalization": "fixed_input_scales; no fitted observation statistics",
                "training_reward_normalization": self.training_config.get("reward_normalization", False),
                "evaluation_normalization_updates": False}

    def reset(self) -> None:
        self.memory = None
        self.start = np.array([True])
        self.diagnostics = {"model_transitions": 0}

    def act(self, observation: Observation) -> np.ndarray:
        action, self.memory = self.model.predict(observation, state=self.memory,
            episode_start=self.start, deterministic=True)
        self.start[:] = False
        self.diagnostics = {"model_transitions": 0, "actor_forward_examples": 1}
        return np.asarray(action, dtype=np.float32)


def train(run_id: str, *, frames: int = 100000, seed: int = 0, workers: int = 4,
          protocol: ProtocolConfig | None = None, runs_dir: Path = Path("runs"),
          resume: Path | None = None) -> Path:
    protocol = protocol or ProtocolConfig()
    if frames < 1 or workers < 1:
        raise ValueError("Positive training budget and worker count required")
    seed_everything(seed)
    settings = {"seed": seed, "requested_native_frames": frames, "workers": workers,
                "n_steps": 128, "batch_size": 128, "n_epochs": 5,
                "lstm_hidden_size": 64, "features_dim": 128, "learning_rate": .0003,
                "checkpoint_selection": "last; no validation-based selection in engineering pilot",
                "normalization": "fixed_input_scales", "reward_normalization": False,
                "continuation": "weights/optimizer/RNG/sampler restored; fresh episodes, not exact mid-episode physics continuation",
                "parent_run": str(resume.resolve()) if resume else None}
    parent_cost = {}
    if resume:
        parent = read_config(resume)
        if parent["protocol"] != asdict(protocol) or parent["settings"]["workers"] != workers or parent["settings"]["seed"] != seed:
            raise ValueError("Continuation must preserve protocol, worker layout and run seed")
        if not (resume / "rng.pt").exists():
            raise ValueError("This older checkpoint lacks complete RNG state; use it only for inference")
        parent_cost = json.loads((resume / "cost.json").read_text())
    root = begin_run("recurrent_training", run_id, protocol, settings, runs_dir)
    environments = [DrivingEnv(protocol, "train", seed=seed, worker=i, workers=workers) for i in range(workers)]
    vector = DummyVecEnv([lambda env=env: env for env in environments])
    if resume:
        model = RecurrentPPO.load(resume / "model.zip", env=vector, device="cpu")
        streams = json.loads((resume / "streams.json").read_text())
        for environment, state in zip(environments, streams, strict=True):
            environment.restore_training_stream(state)
        rng = torch.load(resume / "rng.pt", map_location="cpu", weights_only=False)
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    else:
        model = RecurrentPPO("MultiInputLstmPolicy", vector, n_steps=128, batch_size=128,
            n_epochs=5, learning_rate=.0003, device="cpu", seed=seed, verbose=0,
            policy_kwargs={"features_extractor_class": SensorEncoder, "lstm_hidden_size": 64,
                           "net_arch": {"pi": [64], "vf": [64]}})
    initial_decisions, initial_updates = model.num_timesteps, model._n_updates
    callback = FrameBudget(root, frames)
    start = time.perf_counter()
    try:
        model.learn(total_timesteps=int(np.ceil(frames / protocol.action_repeat)), callback=callback,
                    reset_num_timesteps=resume is None)
        while callback.frames < frames:
            model.learn(total_timesteps=workers * 128, callback=callback, reset_num_timesteps=False)
        model.save(root / "model.zip")
        torch.save({"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                    "python": random.getstate()}, root / "rng.pt")
        write_json(root / "streams.json", [env.stream.state_dict() for env in environments])
        write_json(root / "training_tracks.json", [env.track_seeds for env in environments])
        write_json(root / "training_episodes.json", [env.completed_episodes for env in environments])
        prior_frames = parent_cost.get("cumulative_native_frames", parent_cost.get("native_frames", 0))
        write_json(root / "cost.json", {"native_frames": callback.frames,
            "cumulative_native_frames": prior_frames + callback.frames,
            "decisions": model.num_timesteps - initial_decisions,
            "requested_native_frames": frames, "overshoot_frames": callback.frames - frames,
            "wall_time_s": time.perf_counter() - start, "updates": model._n_updates - initial_updates,
            "partial_episode_restart": bool(resume),
            "expert_labels": 0, "truth_labels": 0})
        # Verify recurrent inference state after reload on train-side observations.
        check_env = DrivingEnv(protocol, "train", seed=seed + 100000)
        try:
            obs, _ = check_env.reset()
            expected, memory = model.predict(obs, state=None, episode_start=np.array([True]), deterministic=True)
            restored = RecurrentDriver(root)
            actual = restored.act(obs)
            np.testing.assert_array_equal(expected, actual)
            nxt, _, _, _, info = check_env.step(actual)
            expected, _ = model.predict(nxt, state=memory, episode_start=np.array([False]), deterministic=True)
            np.testing.assert_array_equal(expected, restored.act(nxt))
            write_json(root / "reload_check.json", {"initial_and_recurrent_action_equal": True,
                "extra_native_frames": info["native_frames"], "track_seeds": check_env.track_seeds})
        finally:
            check_env.close()
    except Exception as error:
        write_json(root / "failure.json", {"type": type(error).__name__, "message": str(error), "native_frames": callback.frames})
        raise
    finally:
        vector.close()
    return root
