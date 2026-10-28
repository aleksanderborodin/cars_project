"""Configured recurrent PPO with native-frame scheduling and immutable checkpoints.

This is a declared local-input adaptation of a pinned RL Zoo recipe. It is not
the old 100k engineering trainer or an original-pixel benchmark reproduction.
"""
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import torch
from torch import nn

from cars.env.protocol import DrivingEnv, ProtocolConfig
from cars.research.artifacts import begin_run, read_config
from cars.research.driver_checkpoint import restore_rng, save_checkpoint
from cars.research.driver_config import DriverTrainingConfig, NativeSchedule
from cars.research.driver_encoder import DriverSensorEncoder
from cars.training import seed_everything, write_json

ENCODER = "DriverSensorEncoder; equivalent deterministic average pooling; fixed ego scales; no fitted observation normalization"


def training_environment(protocol: ProtocolConfig, seed: int, worker: int, workers: int) -> DrivingEnv:
    return DrivingEnv(protocol, "train", seed=seed, worker=worker, workers=workers)


class NativeClock(BaseCallback):
    def __init__(self, schedule: NativeSchedule, root: Path, stop: int):
        super().__init__()
        self.schedule, self.root, self.stop = schedule, root, stop
        self.reported = schedule.frames

    def _on_step(self) -> bool:
        self.schedule.frames += sum(info["native_frames"] for info in self.locals["infos"])
        if self.schedule.frames - self.reported >= 10000:
            self.reported = self.schedule.frames
            write_json(self.root / "progress.json", {"status": "running", "native_frames": self.schedule.frames,
                "stop_at_native_frames": self.stop, "decisions": self.model.num_timesteps,
                "global_schedule_native_frames": self.schedule.total})
        return True  # complete the rollout AND update before saving/stopping


def train_driver(run_id: str, config: DriverTrainingConfig, *, stop_at: int,
                 protocol: ProtocolConfig | None = None, resume: Path | None = None,
                 runs_dir: Path = Path("runs")) -> Path:
    protocol = protocol or ProtocolConfig()
    if protocol.view == "global":
        raise ValueError("The privileged geometry diagnostic is not a training protocol")
    if not 0 < stop_at <= config.total_native_frames:
        raise ValueError("Stop milestone outside declared global budget")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.torch_threads)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA explicitly requested but unavailable")
    seed_everything(config.seed)
    torch.use_deterministic_algorithms(True)
    schedule = NativeSchedule(config.learning_rate, config.total_native_frames)
    completed: list[int] = []
    if resume:
        parent = read_config(resume)
        parent_training = dict(parent["settings"].get("training", {}))
        parent_training.setdefault("target_kl", None)  # older v2 runs had no KL stopping
        if (parent_training != config.resolved() or parent["protocol"] != protocol.__dict__
                or parent["settings"].get("encoder") != ENCODER):
            raise ValueError("Continuation requires identical resolved config and protocol")
        state = json.loads((resume / "training_state.json").read_text())
        schedule.frames = state["native_frames"]
        completed = state["completed_milestones"]
        if stop_at <= schedule.frames:
            raise ValueError("Continuation must advance past the saved native frame")
    initial_frames = schedule.frames
    settings = {"partition": "train", "trainer_version": 3, "training": config.resolved(),
        "stop_at_native_frames": stop_at, "parent_run": str(resume.resolve()) if resume else None,
        "algorithm": "RecurrentPPO", "policy": "MultiInputLstmPolicy",
        "encoder": ENCODER,
        "policy_architecture": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}, "activation": "GELU",
                                "ortho_init": False, "enable_critic_lstm": False},
        "reward_normalization": "VecNormalize norm_obs=False norm_reward=True; training only",
        "discount_units": "gamma per repeated decision; not matched to upstream native-frame discount",
        "selection": "scheduled checkpoints; choose standalone mean development return, then completion, then earlier",
        "schedule": "linear in cumulative actual native frames over declared global budget",
        "deterministic_algorithms": True, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "kl_stopping": "upstream minibatch approximate-KL check at 1.5*target; not a hard trust-region guarantee",
        "continuation": "post-update optimizer/RNG/normalizer/stream restored; fresh episodes and memory",
        "upstream_recipe_commit": "61956d90d9f8ac126043ed4024efa12a462d97e1",
        "adaptation": f"{protocol.view} encoder, repeat {protocol.action_repeat}, native-frame budget; not original pixel recipe"}
    root = begin_run("driver_training", run_id, protocol, settings, runs_dir)
    factories = [partial(training_environment, protocol, config.seed, i, config.workers) for i in range(config.workers)]
    raw = SubprocVecEnv(factories, start_method="spawn") if config.vector_backend == "subproc" else DummyVecEnv(factories)
    normalizer = None
    started = time.perf_counter()
    try:
        if resume:
            normalizer = VecNormalize.load(resume / "normalization.pkl", raw)
            normalizer.training = True
            if normalizer.norm_obs or not normalizer.norm_reward:
                raise ValueError("Unexpected normalization contract")
            model = RecurrentPPO.load(resume / "model.zip", env=normalizer, device=config.device,
                                      custom_objects={"learning_rate": schedule, "lr_schedule": schedule})
            for i, state in enumerate(json.loads((resume / "streams.json").read_text())):
                raw.env_method("restore_training_stream", state, indices=i)
            restore_rng(resume)
        else:
            normalizer = VecNormalize(raw, norm_obs=False, norm_reward=True, clip_reward=config.reward_clip, gamma=config.gamma)
            model = RecurrentPPO("MultiInputLstmPolicy", normalizer, n_steps=config.n_steps,
                batch_size=config.batch_size, n_epochs=config.n_epochs, learning_rate=schedule,
                gamma=config.gamma, gae_lambda=config.gae_lambda, ent_coef=config.ent_coef,
                vf_coef=config.vf_coef, max_grad_norm=config.max_grad_norm, clip_range=config.clip_range,
                target_kl=config.target_kl,
                use_sde=config.use_sde, sde_sample_freq=config.sde_sample_freq, device=config.device,
                seed=config.seed, verbose=0, policy_kwargs={"features_extractor_class": DriverSensorEncoder,
                    "features_extractor_kwargs": {"features_dim": config.features_dim},
                    "lstm_hidden_size": config.lstm_hidden_size, "activation_fn": nn.GELU,
                    "ortho_init": False, "enable_critic_lstm": False, "log_std_init": config.log_std_init,
                    "net_arch": {"pi": [64, 64], "vf": [64, 64]}})
        model.lr_schedule = schedule
        # Byte fingerprinting does not consume any training random stream.
        digest = hashlib.sha256()
        for name, value in sorted(model.policy.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        write_json(root / "initial_policy.json", {"parameters_sha256": digest.hexdigest(),
            "fresh_initialization": resume is None, "target_kl": model.target_kl})
        initial_decisions, initial_updates = model.num_timesteps, model._n_updates
        model.set_logger(configure(str(root / "logs"), ["csv", "json"]))
        callback = NativeClock(schedule, root, stop_at)
        write_json(root / "runtime.json", {"device": str(model.device),
            "gpu": torch.cuda.get_device_name() if model.device.type == "cuda" else None,
            "torch_threads": torch.get_num_threads(), "pid": os.getpid()})
        write_json(root / "status.json", {"status": "running", "native_frames": schedule.frames})
        while schedule.frames < stop_at:
            update_start = time.perf_counter()
            epochs_before = model._n_updates
            model.learn(total_timesteps=config.workers * config.n_steps, callback=callback,
                        reset_num_timesteps=False, log_interval=None)
            training = {key: float(value) for key, value in model.logger.name_to_value.items()
                        if key.startswith("train/") and np.isscalar(value)}
            if not all(np.isfinite(value) for value in training.values()):
                raise FloatingPointError("Nonfinite learner diagnostics")
            if not all(torch.isfinite(parameter).all() for parameter in model.policy.parameters()):
                raise FloatingPointError("Nonfinite policy parameters")
            row = {"native_frames": schedule.frames, "decisions": model.num_timesteps,
                   "update_wall_time_s": time.perf_counter() - update_start,
                   "epochs_entered_this_rollout": model._n_updates - epochs_before,
                   "optimizer_steps_cumulative": max(int(state["step"]) for state in model.policy.optimizer.state.values() if "step" in state),
                   **training}
            with (root / "learning.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            model.logger.dump(step=model.num_timesteps)
            reached = [frame for frame in config.checkpoints if frame <= schedule.frames and frame not in completed]
            if reached:
                completed.extend(reached)
                checkpoint = root / "checkpoints" / f"frames_{schedule.frames:09d}"
                save_checkpoint(checkpoint, root, model, normalizer, raw, schedule, completed)
                print(f"checkpoint {checkpoint.name}; native={schedule.frames}; lr={schedule(0.):.8f}", flush=True)
        save_checkpoint(root, root, model, normalizer, raw, schedule, completed)
        # Verify two consecutive actions after reload on a train-only observation.
        from cars.research.recurrent import RecurrentDriver
        check_env = DrivingEnv(protocol, "train", seed=config.seed + 100000)
        try:
            obs, _ = check_env.reset()
            restored = RecurrentDriver(root)
            memory = None
            frames = 0
            for i in range(2):
                expected, memory = model.predict(obs, state=memory, episode_start=np.array([i == 0]), deterministic=True)
                actual = restored.act(obs)
                # CPU/GPU inference may differ by floating-point roundoff.
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                obs, _, _, _, info = check_env.step(actual)
                frames += info["native_frames"]
            write_json(root / "reload_check.json", {"initial_and_recurrent_actions_match": True,
                "rtol": 1e-5, "atol": 1e-6, "extra_native_frames": frames,
                "track_seeds": check_env.track_seeds})
        finally:
            check_env.close()
        elapsed = time.perf_counter() - started
        write_json(root / "cost.json", {"native_frames": schedule.frames - initial_frames,
            "cumulative_native_frames": schedule.frames, "stop_at_native_frames": stop_at,
            "overshoot_frames": schedule.frames - stop_at, "decisions": model.num_timesteps - initial_decisions,
            "updates": model._n_updates - initial_updates, "wall_time_s": elapsed,
            "training_native_frames_per_s": (schedule.frames - initial_frames) / elapsed,
            "expert_labels": 0, "truth_labels": 0, "reload_check_native_frames": frames,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if model.device.type == "cuda" else 0})
        write_json(root / "status.json", {"status": "complete", "native_frames": schedule.frames})
        write_json(root / "progress.json", {"status": "complete", "native_frames": schedule.frames,
            "stop_at_native_frames": stop_at, "decisions": model.num_timesteps})
        return root
    except Exception as error:
        write_json(root / "failure.json", {"error": str(error), "native_frames": schedule.frames})
        write_json(root / "status.json", {"status": "failed", "native_frames": schedule.frames})
        raise
    finally:
        (normalizer if normalizer is not None else raw).close()
