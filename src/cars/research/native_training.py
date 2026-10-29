"""Native CNN-LSTM PPO with cumulative scheduling and auditable bounded profiling."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch
from torch import nn

from cars.env.native_recipe import NativeRecipeProtocol
from cars.research.artifacts import begin_run, read_config
from cars.research.driver_checkpoint import restore_rng, save_checkpoint
from cars.research.driver_config import NativeSchedule
from cars.research.native_config import NativeTrainingConfig
from cars.research.native_vector import native_vector
from cars.training import seed_everything, write_json

RECIPE_COMMIT = "61956d90d9f8ac126043ed4024efa12a462d97e1"


def policy_kwargs(config: NativeTrainingConfig) -> dict:
    return {"log_std_init": config.log_std_init, "ortho_init": False,
            "enable_critic_lstm": False, "activation_fn": nn.GELU,
            "lstm_hidden_size": config.lstm_hidden_size}


class RecipeClock(BaseCallback):
    def __init__(self, schedule: NativeSchedule):
        super().__init__()
        self.schedule = schedule
        self.rollout_seconds = 0.
        self.started = 0.

    def _on_rollout_start(self) -> None:
        self.started = time.perf_counter()

    def _on_rollout_end(self) -> None:
        self.rollout_seconds = time.perf_counter() - self.started

    def _on_step(self) -> bool:
        frames = sum(info["native_frames"] for info in self.locals["infos"])
        if frames != self.training_env.num_envs:
            raise RuntimeError("Native recipe requires exactly one frame per decision")
        self.schedule.frames += frames
        return True


def verify_reload(root: Path, model: RecurrentPPO, config: NativeTrainingConfig) -> dict:
    check_config = NativeTrainingConfig(seed=config.seed + 100000, workers=1)
    raw, normalizer, env = native_vector(check_config, normalization=root / "normalization.pkl", training=False)
    frames = 0
    try:
        # Check serialization on the same device. CPU/CUDA numerical agreement
        # is a separate diagnostic, not evidence that a checkpoint was corrupted.
        loaded = RecurrentPPO.load(root / "model.zip", device=model.device)
        cpu_loaded = RecurrentPPO.load(root / "model.zip", device="cpu")
        cpu_reference = RecurrentPPO.load(root / "model.zip", device="cpu")
        obs = env.reset()
        expected_memory = actual_memory = cpu_memory = cpu_reference_memory = None
        before = float(normalizer.ret_rms.count)
        error = cross_action_error = cross_state_error = 0.
        for i in range(2):
            episode_start = np.array([i == 0])
            expected, expected_memory = model.predict(obs, state=expected_memory,
                episode_start=episode_start, deterministic=True)
            actual, actual_memory = loaded.predict(obs, state=actual_memory,
                episode_start=episode_start, deterministic=True)
            np.testing.assert_array_equal(actual, expected)
            for value, reference in zip(actual_memory, expected_memory, strict=True):
                np.testing.assert_array_equal(value, reference)
            cpu_action, cpu_memory = cpu_loaded.predict(obs, state=cpu_memory,
                episode_start=episode_start, deterministic=True)
            reference_action, cpu_reference_memory = cpu_reference.predict(obs, state=cpu_reference_memory,
                episode_start=episode_start, deterministic=True)
            np.testing.assert_array_equal(cpu_action, reference_action)
            for value, reference in zip(cpu_memory, cpu_reference_memory, strict=True):
                np.testing.assert_array_equal(value, reference)
            cross_action_error = max(cross_action_error, float(np.max(np.abs(cpu_action - expected))))
            cross_state_error = max(cross_state_error, max(float(np.max(np.abs(value - reference)))
                for value, reference in zip(cpu_memory, expected_memory, strict=True)))
            error = max(error, float(np.max(np.abs(actual - expected))))
            obs, _, done, _ = env.step(actual)
            frames += 1
            if done.any():
                raise RuntimeError("Unexpected terminal during two-step reload check")
        if float(normalizer.ret_rms.count) != before:
            raise RuntimeError("Evaluation mutated normalization")
        return {"passed": True, "action_max_absolute_error": error, "native_frames": 2,
                "reset_physics_ticks": sum(map(len, raw.get_attr("track_seeds"))),
                "partition": "train", "normalization_frozen": True,
                "same_device_exact": True, "cpu_repeated_reload_exact": True,
                "cpu_vs_training_device_action_max_absolute_error": cross_action_error,
                "cpu_vs_training_device_memory_max_absolute_error": cross_state_error,
                "cross_device_note": "diagnostic only; no cross-device bitwise equivalence claim"}
    except Exception as error:
        write_json(root / "reload_check.json", {"passed": False, "error": str(error),
            "native_frames": frames, "reset_physics_ticks": sum(map(len, raw.get_attr("track_seeds")))})
        raise
    finally:
        env.close()


def train_native(run_id: str, config: NativeTrainingConfig, *, stop_at: int,
                 resume: Path | None = None, runs_dir: Path = Path("runs")) -> Path:
    if not 0 < stop_at <= config.total_decisions:
        raise ValueError("Stop must be within the declared aggregate decision budget")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    torch.set_num_threads(config.torch_threads)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    seed_everything(config.seed, threads=config.torch_threads)
    torch.use_deterministic_algorithms(True)
    protocol = NativeRecipeProtocol()
    schedule = NativeSchedule(config.learning_rate, config.total_decisions)
    completed: list[int] = []
    if resume:
        parent = read_config(resume)
        if (parent["protocol"] != protocol.__dict__ or
                parent["settings"].get("training") != config.resolved()):
            raise ValueError("Native continuation requires identical protocol and resolved config")
        state = json.loads((resume / "training_state.json").read_text())
        schedule.frames, completed = state["native_frames"], state["completed_milestones"]
        if state["decisions"] != schedule.frames or stop_at <= schedule.frames:
            raise ValueError("Continuation must advance a valid one-frame checkpoint")
    initial_frames = schedule.frames
    settings = {"partition": "train", "trainer_version": 1, "training": config.resolved(),
        "stop_at_decisions": stop_at, "parent_run": str(resume.resolve()) if resume else None,
        "algorithm": "RecurrentPPO", "policy": "CnnLstmPolicy",
        "architecture": "NatureCNN512; actor LSTM; empty MLP; linear critic; GELU; shared CNN",
        "wrappers": ["native RGB only", "Monitor", "resize64", "grayscale",
                     "reward-only VecNormalize", "VecFrameStack2", "VecTransposeImage"],
        "opencv_version": version("opencv-python-headless"),
        "schedule": "linear in cumulative aggregate decisions/native frames over full budget",
        "upstream_recipe_commit": RECIPE_COMMIT,
        "adaptations": "partition-enforced streams, recorded dependencies, deterministic kernels, post-update checkpoints",
        "continuation": "optimizer/RNG/normalizer/streams restored; fresh episodes and memory",
        "selection": "external declared campaign; trainer does not evaluate or select checkpoints",
        "deterministic_algorithms": True, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
    root = begin_run("native_training", run_id, protocol, settings, runs_dir)
    started = time.perf_counter()
    env = None
    try:
        raw, normalizer, env = native_vector(config, normalization=resume / "normalization.pkl" if resume else None)
        if resume:
            model = RecurrentPPO.load(resume / "model.zip", env=env, device=config.device,
                                     custom_objects={"learning_rate": schedule, "lr_schedule": schedule})
            for i, stream in enumerate(json.loads((resume / "streams.json").read_text())):
                raw.env_method("restore_training_stream", stream, indices=i)
            restore_rng(resume)
        else:
            kwargs = {key: getattr(config, key) for key in (
                "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef",
                "vf_coef", "max_grad_norm", "clip_range", "use_sde", "sde_sample_freq", "device", "seed")}
            model = RecurrentPPO("CnnLstmPolicy", env, learning_rate=schedule,
                                 policy_kwargs=policy_kwargs(config), verbose=0, **kwargs)
        model.lr_schedule = schedule
        digest = hashlib.sha256()
        for name, value in sorted(model.policy.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        write_json(root / "initial_policy.json", {"parameters_sha256": digest.hexdigest(),
            "fresh_initialization": resume is None, "net_arch": model.policy.net_arch,
            "features_dim": model.policy.features_dim})
        if model.policy.net_arch != [] or model.policy.features_dim != 512:
            raise RuntimeError("Standard CNN recipe architecture changed")
        if config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        write_json(root / "runtime.json", {"device": str(model.device), "pid": os.getpid(),
            "gpu": torch.cuda.get_device_name() if config.device == "cuda" else None,
            "torch_threads": torch.get_num_threads(), "cpu_count": os.cpu_count(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32})
        model.set_logger(configure(str(root / "logs"), ["csv", "json"]))
        callback = RecipeClock(schedule)
        initial_updates = model._n_updates
        rollouts = 0
        while schedule.frames < stop_at:
            write_json(root / "status.json", {"status": "running", "native_frames": schedule.frames,
                                               "stop_at_decisions": stop_at})
            tick = time.perf_counter()
            model.learn(config.workers * config.n_steps, callback=callback,
                        reset_num_timesteps=False, log_interval=None)
            elapsed = time.perf_counter() - tick
            training = {key: float(value) for key, value in model.logger.name_to_value.items()
                        if key.startswith("train/") and np.isscalar(value)}
            if not all(np.isfinite(value) for value in training.values()) or not all(
                    torch.isfinite(parameter).all() for parameter in model.policy.parameters()):
                raise FloatingPointError("Nonfinite training diagnostics or policy")
            if schedule.frames != model.num_timesteps:
                raise RuntimeError("Native-frame/decision clock mismatch")
            rollouts += 1
            row = {"decisions": model.num_timesteps, "rollout_seconds": callback.rollout_seconds,
                "learn_seconds": elapsed, "outside_rollout_seconds": elapsed - callback.rollout_seconds,
                "optimizer_steps_cumulative": max(int(s["step"]) for s in model.policy.optimizer.state.values() if "step" in s),
                **training}
            with (root / "learning.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            model.logger.dump(step=model.num_timesteps)
            print(f"{run_id}: {model.num_timesteps} decisions; learn={elapsed:.2f}s", flush=True)
            reached = [step for step in config.checkpoints if step <= schedule.frames and step not in completed]
            if reached:
                completed.extend(reached)
                save_checkpoint(root / "checkpoints" / f"frames_{schedule.frames:09d}", root,
                    model, normalizer, raw, schedule, completed, checkpoint_method="native_checkpoint")
        save_checkpoint(root, root, model, normalizer, raw, schedule, completed)
        wall = time.perf_counter() - started
        cost = {"native_frames": schedule.frames - initial_frames,
            "learner_decisions": schedule.frames - initial_frames, "rollout_updates": rollouts,
            "epochs_entered": model._n_updates - initial_updates, "wall_seconds": wall,
            "training_reset_physics_ticks": sum(map(len, raw.get_attr("track_seeds"))),
            "reload_check": None, "internal_physics_steps": 0,
            "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "child_peak_rss_kib": "not measured; parent RSS is not aggregate subprocess memory",
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if config.device == "cuda" else 0,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if config.device == "cuda" else 0}
        write_json(root / "cost.json", cost)  # retain training costs even if verification fails
        reload_check = verify_reload(root, model, config)
        write_json(root / "reload_check.json", reload_check)
        cost.update(reload_check=reload_check, wall_seconds=time.perf_counter() - started)
        write_json(root / "cost.json", cost)
        write_json(root / "status.json", {"status": "complete", "native_frames": schedule.frames,
            "stop_at_decisions": stop_at, "decisions": model.num_timesteps})
        return root
    except Exception as error:
        write_json(root / "failure.json", {"error": str(error), "native_frames": schedule.frames})
        write_json(root / "status.json", {"status": "failed", "native_frames": schedule.frames})
        raise
    finally:
        if env is not None:
            env.close()
