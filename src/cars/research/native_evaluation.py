"""Development-only native evaluation with frozen statistics and episode memory."""
from dataclasses import replace
from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
import torch

from cars.env.native_recipe import NativeRecipeProtocol
from cars.env.partitions import Partitions
from cars.research.artifacts import begin_run, read_config
from cars.research.native_config import NativeTrainingConfig
from cars.research.native_vector import native_vector
from cars.training import sha256, write_json


def evaluate_native(checkpoint: Path, run_id: str, *, episodes: int = 20,
                    runs_dir: Path = Path("runs"), device: str = "cpu") -> Path:
    if not 0 < episodes <= Partitions().development_count:
        raise ValueError("Development episode count outside reserved manifest")
    if device not in ("cpu", "cuda"):
        raise ValueError("Explicit cpu or cuda inference device required")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA inference requested but unavailable")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    parent = read_config(checkpoint)
    protocol = NativeRecipeProtocol(**parent["protocol"])
    config = replace(NativeTrainingConfig(**parent["settings"]["training"]),
                     workers=1, vector_backend="dummy", device=device)
    torch.set_num_threads(config.torch_threads)
    model = RecurrentPPO.load(checkpoint / "model.zip", device=device)
    root = begin_run("native_evaluation", run_id, protocol, {"partition": "development",
        "episodes": episodes, "checkpoint": str(checkpoint.resolve()),
        "artifact_sha256": {name: sha256(checkpoint / name) for name in
                            ("config.json", "model.zip", "normalization.pkl")},
        "deterministic_actions": True, "reward_normalization": False, "inference_device": device,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "history": "frame stack and recurrent memory reset on each episode"}, runs_dir)
    raw, normalizer, env = native_vector(config, partition="development",
        normalization=checkpoint / "normalization.pkl", training=False, evaluation_episodes=episodes)
    before = (float(normalizer.ret_rms.count), float(normalizer.ret_rms.mean), float(normalizer.ret_rms.var))
    started = time.perf_counter()
    rows, records = [], []
    frames = 0
    try:
        write_json(root / "runtime.json", {"pid": os.getpid(), "device": str(model.device),
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "torch_threads": torch.get_num_threads()})
        obs, memory, episode_start = env.reset(), None, np.ones(1, dtype=bool)
        while len(rows) < episodes:
            tick = time.perf_counter()
            action, memory = model.predict(obs, state=memory, episode_start=episode_start, deterministic=True)
            policy_seconds = time.perf_counter() - tick
            old = obs.copy()
            obs, reward, done, info = env.step(action)
            frames += info[0]["native_frames"]
            records.append((old[0], action[0], reward[0], policy_seconds))
            episode_start = done
            if done[0]:
                metrics = raw.get_attr("completed_episodes")[0][-1]
                index = len(rows)
                rows.append({"episode": index, "protocol_id": protocol.identifier, **metrics})
                images, actions, rewards, times = zip(*records, strict=True)
                directory = root / "episodes"
                directory.mkdir(exist_ok=True)
                np.savez_compressed(directory / f"{index:04d}.npz", observation=np.asarray(images),
                    action=np.asarray(actions), reward=np.asarray(rewards), policy_seconds=np.asarray(times),
                    terminal_observation=info[0]["terminal_observation"])
                records = []
                # Flags also reset internally, but clear the real memory explicitly.
                memory = None
        after = (float(normalizer.ret_rms.count), float(normalizer.ret_rms.mean), float(normalizer.ret_rms.var))
        if before != after:
            raise RuntimeError("Evaluation modified frozen normalization")
        pd.DataFrame(rows).to_csv(root / "episodes.csv", index=False)
        write_json(root / "summary.json", {"episodes": episodes,
            "mean_return": float(np.mean([row["return"] for row in rows])),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "normalization_frozen": True, "status": "development_only"})
        write_json(root / "cost.json", {"native_frames": frames, "decisions": frames,
            "reset_physics_ticks": sum(map(len, raw.get_attr("track_seeds"))),
            "reset_note": "Final VecEnv autoreset returns an unused terminal placeholder; no extra track consumed",
            "internal_physics_steps": 0, "wall_seconds": time.perf_counter() - started})
        write_json(root / "status.json", {"status": "complete", "episodes": episodes})
        return root
    except Exception as error:
        write_json(root / "failure.json", {"error": str(error), "native_frames": frames, "episodes": len(rows)})
        write_json(root / "status.json", {"status": "failed", "native_frames": frames, "episodes": len(rows)})
        raise
    finally:
        env.close()
