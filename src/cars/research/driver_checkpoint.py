"""Post-update checkpoints: model, normalization, schedules, RNG and train streams."""
import copy
import json
from pathlib import Path
import random
import shutil

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
import torch

from cars.research.driver_config import NativeSchedule
from cars.training import write_json


def restore_rng(path: Path) -> None:
    state = torch.load(path / "rng.pt", map_location="cpu", weights_only=False)
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, root: Path, model: RecurrentPPO, normalizer: VecNormalize,
                    raw: VecEnv, schedule: NativeSchedule, milestones: list[int], *,
                    checkpoint_method: str = "recurrent_checkpoint") -> None:
    if path != root:
        path.mkdir(parents=True, exist_ok=False)
        configuration = copy.deepcopy(json.loads((root / "config.json").read_text()))
        configuration["method"] = checkpoint_method
        configuration["run_id"] = path.name
        configuration["settings"]["parent_training_run"] = str(root)
        configuration["settings"]["checkpoint_native_frames"] = schedule.frames
        write_json(path / "config.json", configuration)
        shutil.copytree(root / "source", path / "source")
    model.save(path / "model.zip")
    normalizer.save(path / "normalization.pkl")
    torch.save({"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                "python": random.getstate(),
                "cuda": torch.cuda.get_rng_state_all() if model.device.type == "cuda" else None}, path / "rng.pt")
    streams = [stream.state_dict() for stream in raw.get_attr("stream")]
    write_json(path / "streams.json", streams)
    write_json(path / "training_tracks.json", raw.get_attr("track_seeds"))
    write_json(path / "training_episodes.json", raw.get_attr("completed_episodes"))
    write_json(path / "training_state.json", {"native_frames": schedule.frames,
        "decisions": model.num_timesteps, "updates": model._n_updates,
        "learning_rate": schedule(0.), "global_schedule_native_frames": schedule.total,
        "completed_milestones": milestones, "saved_after_optimizer_update": True,
        "reward_normalization_count": float(normalizer.ret_rms.count),
        "normalization_mean": float(normalizer.ret_rms.mean), "normalization_var": float(normalizer.ret_rms.var),
        "resume": "fresh episodes; recurrent/environment state and discounted return accumulators reset"})
