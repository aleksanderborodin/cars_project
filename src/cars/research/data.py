"""Collect permitted transitions and fit an ordinary supervised dynamics model."""
from dataclasses import asdict
from pathlib import Path
import json
import time

import numpy as np

from cars.env.partitions import Partitions, random_stream
from cars.env.protocol import DrivingEnv, ProtocolConfig
from cars.models.dynamics import Dynamics
from cars.planning.local import FixedBehavior, LocalPursuit
from cars.research.artifacts import begin_run, read_config
from cars.training import sha256, write_json


def collect(run_id: str, *, partition: str = "train", episodes: int = 20, seed: int = 0,
            noise: float = .08, protocol: ProtocolConfig | None = None,
            runs_dir: Path = Path("runs"), collector: str = "local") -> Path:
    protocol = protocol or ProtocolConfig()
    if protocol.view != "local" or partition not in ("train", "development") or episodes < 1 or noise < 0 or collector not in ("local", "mixed"):
        raise ValueError("Collection requires local train/development episodes")
    if partition == "development" and episodes > Partitions().development_count:
        raise ValueError("Development partition capacity exceeded")
    root = begin_run("collection", run_id, protocol,
                     {"partition": partition, "episodes": episodes, "seed": seed,
                      "noise": noise, "collector": collector,
                      "mixed_schedule": ["local", "fixed40", "fixed60", "fixed80"] if collector == "mixed" else None}, runs_dir)
    env = DrivingEnv(protocol, partition, seed=seed)
    policy = LocalPursuit(protocol)
    rng = random_stream(seed, "collection_action_noise")
    columns: dict[str, list] = {key: [] for key in ("ego", "action", "previous", "grip", "next_ego",
        "frames", "reward", "terminated", "truncated", "track_seed", "episode", "road", "next_road")}
    summary = []
    frames = 0
    start = time.perf_counter()
    try:
        for episode in range(episodes):
            obs, _ = env.reset()
            policy = (FixedBehavior(protocol, (40., 60., 80.)[(episode % 4) - 1])
                      if collector == "mixed" and episode % 4 else LocalPursuit(protocol))
            policy.reset()
            while True:
                action = policy.act(obs)
                if noise:
                    action = np.clip(action + rng.normal(0, [noise, noise, noise / 3]), [-1, 0, 0], [1, 1, 1]).astype(np.float32)
                nxt, reward, terminated, truncated, info = env.step(action)
                values = {"ego": obs["ego"], "action": action, "previous": obs["previous_action"],
                          "grip": obs["road"][protocol.size // 2, protocol.size // 2], "next_ego": nxt["ego"],
                          "frames": info["native_frames"], "reward": reward, "terminated": terminated,
                          "truncated": truncated, "track_seed": env.track_seeds[-1], "episode": episode,
                          "road": obs["road"].astype(np.uint8), "next_road": nxt["road"].astype(np.uint8)}
                for key, value in values.items():
                    columns[key].append(value)
                frames += info["native_frames"]
                obs = nxt
                if terminated or truncated:
                    break
            summary.append({"episode": episode, "track_seed": env.track_seeds[-1], **env.last_metrics})
            print(f"collection {partition}: {episode + 1}/{episodes}; native frames={frames}", flush=True)
        np.savez_compressed(root / "dataset.npz", **{key: np.asarray(value) for key, value in columns.items()})
        write_json(root / "episodes.json", summary)
        write_json(root / "stream.json", env.stream.state_dict())
        write_json(root / "cost.json", {"native_frames": frames, "decisions": len(columns["action"]),
                                       "expert_labels": 0, "truth_labels": 0,
                                       "wall_time_s": time.perf_counter() - start,
                                       "dataset_bytes": (root / "dataset.npz").stat().st_size,
                                       "dataset_sha256": sha256(root / "dataset.npz")})
    finally:
        env.close()
    return root


def fit(run_id: str, training: Path, development: Path, *, ridge: float = .01,
        runs_dir: Path = Path("runs")) -> Path:
    if ridge <= 0:
        raise ValueError("Positive ridge penalty required")
    train_config, dev_config = read_config(training), read_config(development)
    for source in (training, development):
        if sha256(source / "dataset.npz") != json.loads((source / "cost.json").read_text())["dataset_sha256"]:
            raise ValueError("Collection artifact was modified after completion")
    if train_config["settings"]["partition"] != "train" or dev_config["settings"]["partition"] != "development":
        raise ValueError("Fitting and diagnostics require separate train/development artifacts")
    if train_config["protocol_id"] != dev_config["protocol_id"]:
        raise ValueError("Collection protocols differ")
    protocol = ProtocolConfig(**train_config["protocol"])
    partitions = Partitions()
    root = begin_run("dynamics", run_id, protocol, {"training": str(training.resolve()),
        "development": str(development.resolve()), "ridge": ridge,
        "training_data_sha256": sha256(training / "dataset.npz"),
        "development_data_sha256": sha256(development / "dataset.npz")}, runs_dir)
    start = time.perf_counter()
    with np.load(training / "dataset.npz", allow_pickle=False) as data:
        if not all(partitions.contains("train", int(seed)) for seed in np.unique(data["track_seed"])):
            raise ValueError("Non-training track in fit data")
        keep = data["frames"] == protocol.action_repeat
        model = Dynamics.fit(data["ego"][keep], data["action"][keep], data["previous"][keep],
                             data["grip"][keep], data["next_ego"][keep], repeat=protocol.action_repeat, ridge=ridge)
        fitting_rows = int(keep.sum())
    model.save(root / "model.npz")
    restored = Dynamics.load(root / "model.npz")
    with np.load(development / "dataset.npz", allow_pickle=False) as data:
        if not all(partitions.contains("development", int(seed)) for seed in np.unique(data["track_seed"])):
            raise ValueError("Invalid development diagnostic data")
        keep = data["frames"] == protocol.action_repeat
        predictions = restored.predict(data["ego"][keep], data["action"][keep], data["previous"][keep], data["grip"][keep])
        reference = model.predict(data["ego"][keep], data["action"][keep], data["previous"][keep], data["grip"][keep])
        np.testing.assert_array_equal(predictions, reference)
        rmse = np.sqrt(np.mean((predictions - data["next_ego"][keep])**2, axis=0))
        persistence = np.sqrt(np.mean((data["ego"][keep] - data["next_ego"][keep])**2, axis=0))
        multistep = {}
        for horizon in (5, 10, 20):
            errors = []
            for first in range(0, len(data["ego"]) - horizon + 1, horizon):
                end = first + horizon
                if data["episode"][first] != data["episode"][end - 1] or not np.all(keep[first:end]):
                    continue
                ego = data["ego"][first:first + 1].copy()
                for index in range(first, end):
                    ego = restored.predict(ego, data["action"][index:index + 1],
                                           data["previous"][index:index + 1], data["grip"][index:index + 1])
                errors.append((ego[0] - data["next_ego"][end - 1])**2)
            multistep[str(horizon)] = {"windows": len(errors), "ego_rmse": np.sqrt(np.mean(errors, axis=0)).tolist() if errors else None,
                "caveat": "recorded actions and observed grip supplied; not closed-loop candidate-ranking validation"}
    write_json(root / "diagnostics.json", {"ego_units": ["forward_m_s", "left_m_s", "yaw_rad_s"],
        "one_step_rmse": rmse.tolist(), "persistence_rmse": persistence.tolist(),
        "multistep": multistep, "save_reload_equal": True})
    write_json(root / "cost.json", {"fit_rows": fitting_rows, "new_native_frames": 0,
                                   "wall_time_s": time.perf_counter() - start})
    return root
