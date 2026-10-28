"""Common development tracks, observation-only policy calls, full cost records."""
from pathlib import Path
import time
from typing import Protocol

import numpy as np
import pandas as pd

from cars.env.partitions import Partitions, random_stream
from cars.env.protocol import DrivingEnv, Observation, ProtocolConfig
from cars.env.state import CarState
from cars.models.dynamics import Dynamics
from cars.policies.baseline import BaselinePolicy
from cars.research.artifacts import begin_run, read_config
from cars.training import sha256, write_json


class Driver(Protocol):
    method: str
    def reset(self) -> None: ...
    def act(self, observation: Observation) -> np.ndarray: ...
    def configuration(self) -> dict: ...


def evaluate_driver(driver: Driver | BaselinePolicy, run_id: str, *, episodes: int = 20,
                    protocol: ProtocolConfig | None = None, runs_dir: Path = Path("runs"),
                    user_monitor: bool = False) -> Path:
    protocol = protocol or ProtocolConfig()
    if not 0 < episodes <= Partitions().development_count:
        raise ValueError("Invalid development episode count")
    privileged = isinstance(driver, BaselinePolicy)
    if user_monitor and not isinstance(getattr(driver, "model", None), Dynamics):
        raise ValueError("Frozen-candidate monitoring currently supports model search only")
    method = "global_expert" if privileged else driver.method
    root = begin_run(method, run_id, protocol,
                     {"partition": "development", "episodes": episodes,
                      "driver": driver.configuration(), "privileged_reference": privileged or protocol.view == "global",
                      "user_only_oracle_monitor": user_monitor,
                      "checkpoint_selection": "fixed_before_this_development_pass"}, runs_dir)
    if isinstance(getattr(driver, "model", None), Dynamics):
        driver.model.save(root / "evaluated_dynamics.npz")
    monitor = None
    if user_monitor:
        from cars.monitoring.bridge import UserMonitor
        monitor = UserMonitor(run_id, root, protocol)
    env = DrivingEnv(protocol, "development")
    rows = []
    all_latency = []
    total_frames = transitions = physics_frames = actor_examples = 0
    started = time.perf_counter()
    try:
        for episode in range(episodes):
            obs, _ = env.reset()
            if privileged:
                driver.reset(env._sim.unwrapped.track_info)
            else:
                driver.reset()
            records: dict[str, list] = {key: [] for key in ("action", "reward", "frames", "latency_ms", "sensor_ms", "policy_ms", "terminated", "truncated")}
            observations: dict[str, list] = {key: [value.copy()] for key, value in obs.items()}
            decision_diagnostics = []
            model_work = 0
            while True:
                sensor_ms = env.sensor_seconds * 1000
                start = time.perf_counter()
                if privileged:
                    action = driver.act(CarState.from_env(env._sim), np.empty(0, np.float32))
                else:
                    action = driver.act(obs)
                policy_ms = (time.perf_counter() - start) * 1000
                latency = policy_ms + sensor_ms
                model_work += getattr(driver, "diagnostics", {}).get("model_transitions", 0)
                physics_frames += getattr(driver, "diagnostics", {}).get("internal_physics_frames", 0)
                actor_examples += getattr(driver, "diagnostics", {}).get("actor_forward_examples", 0)
                decision_diagnostics.append(dict(getattr(driver, "diagnostics", {})))
                if monitor is not None:
                    monitor.capture(episode, len(records["action"]), track_seed=env.track_seeds[-1],
                        prefix=records["action"], prefix_frames=records["frames"], observation=obs,
                        candidates=driver.candidate_actions, predicted_costs=driver.candidate_costs,
                        selected=driver.diagnostics["selected_candidate"])
                obs, reward, terminated, truncated, info = env.step(action)
                for key, value in {"action": action, "reward": reward, "frames": info["native_frames"],
                                   "latency_ms": latency, "sensor_ms": sensor_ms, "policy_ms": policy_ms,
                                   "terminated": terminated, "truncated": truncated}.items():
                    records[key].append(value)
                for key, value in obs.items():
                    observations[key].append(value.copy())
                total_frames += info["native_frames"]
                if terminated or truncated:
                    break
            latency = np.asarray(records["latency_ms"])
            all_latency.extend(latency.tolist())
            transitions += model_work
            row = {"method": method, "run_id": run_id, "protocol_id": protocol.identifier,
                   "information": "global_privileged" if privileged else ("global_geometry" if protocol.view == "global" else protocol.view),
                   "partition": "development", "partition_sha256": Partitions().fingerprint,
                   "episode": episode, "track_seed": env.track_seeds[-1],
                   **env.last_metrics, "decisions": len(latency), "model_transitions": model_work,
                   "latency_p50_ms": float(np.median(latency)), "latency_p95_ms": float(np.quantile(latency, .95)),
                   "deadline_miss_fraction": float(np.mean(latency > protocol.action_repeat * 20))}
            rows.append(row)
            logs = root / "episodes"
            logs.mkdir(exist_ok=True)
            arrays = {key: np.asarray(value) for key, value in records.items()}
            arrays.update({f"observation_{key}": np.asarray(value, dtype=np.uint8 if key in ("road", "visible", "image") else np.float32)
                           for key, value in observations.items()})
            # Post-episode evaluator metric; never part of the driver's observation.
            arrays["evaluation_offroad_frame_indices"] = np.asarray(
                env._sim.get_wrapper_attr("off_track_step_indices"), dtype=np.int32)
            np.savez_compressed(logs / f"{episode:04d}.npz", **arrays)
            write_json(logs / f"{episode:04d}_diagnostics.json", decision_diagnostics)
            pd.DataFrame(rows).to_csv(root / "metrics.csv", index=False)
            print(f"{method}: {episode + 1}/{episodes}; return={row['return']:.1f}; completion={row['success']}", flush=True)
        write_json(root / "summary.json", {"episodes": episodes, "mean_return": float(np.mean([x["return"] for x in rows])),
            "completion": float(np.mean([x["success"] for x in rows])),
            "mean_progress": float(np.mean([x["progress"] for x in rows])),
            "pipeline_latency_p50_ms": float(np.median(all_latency)),
            "pipeline_latency_p95_ms": float(np.quantile(all_latency, .95)),
            "pipeline_latency_p99_ms": float(np.quantile(all_latency, .99)),
            "latency_scope": "sensor construction plus policy; excludes physics, disk IO and repeat-frame unused renders",
            "status": "engineering_development_result_not_confirmatory"})
        write_json(root / "cost.json", {"native_frames": total_frames, "decisions": len(all_latency),
                                      "model_transitions": transitions, "internal_physics_frames": physics_frames,
                                      "actor_forward_examples": actor_examples,
                                      "wall_time_s": time.perf_counter() - started})
    except Exception as error:
        write_json(root / "failure.json", {"type": type(error).__name__, "message": str(error),
                                           "completed_episodes": len(rows), "native_frames": total_frames})
        raise
    finally:
        env.close()
        if monitor is not None:
            monitor.close()
    return root


def paired_report(paths: list[Path], output: Path) -> None:
    frames = [pd.read_csv(path / "metrics.csv") for path in paths]
    for path, frame in zip(paths, frames, strict=True):
        configuration = read_config(path)
        if len(frame) != configuration["settings"]["episodes"] or not np.isfinite(frame["return"]).all():
            raise ValueError("Incomplete/invalid campaign; do not silently omit failures")
        if frame.protocol_id.nunique() != 1 or frame.partition_sha256.nunique() != 1:
            raise ValueError("Mixed protocols in one artifact")
    local = [frame for frame in frames if (frame["information"] == "local").all()]
    if local and (len({frame["protocol_id"].iloc[0] for frame in local}) != 1 or
                  len({frame["partition_sha256"].iloc[0] for frame in local}) != 1):
        raise ValueError("Cannot pool incompatible experiments")
    summaries = [{"path": str(path), "method": frame.method.iloc[0], "run_id": frame.run_id.iloc[0], "information": frame.information.iloc[0],
                  "episodes": len(frame), "mean_return": float(frame["return"].mean()),
                  "completion": float(frame.success.mean())} for path, frame in zip(paths, frames, strict=True)]
    differences = []
    if local:
        reference = local[0]
        for frame in local[1:]:
            paired = reference.merge(frame, on="track_seed", suffixes=("_reference", "_candidate"), validate="one_to_one")
            if len(paired) != len(reference) or len(paired) != len(frame):
                raise ValueError("Paired development sets differ")
            delta = (paired["return_candidate"] - paired["return_reference"]).to_numpy()
            rng = random_stream(0, "report_track_bootstrap")
            draws = rng.choice(delta, (5000, len(delta)), replace=True).mean(axis=1)
            differences.append({"reference": reference.method.iloc[0], "candidate": frame.method.iloc[0],
                "reference_run": reference.run_id.iloc[0], "candidate_run": frame.run_id.iloc[0],
                "mean_difference": float(delta.mean()), "track_bootstrap_95_percent": np.quantile(draws, [.025, .975]).tolist(),
                "uncertainty_scope": "paired development tracks only; no training-seed uncertainty"})
    write_json(output, {"summaries": summaries, "paired_differences": differences,
        "inputs": [{"path": str(path), "metrics_sha256": sha256(path / "metrics.csv"),
                    "config_sha256": sha256(path / "config.json")} for path in paths],
        "analysis": {"track_bootstrap_reps": 5000, "random_seed": 0, "metric": "mean_official_return",
                     "tuning_selection_adjusted": False},
        "status": "exploratory; single trained instance; no SOTA inference"})
