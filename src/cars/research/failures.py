"""Descriptive episode outcomes from public metrics, never candidate oracles."""
import numpy as np


def describe_episode(metrics: dict, arrays: dict, definitions: dict) -> dict:
    frames = np.asarray(arrays["frames"], int)
    offroad = np.asarray(arrays["evaluation_offroad_frame_indices"], int)
    total = int(frames.sum())
    if total != metrics["steps"] or len(offroad) != metrics["off_track_steps"]:
        raise ValueError("Episode frame accounting mismatch")
    if len(offroad) and (offroad[0] < 1 or offroad[-1] > total or np.any(np.diff(offroad) <= 0)):
        raise ValueError("Invalid off-road frame indices")
    end_offroad = bool(len(offroad) and offroad[-1] == total)
    if metrics["success"]:
        outcome = "completed"
    elif metrics["crash"]:
        outcome = "playfield_exit"
    elif metrics.get("lap_time_s") is not None and np.isfinite(float(metrics["lap_time_s"])):
        outcome = "finished_at_time_limit"
    elif bool(arrays["truncated"][-1]):
        outcome = "timeout_offroad" if end_offroad else "timeout_onroad"
    else:
        outcome = "other_termination"
    ends = np.cumsum(frames)
    starts = ends - frames
    tail_start = max(0, total - definitions["tail_native_frames"])
    weights = np.maximum(0, ends - np.maximum(starts, tail_start))
    speeds = np.linalg.norm(np.asarray(arrays["observation_ego"])[1:, :2], axis=1)
    tail_speed = float(np.average(speeds, weights=weights))
    runs = np.split(offroad, np.flatnonzero(np.diff(offroad) != 1) + 1)
    return {"outcome": outcome, "tail_mean_speed": tail_speed,
            "low_speed_tail": tail_speed < definitions["low_speed_mps"],
            "low_progress": metrics["progress"] < definitions["low_progress_fraction"],
            "offroad_at_end": end_offroad,
            "road_contact_restored_by_end": bool(len(offroad) and not end_offroad),
            "longest_offroad_frames": max(map(len, runs)),
            "final_offroad_frames": len(runs[-1]) if end_offroad else 0,
            "offroad_fraction": len(offroad) / total}
