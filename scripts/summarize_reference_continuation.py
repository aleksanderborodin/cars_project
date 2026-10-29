"""Report the declared reference-tail contrast, with historical controls explicit."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cars.env.partitions import Partitions
from cars.research.artifacts import read_config
from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((args.campaign / "config.json").read_text())
    state = json.loads((args.campaign / "status.json").read_text())
    if (state["status"] not in ("evaluations_complete", "complete") or spec["partition"] != "development"
            or spec["held_out_evaluation"] or spec["private_oracle_access"]):
        raise ValueError("Complete public development campaign required")
    if not json.loads((args.campaign / "replay_check.json").read_text())["exact_observation_action_reward_replay"]:
        raise ValueError("Negative-control replay failed")
    for item in spec["required_artifacts"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Required historical artifact changed")
    new_path = Path("runs/policy_geometry_intervention") / spec["new_run_id"]
    table_rows, tables, provenance, trajectory_rows = [], {}, [], []
    for name, path, reference, continuation in (
            ("local_search_context", Path(spec["context_run"]), "reconstructed", "prior"),
            ("oracle_tangent_control", Path(spec["control_run"]), "oracle_visible", "oracle"),
            ("oracle_true_continuation", new_path, "oracle_continued", "oracle")):
        config = read_config(path)
        driver = config["settings"]["driver"]
        if (config["protocol"] != spec["protocol"] or config["versions"] != spec["versions"]
                or driver["actor"]["checkpoint_sha256"] != spec["checkpoint_sha256"]
                or config["settings"]["partition"] != "development" or config["settings"]["user_only_oracle_monitor"]
                or driver["intervention"] != {"reference": reference, "continuation": continuation, "version": 1}):
            raise ValueError("Mismatched condition")
        if path == new_path and config["source_sha256"] != spec["source_sha256"]:
            raise ValueError("New condition did not use frozen sources")
        for key, value in spec["search"].items():
            if driver[key] != value:
                raise ValueError(f"Changed search setting: {key}")
        if driver["simulator"]["vehicle_source_sha256"] != spec["vehicle_source_sha256"]:
            raise ValueError("Vehicle source changed")
        table = pd.read_csv(path / "metrics.csv").sort_values("episode")
        np.testing.assert_array_equal(table.episode, np.arange(spec["episodes"]))
        np.testing.assert_array_equal(table.track_seed, Partitions().seeds("development", spec["episodes"]))
        if not np.isfinite(table["return"]).all():
            raise ValueError("Invalid outcome; retain and investigate failures")
        cost = json.loads((path / "cost.json").read_text())
        timing = json.loads((path / "summary.json").read_text())
        assert cost["native_frames"] == int(table.steps.sum())
        assert cost["internal_physics_frames"] == int(table.decisions.sum()) * spec["candidates"] * spec["horizon"] * spec["action_repeat"]
        speeds, brakes, fallbacks = [], [], 0
        trajectories = []
        for episode in table.episode:
            source = path / f"episodes/{episode:04d}.npz"
            with np.load(source) as data:
                speeds.append(float(np.average(np.linalg.norm(data["observation_ego"][1:, :2], axis=1), weights=data["frames"])))
                brakes.append(float(np.average(data["action"][:, 2], weights=data["frames"])))
                trajectory_rows.append({"condition": name, "episode": int(episode),
                    "sampled_mean_speed_mps": speeds[-1], "mean_brake": brakes[-1],
                    "truncated": bool(data["truncated"][-1])})
            fallbacks += sum(item["visible_reference_fallback"] for item in
                json.loads((path / f"episodes/{episode:04d}_diagnostics.json").read_text()))
            trajectories.append({"path": str(source), "sha256": sha256(source)})
        tables[name] = table
        table_rows.append({"condition": name, "historical_reuse": path != new_path, "episodes": len(table),
            "mean_return": float(table["return"].mean()), "completion": float(table.success.mean()),
            "mean_progress": float(table.progress.mean()), "mean_offroad_frames": float(table.off_track_steps.mean()),
            "sampled_mean_speed_mps": float(np.mean(speeds)), "mean_brake": float(np.mean(brakes)),
            "reference_fallback_decisions": fallbacks, "pipeline_p95_ms": timing["pipeline_latency_p95_ms"], **cost})
        provenance.append({"condition": name, "run": str(path), "config_sha256": sha256(path / "config.json"),
            "metrics_sha256": sha256(path / "metrics.csv"), "cost_sha256": sha256(path / "cost.json"), "trajectories": trajectories})
    contrasts = []
    new = tables["oracle_true_continuation"]
    for reference in ("oracle_tangent_control", "local_search_context"):
        for metric in ("return", "success", "off_track_steps"):
            delta = new[metric].to_numpy(float) - tables[reference][metric].to_numpy(float)
            contrasts.append({"reference": reference, "metric": metric, "mean_difference": float(delta.mean()),
                "per_track": delta.tolist(), "role": "primary" if reference == "oracle_tangent_control" else "context"})
    replay_path = Path("runs/policy_geometry_intervention") / spec["replay_run_id"]
    replay = json.loads((replay_path / "cost.json").read_text())
    new_cost = json.loads((new_path / "cost.json").read_text())
    native = new_cost["native_frames"] + replay["native_frames"]
    internal = new_cost["internal_physics_frames"] + replay["internal_physics_frames"]
    assert native <= spec["native_frame_cap"] and internal <= spec["internal_physics_frame_cap"]
    args.output.mkdir(parents=True, exist_ok=False)
    result = pd.DataFrame(table_rows)
    result.to_csv(args.output / "summary.csv", index=False)
    pd.concat([table.assign(condition=name) for name, table in tables.items()]).to_csv(args.output / "episodes.csv", index=False)
    pd.DataFrame(trajectory_rows).to_csv(args.output / "executed_behavior.csv", index=False)
    write_json(args.output / "analysis.json", {"manifest_sha256": sha256(args.campaign / "config.json"),
        "inputs": provenance, "paired_descriptive_contrasts": contrasts, "new_native_frames_including_replay": native,
        "new_internal_physics_frames_including_replay": internal, "new_condition_cost": new_cost, "verification_cost": replay,
        "new_training_frames": 0, "historical_costs_are_not_new_work": True,
        "speed_definition": "decision-end observed speed weighted by action duration, then equal episode means",
        "latency_limit": "historical controls measured in previous campaign; no paired timing superiority claim",
        "interpretation": "five previously inspected development tracks, one frozen actor/planning stream; no significance, adequacy or SOTA claim",
        "held_out_evaluation": False, "private_oracle_outputs_read": False})
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
