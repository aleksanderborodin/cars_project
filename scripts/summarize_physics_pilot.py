"""Public driving and ego-prediction analysis of the programmed-physics pilot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cars.eval.paired import difference_interval
from cars.models.dynamics import Dynamics
from cars.research.artifacts import read_config
from cars.research.failures import describe_episode
from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/campaigns/programmed-physics-2026-10-28.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/physics_pilot_2026-10-28"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["partition"] != "development" or manifest["user_only_oracle_access"]:
        raise ValueError("Only public development evidence allowed")
    args.output.mkdir(parents=True, exist_ok=True)
    common_tracks = None
    inputs = []

    def read(path: Path) -> pd.DataFrame:
        nonlocal common_tracks
        config = read_config(path)
        if config["protocol"] != manifest["protocol"] or config["settings"]["partition"] != "development":
            raise ValueError("Protocol/partition mismatch")
        frame = pd.read_csv(path / "metrics.csv").sort_values("track_seed")
        if len(frame) != manifest["tracks"] or not np.isfinite(frame["return"]).all() or not (path / "cost.json").exists():
            raise ValueError("Campaign not complete; do not omit episodes")
        if common_tracks is None:
            common_tracks = frame.track_seed.to_numpy()
        np.testing.assert_array_equal(frame.track_seed, common_tracks)
        inputs.append({"path": str(path), "config_sha256": sha256(path / "config.json"),
                       "metrics_sha256": sha256(path / "metrics.csv"), "cost_sha256": sha256(path / "cost.json")})
        return frame

    fixed = read(Path(manifest["references"]["fixed"]))
    fixed60 = read(Path(manifest["references"]["fixed60"]))
    ridge_paths = [Path(x) for x in manifest["references"]["ridge"]]
    ridge_frames = [read(path) for path in ridge_paths]
    ridge_returns = np.stack([frame["return"].to_numpy() for frame in ridge_frames])
    ridge_models = [Dynamics.load(path / "evaluated_dynamics.npz") for path in ridge_paths]
    results, rows, episode_rows, matrices = [], [], [], []
    definitions = {"tail_native_frames": 100, "low_speed_mps": 2., "low_progress_fraction": .5}
    for condition in manifest["conditions"]:
        path = Path("runs/physics_search") / condition["run_id"]
        frame = read(path)
        config = read_config(path)["settings"]["driver"]
        if (config["speeds"] != condition["speeds"] or config["steering_gains"] != manifest["steering_gains"]
                or config["horizon"] != manifest["horizon_decisions"]
                or config["simulator"]["vehicle_source_sha256"] != manifest["vehicle_source_sha256"]):
            raise ValueError("Condition differs from declared comparison")
        predictions, targets, ridge_predictions = [], [], [[] for _ in ridge_models]
        speed_choices = []
        for _, row in frame.iterrows():
            log = path / "episodes" / f'{int(row.episode):04d}.npz'
            diagnostics = json.loads(log.with_name(log.stem + "_diagnostics.json").read_text())
            with np.load(log) as arrays:
                diagnosis = describe_episode(row.to_dict(), arrays, definitions)
                keep = arrays["frames"] == manifest["protocol"]["action_repeat"]
                predictions.append(np.array([d["predicted_next_ego"] for d in diagnostics])[keep])
                targets.append(arrays["observation_ego"][1:][keep])
                ego = arrays["observation_ego"][:-1][keep]
                action = arrays["action"][keep]
                previous = arrays["observation_previous_action"][:-1][keep]
                size = arrays["observation_road"].shape[-1]
                grip = arrays["observation_road"][:-1, size // 2, size // 2][keep]
                for index, model in enumerate(ridge_models):
                    ridge_predictions[index].append(model.predict(ego, action, previous, grip))
            speed_choices.extend(d["target_speed"] for d in diagnostics)
            episode_rows.append({"run_id": condition["run_id"], "episode": row.episode,
                "track_seed": row.track_seed, "return": row["return"], "success": row.success,
                "progress": row.progress, **diagnosis, "trajectory_sha256": sha256(log)})
        actual, predicted = np.concatenate(targets), np.concatenate(predictions)
        values = frame["return"].to_numpy()[None]
        matrices.append(values)
        cost = json.loads((path / "cost.json").read_text())
        timing = json.loads((path / "summary.json").read_text())
        row = {"run_id": condition["run_id"], "mean_return": float(values.mean()),
            "completion": float(frame.success.mean()), "mean_progress": float(frame.progress.mean()),
            "mean_offroad_frames": float(frame.off_track_steps.mean()),
            "pipeline_latency_p50_ms": timing["pipeline_latency_p50_ms"],
            "pipeline_latency_p95_ms": timing["pipeline_latency_p95_ms"],
            "mean_deadline_miss_fraction": float(frame.deadline_miss_fraction.mean()), **cost}
        rows.append(row)
        unique, counts = np.unique(speed_choices, return_counts=True)
        results.append({**row, "minus_fixed50_gain12": difference_interval(values, fixed["return"].to_numpy()[None]),
            "minus_fixed60_gain10": difference_interval(values, fixed60["return"].to_numpy()[None]),
            "minus_ridge_bounded": difference_interval(values, ridge_returns),
            "target_speed_counts": dict(zip(map(str, unique), map(int, counts), strict=True)),
            "one_step_ego_diagnostics": {"rows": len(actual), "units": ["forward_m_s", "left_m_s", "yaw_rad_s"],
                "physics_rmse": np.sqrt(np.mean((predicted - actual)**2, axis=0)).tolist(),
                "ridge_rmse_on_same_transitions": [np.sqrt(np.mean((np.concatenate(p) - actual)**2, axis=0)).tolist()
                                                   for p in ridge_predictions],
                "interpretation": "same actual actions/transitions per condition; ordinary permitted ego observations; no candidate oracle; not long-horizon proof"}})
    analysis = {"manifest_sha256": sha256(args.manifest), "inputs": inputs, "conditions": results,
        "wide_minus_bounded": difference_interval(matrices[1], matrices[0]),
        "native_frames": sum(row["native_frames"] for row in rows),
        "internal_physics_frames": sum(row["internal_physics_frames"] for row in rows),
        "new_training_frames": 0, "held_out_evaluation": False, "user_only_oracle_values_read": False,
        "interpretation": "adaptive development; one deterministic programmed planner per condition on 20 common tracks; no SOTA/novelty claim"}
    if (analysis["native_frames"] > manifest["evaluation_native_frame_cap"]
            or analysis["internal_physics_frames"] > manifest["maximum_internal_physics_frames"]):
        raise ValueError("Declared budget exceeded")
    pd.DataFrame(rows).to_csv(args.output / "summary.csv", index=False)
    episode_table = pd.DataFrame(episode_rows)
    episode_table.to_csv(args.output / "episodes.csv", index=False)
    episode_table.groupby(["run_id", "outcome"]).size().rename("episodes").reset_index().to_csv(
        args.output / "outcomes.csv", index=False)
    write_json(args.output / "analysis.json", analysis)
    labels = ["Fixed 50 / gain 1.2", "Ridge search (3-model mean)", "Coded physics / bounded", "Coded physics / wide"]
    means = [fixed["return"].mean(), ridge_returns.mean(), *[r["mean_return"] for r in rows]]
    completion = [fixed.success.mean(), np.mean([frame.success.mean() for frame in ridge_frames]),
                  *[r["completion"] for r in rows]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = ["#777777", "#a46b39", "#237696", "#17583b"]
    axes[0].barh(labels, means, color=colors)
    axes[0].set(xlabel="Mean official return", title="Development driving")
    axes[0].invert_yaxis()
    axes[1].barh(labels, np.array(completion) * 100, color=colors)
    axes[1].set(xlabel="Completion (%)", xlim=(0, 100), title="All scheduled episodes")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.02, .015, "20 common development tracks. Earlier constant selected on this panel. Programmed physics uses approximate state/contact reconstruction.", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(args.output / "physics_pilot.png", dpi=160)
    fig.savefig(args.output / "physics_pilot.pdf")
    plt.close(fig)
    print(json.dumps({k: v for k, v in analysis.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
