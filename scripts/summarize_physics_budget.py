"""Matched wide-candidate reference and reduced-budget simulation search."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cars.eval.paired import difference_interval
from cars.research.artifacts import read_config
from cars.research.failures import describe_episode
from cars.training import sha256, write_json


def main() -> None:
    manifest_path = Path("configs/campaigns/physics-budget-2026-10-28.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["partition"] != "development" or manifest["user_only_oracle_access"]:
        raise ValueError("Development-only analysis")
    output = Path("reports/physics_budget_2026-10-28")
    output.mkdir(parents=True, exist_ok=True)
    common_tracks = None
    provenance = []

    def read(path: Path) -> pd.DataFrame:
        nonlocal common_tracks
        config = read_config(path)
        if config["protocol"] != manifest["protocol"] or config["settings"]["partition"] != "development":
            raise ValueError("Incompatible experiment")
        frame = pd.read_csv(path / "metrics.csv").sort_values("track_seed")
        if len(frame) != manifest["tracks"] or not np.isfinite(frame["return"]).all() or not (path / "cost.json").exists():
            raise ValueError("Incomplete campaign")
        if common_tracks is None:
            common_tracks = frame.track_seed.to_numpy()
        np.testing.assert_array_equal(frame.track_seed, common_tracks)
        provenance.append({"path": str(path), "config_sha256": sha256(path / "config.json"),
                           "metrics_sha256": sha256(path / "metrics.csv")})
        return frame

    fixed_rows, fixed_frames = [], {}
    for candidate in manifest["all_fixed_candidates"]:
        path = Path(candidate["path"])
        frame = read(path)
        params = read_config(path)["settings"]["driver"]
        if params["speed"] != candidate["speed"] or params["gain"] != candidate["gain"]:
            raise ValueError("Reference parameters differ")
        fixed_frames[candidate["candidate_index"]] = frame
        fixed_rows.append({**candidate, "mean_return": frame["return"].mean(), "completion": frame.success.mean()})
    fixed_table = pd.DataFrame(fixed_rows).sort_values(
        ["mean_return", "completion", "candidate_index"], ascending=[False, False, True])
    selected = fixed_table.iloc[0]
    reference = fixed_frames[int(selected.candidate_index)]
    spec = manifest["physics_condition"]
    root = Path("runs/physics_search") / spec["run_id"]
    small = read(root)
    old = read(Path(manifest["previous_search"]))
    params = read_config(root)["settings"]["driver"]
    if params["speeds"] != spec["speeds"] or params["steering_gains"] != spec["gains"] or params["horizon"] != spec["horizon"]:
        raise ValueError("Reduced search differs from declaration")
    definitions = {"tail_native_frames": 100, "low_speed_mps": 2., "low_progress_fraction": .5}
    episodes = []
    for _, row in small.iterrows():
        path = root / "episodes" / f'{int(row.episode):04d}.npz'
        with np.load(path) as arrays:
            diagnosis = describe_episode(row.to_dict(), arrays, definitions)
        episodes.append({"episode": row.episode, "track_seed": row.track_seed, "return": row["return"],
                         "success": row.success, "progress": row.progress, **diagnosis})
    cost = json.loads((root / "cost.json").read_text())
    fixed_costs = [json.loads((Path("runs/fixed_behavior") / job["run_id"] / "cost.json").read_text())
                   for job in manifest["new_fixed_jobs"]]
    timing = json.loads((root / "summary.json").read_text())
    contrasts = {}
    for name, frame in (("seven_minus_21", old), ("seven_minus_best_fixed", reference)):
        contrasts[name] = difference_interval(small["return"].to_numpy()[None], frame["return"].to_numpy()[None])
        contrasts[name + "_completion"] = difference_interval(small.success.to_numpy()[None], frame.success.to_numpy()[None])
    contrasts["21_minus_best_fixed"] = difference_interval(old["return"].to_numpy()[None], reference["return"].to_numpy()[None])
    analysis = {"manifest_sha256": sha256(manifest_path), "inputs": provenance,
        "selected_fixed": selected.to_dict(),
        "seven_candidate_search": {"mean_return": float(small["return"].mean()), "completion": float(small.success.mean()),
            "pipeline_latency_p50_ms": timing["pipeline_latency_p50_ms"],
            "pipeline_latency_p95_ms": timing["pipeline_latency_p95_ms"],
            "mean_deadline_miss_fraction": float(small.deadline_miss_fraction.mean()), **cost},
        "contrasts": contrasts,
        "cost": {"native_frames": cost["native_frames"] + sum(c["native_frames"] for c in fixed_costs),
            "internal_physics_frames": cost["internal_physics_frames"], "new_training_frames": 0},
        "interpretation": "adaptive development; gain chosen from prior audit; best fixed selected on these same tracks; intervals not selection-adjusted",
        "held_out_evaluation": False, "user_only_oracle_values_read": False}
    if (analysis["cost"]["native_frames"] > manifest["evaluation_native_frame_cap"] or
            cost["internal_physics_frames"] > manifest["maximum_internal_physics_frames"]):
        raise ValueError("Declared budget exceeded")
    fixed_table.to_csv(output / "fixed_candidates.csv", index=False)
    pd.DataFrame(episodes).to_csv(output / "episodes.csv", index=False)
    write_json(output / "analysis.json", analysis)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    labels = ["Best fixed candidate", "21-candidate physics", "7-candidate physics"]
    means = [selected.mean_return, old["return"].mean(), small["return"].mean()]
    completion = [selected.completion, old.success.mean(), small.success.mean()]
    colors = ["#777777", "#237696", "#17583b"]
    axes[0].bar(labels, means, color=colors)
    axes[0].set(ylabel="Mean official return", title="Same 20 development tracks")
    axes[1].bar(labels, np.array(completion) * 100, color=colors)
    axes[1].set(ylabel="Completion (%)", ylim=(0, 100), title="All episodes retained")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.02, .015, "Adaptive development. Fixed controller selected on this panel. Reduced search also changes steering-gain coverage.", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(output / "physics_budget.png", dpi=160)
    fig.savefig(output / "physics_budget.pdf")
    plt.close(fig)
    print(json.dumps({k: v for k, v in analysis.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
