"""Post-hoc executed-trajectory audit; no candidate-oracle or map reads."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cars.research.failures import describe_episode
from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((args.campaign / "config.json").read_text())
    analysis = json.loads((args.report / "analysis.json").read_text())
    if (json.loads((args.campaign / "status.json").read_text())["status"] != "complete"
            or spec["partition"] != "development" or spec["held_out_evaluation"] or spec["user_only_oracle_access"]
            or analysis["manifest_sha256"] != sha256(args.campaign / "config.json")):
        raise ValueError("Matching completed public development campaign required")
    if (args.report / "trajectory_audit.json").exists():
        raise FileExistsError("Do not overwrite a completed audit")
    definitions_path = Path("configs/campaigns/control-audit-2026-10-28.json")
    definitions = json.loads(definitions_path.read_text())["failure_definitions"]
    rows, provenance = [], []
    keys = ("frames", "evaluation_offroad_frame_indices", "observation_ego", "truncated", "terminated", "action")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=True)
    colors = ("#237696", "#c87930", "#9a456b", "#3c8063")
    for cell, color, verified in zip(spec["conditions"], colors, analysis["inputs"], strict=True):
        root = Path("runs/policy_geometry_intervention") / cell["run_id"]
        if str(root) != verified["run"] or sha256(root / "metrics.csv") != verified["metrics_sha256"]:
            raise ValueError("Changed episode metrics")
        metrics = pd.read_csv(root / "metrics.csv").sort_values("episode")
        for row in metrics.to_dict("records"):
            episode = int(row["episode"])
            path = root / f"episodes/{episode:04d}.npz"
            with np.load(path) as data:
                arrays = {key: data[key] for key in keys}
            diagnosis = describe_episode(row, arrays, definitions)
            speed = np.linalg.norm(arrays["observation_ego"][1:, :2], axis=1)
            weights, actions = arrays["frames"], arrays["action"]
            rows.append({"condition": cell["name"], "episode": episode,
                "return": row["return"], "progress": row["progress"], **diagnosis,
                "sampled_mean_speed_mps": float(np.average(speed, weights=weights)),
                "mean_gas": float(np.average(actions[:, 1], weights=weights)),
                "mean_brake": float(np.average(actions[:, 2], weights=weights))})
            provenance.append({"path": str(path), "sha256": sha256(path)})
            ax = axes.flat[episode]
            ax.plot(np.cumsum(weights) / 50, speed, color=color, linewidth=1, alpha=.85,
                    label=cell["name"].replace("_", " "))
            ax.set_title(f"Development track {episode + 1}")
            ax.set(xlabel="Episode time (s)", ylabel="Observed speed (m/s)", xlim=(0, 20))
            ax.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="upper left", frameon=False)
    axes.flat[-1].text(.03, .35, "Five common tracks, one frozen actor.\nPost-hoc executed-trajectory description.\nOracle arms: on-road timeouts.\nNo private candidate-oracle data.",
                       transform=axes.flat[-1].transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.report / "executed_speeds.png", dpi=160)
    fig.savefig(args.report / "executed_speeds.pdf")
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(args.report / "trajectory_audit_episodes.csv", index=False)
    summary = table.groupby("condition", sort=False).agg(episodes=("episode", "size"),
        mean_progress=("progress", "mean"), sampled_mean_speed_mps=("sampled_mean_speed_mps", "mean"),
        mean_tail_speed_mps=("tail_mean_speed", "mean"), mean_gas=("mean_gas", "mean"),
        mean_brake=("mean_brake", "mean"), low_speed_tail_episodes=("low_speed_tail", "sum"))
    summary.to_csv(args.report / "trajectory_audit_summary.csv")
    write_json(args.report / "trajectory_audit.json", {"analysis": "post_hoc; descriptive; no causal inference or statistical test",
        "main_report_sha256": sha256(args.report / "analysis.json"), "script_sha256": sha256(Path(__file__)),
        "inputs": provenance, "read_npz_fields": keys, "failure_definitions": definitions,
        "definitions_source": str(definitions_path), "definitions_source_sha256": sha256(definitions_path),
        "speed_estimator": "decision-end observed body-speed samples weighted by actual action duration; equal episode weighting in condition means",
        "new_environment_frames": 0, "private_oracle_outputs_read": False,
        "outcomes": table.groupby(["condition", "outcome"]).size().rename("episodes").reset_index().to_dict("records")})
    print(summary.to_string())


if __name__ == "__main__":
    main()
