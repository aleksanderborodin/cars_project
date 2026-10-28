"""Summarize one completed, declared training block and public development checks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cars.research.artifacts import read_config
from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("runs/campaigns/driver_first_block_20261028"))
    parser.add_argument("--output", type=Path, default=Path("reports/driver_first_block_2026-10-28"))
    args = parser.parse_args()
    manifest = json.loads((args.campaign / "config.json").read_text())
    status = json.loads((args.campaign / "status.json").read_text())
    if status["status"] != "complete" or manifest["evaluation_partition"] != "development":
        raise ValueError("Only complete development campaigns may be summarized")
    results = json.loads((args.campaign / "checkpoint_results.json").read_text())
    if len(results) != len(manifest["scheduled_checkpoints"]):
        raise ValueError("Missing checkpoints")
    rows, provenance, tracks = [], [], None
    for item in sorted(results, key=lambda row: row["training_native_frames"]):
        root = Path(item["evaluation"])
        config = read_config(root)
        frame = pd.read_csv(root / "metrics.csv").sort_values("track_seed")
        assert config["settings"]["partition"] == "development" and config["protocol"]["view"] == "local"
        assert len(frame) == manifest["development_tracks_per_checkpoint"] and np.isfinite(frame["return"]).all()
        if tracks is None:
            tracks = frame.track_seed.to_numpy()
        np.testing.assert_array_equal(tracks, frame.track_seed)
        checkpoint = Path(item["checkpoint"])
        state = json.loads((checkpoint / "training_state.json").read_text())
        assert state["native_frames"] == item["training_native_frames"]
        assert Path(config["settings"]["driver"]["checkpoint"]).resolve() == checkpoint.resolve()
        rows.append({"training_native_frames": state["native_frames"], "checkpoint": str(checkpoint),
            "mean_return": frame["return"].mean(), "return_std_tracks": frame["return"].std(),
            "completion": frame.success.mean(), "mean_progress": frame.progress.mean(),
            "mean_offroad_frames": frame.off_track_steps.mean(), "learning_rate": state["learning_rate"],
            "reward_normalization_count": state["reward_normalization_count"],
            "evaluation_native_frames": item["evaluation_native_frames"]})
        provenance.append({"evaluation": str(root), "evaluation_config_sha256": sha256(root / "config.json"),
            "metrics_sha256": sha256(root / "metrics.csv"), "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint / "model.zip")})
    training = Path("runs/driver_training") / manifest["training_run_id"]
    cost = json.loads((training / "cost.json").read_text())
    diagnostics = pd.read_json(training / "learning.jsonl", lines=True)
    assert np.isfinite(diagnostics.select_dtypes(include="number").to_numpy()).all()
    table = pd.DataFrame(rows)
    selected = table.sort_values(["mean_return", "completion", "training_native_frames"], ascending=[False, False, True]).iloc[0]
    results = {"manifest_sha256": sha256(args.campaign / "config.json"), "inputs": provenance,
        "selected": selected.to_dict(), "training_cost": cost,
        "evaluation_native_frames": int(table.evaluation_native_frames.sum()),
        "learning_diagnostics_finite": True, "completed_updates": len(diagnostics),
        "all_scheduled_checkpoints_evaluated": True,
        "interpretation": "one training seed, selected on 50 repeatedly used development tracks; competence and convergence require further assessment; no SOTA result",
        "held_out_evaluation": False, "user_only_oracle_values_read": False}
    assert cost["native_frames"] <= manifest["training_frame_cap_including_maximum_rollout_overshoot"]
    assert results["evaluation_native_frames"] <= manifest["evaluation_native_frame_cap"]
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "checkpoints.csv", index=False)
    diagnostics.to_csv(args.output / "learning.csv", index=False)
    write_json(args.output / "analysis.json", results)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    x = table.training_native_frames / 1e6
    axes[0].plot(x, table.mean_return, "o-", color="#237696")
    axes[0].set(ylabel="Mean official return", title="50 shared development tracks")
    axes[1].plot(x, table.completion * 100, "o-", color="#17583b")
    axes[1].set(ylabel="Completion (%)", ylim=(0, 100), title="All scheduled checkpoints")
    for ax in axes:
        ax.set_xlabel("Native training frames (millions)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
    fig.text(.02, .015, "One training seed. Checkpoint selection uses this development panel. Curves are not independent replications.", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(args.output / "learning_curve.png", dpi=160)
    fig.savefig(args.output / "learning_curve.pdf")
    plt.close(fig)
    print(table.to_string(index=False))
    print(json.dumps({k: v for k, v in results.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
