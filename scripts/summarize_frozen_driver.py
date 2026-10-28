"""Report the complete frozen-policy development campaign, including all controls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cars.eval.paired import paired_checkpoint_interval
from cars.research.artifacts import read_config
from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("runs/campaigns/frozen_driver_20261028"))
    parser.add_argument("--output", type=Path, default=Path("reports/frozen_driver_2026-10-28"))
    args = parser.parse_args()
    spec = json.loads((args.campaign / "config.json").read_text())
    status = json.loads((args.campaign / "status.json").read_text())
    if (status["status"] not in ("evaluations_complete", "complete") or spec["partition"] != "development"
            or spec["held_out_evaluation"] or spec["user_only_oracle_access"]):
        raise ValueError("Require a complete public development campaign")
    expected_hash = spec["checkpoint_sha256"]
    if sha256(Path(spec["checkpoint"]) / "model.zip") != expected_hash:
        raise ValueError("Frozen checkpoint changed")
    tables, rows, provenance, tracks = {}, [], [], None
    for condition in spec["conditions"]:
        root = Path("runs") / condition["directory"] / condition["run_id"]
        config = read_config(root)
        driver = config["settings"]["driver"]
        actor = driver.get("actor", driver)
        if (config["settings"]["partition"] != "development" or config["protocol_id"] != spec["protocol_id"]
                or config["partition_sha256"] != spec["partition_sha256"]
                or actor["checkpoint_sha256"] != expected_hash
                or config["settings"]["user_only_oracle_monitor"]):
            raise ValueError("Unmatched checkpoint, protocol or observation access")
        frame = pd.read_csv(root / "metrics.csv").sort_values("track_seed")
        if len(frame) != spec["tracks_per_condition"] or not np.isfinite(frame["return"]).all():
            raise ValueError("Incomplete condition; do not silently drop failures")
        if tracks is None:
            tracks = frame.track_seed.to_numpy()
        np.testing.assert_array_equal(tracks, frame.track_seed)
        summary = json.loads((root / "summary.json").read_text())
        cost = json.loads((root / "cost.json").read_text())
        diagnostics = [item for ep in sorted(root.glob("episodes/*_diagnostics.json"))
                       for item in json.loads(ep.read_text())]
        if len(diagnostics) != cost["decisions"]:
            raise ValueError("Missing decision diagnostics")
        override = float(np.mean([item.get("overrode_base", False) for item in diagnostics]))
        tables[condition["name"]] = frame
        rows.append({"condition": condition["name"], "run": str(root),
            "mean_return": float(frame["return"].mean()), "completion": float(frame.success.mean()),
            "mean_progress": float(frame.progress.mean()), "mean_offroad_frames": float(frame.off_track_steps.mean()),
            "playfield_exits": int(frame.crash.sum()), "override_fraction": override,
            "pipeline_p50_ms": summary["pipeline_latency_p50_ms"],
            "pipeline_p95_ms": summary["pipeline_latency_p95_ms"],
            "episode_mean_deadline_miss_fraction": float(frame.deadline_miss_fraction.mean()), **cost})
        provenance.append({"condition": condition["name"], "config_sha256": sha256(root / "config.json"),
            "metrics_sha256": sha256(root / "metrics.csv"), "checkpoint_sha256": actor["checkpoint_sha256"]})
    differences = []
    reference_names = [("search7", "direct"), ("search21", "direct"), ("search21", "search7")]
    for metric in ["return", "success"]:
        values = {name: frame[metric].to_numpy(dtype=float)[None] for name, frame in tables.items()}
        values["random7_mean"] = np.mean([values[f"random7_s{s}"] for s in range(3)], axis=0)
        for candidate, reference in reference_names + [("search7", "random7_mean")]:
            differences.append({"candidate": candidate, "reference": reference, "metric": metric,
                **paired_checkpoint_interval(values[candidate], values[reference], reps=5000, seed=0),
                "uncertainty_scope": "conditional on one selected checkpoint and fixed planning streams; tracks only"})
    table = pd.DataFrame(rows)
    native, internal = int(table.native_frames.sum()), int(table.internal_physics_frames.sum())
    if native > spec["native_frame_cap"] or internal > spec["internal_physics_frame_cap"]:
        raise ValueError("Resource cap exceeded")
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "summary.csv", index=False)
    pd.concat([frame.assign(condition=name) for name, frame in tables.items()]).to_csv(args.output / "episodes.csv", index=False)
    write_json(args.output / "analysis.json", {"manifest_sha256": sha256(args.campaign / "config.json"),
        "checkpoint_sha256": expected_hash, "inputs": provenance, "paired_differences": differences,
        "native_frames": native, "internal_physics_frames": internal,
        "actor_forward_examples": int(table.actor_forward_examples.sum()),
        "training_cost": "reused selected checkpoint; all training and checkpoint screening costs remain in driver-first-block report",
        "interpretation": "exploratory adaptive development panel; all six conditions retained; no convergence or SOTA claim",
        "planning_repeat_scope": "three random streams shared across tracks; averaged per track for the random contrast, no planner-seed population interval",
        "held_out_evaluation": False, "user_only_oracle_values_read": False})
    labels = ["Direct", "Random 0", "Random 1", "Random 2", "Search 7", "Search 21"]
    colors = ["#6c7d86", "#b9b3a9", "#b9b3a9", "#b9b3a9", "#237696", "#17583b"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, field, title in zip(axes, ["mean_return", "completion", "pipeline_p95_ms"],
                               ["Mean official return", "Completion (%)", "Pipeline latency p95 (ms)"], strict=True):
        scale = 100 if field == "completion" else 1
        ax.bar(labels, table[field] * scale, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=40)
        ax.spines[["top", "right"]].set_visible(False)
        if field == "completion":
            ax.set_ylim(0, 100)
        if field == "pipeline_p95_ms":
            ax.axhline(80, linestyle="--", color="#a14124", linewidth=1, label="80 ms target")
            ax.legend(frameon=False, fontsize=8)
    fig.text(.02, .015, "One frozen checkpoint, 20 shared development tracks. Random controls use three fixed planning streams. No deadline fallback.", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(args.output / "comparison.png", dpi=160)
    fig.savefig(args.output / "comparison.pdf")
    plt.close(fig)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
