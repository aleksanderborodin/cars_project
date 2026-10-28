"""Summarize all four complete conditions on the fixed 200-track development panel."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from cars.env.partitions import Partitions
from cars.eval.paired import paired_checkpoint_interval
from cars.research.artifacts import read_config
from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("runs/campaigns/broad_driver_20261028"))
    parser.add_argument("--output", type=Path, default=Path("reports/broad_driver_2026-10-28"))
    args = parser.parse_args()
    spec = json.loads((args.campaign / "config.json").read_text())
    status = json.loads((args.campaign / "status.json").read_text())
    if (status["status"] not in ("evaluations_complete", "complete") or spec["partition"] != "development"
            or spec["held_out_evaluation"] or spec["user_only_oracle_access"]):
        raise ValueError("Complete public development campaign required")
    if sha256(Path(spec["checkpoint"]) / "model.zip") != spec["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    tables, rows, provenance = {}, [], []
    for condition in spec["conditions"]:
        path = Path("runs") / condition["directory"] / condition["run_id"]
        config = read_config(path)
        driver = config["settings"]["driver"]
        actor = driver.get("actor", driver)
        if (config["settings"]["partition"] != "development" or config["protocol_id"] != spec["protocol_id"]
                or config["source_sha256"] != spec["source_sha256"] or config["versions"] != spec["versions"]
                or actor["checkpoint_sha256"] != spec["checkpoint_sha256"] or config["settings"]["user_only_oracle_monitor"]):
            raise ValueError("Unmatched observation/source/checkpoint contract")
        if condition["method"] == "policy-search":
            for name in ("planning_seed", "horizon", "steer_sigma", "gas_sigma", "brake_sigma", "observation_renderer",
                         "departure_cost", "centre_cost", "unknown_cost", "half_width_prior"):
                if driver[name] != spec[name]:
                    raise ValueError(f"Search parameter changed: {name}")
            assert driver["candidates"] == condition["candidates"]
            assert driver["simulator"]["vehicle_source_sha256"] == spec["vehicle_source_sha256"]
        if condition["method"] == "policy-bias":
            np.testing.assert_array_equal(np.asarray(driver["constant_action_bias"], dtype=np.float32),
                                          np.asarray(condition["bias"], dtype=np.float32))
        frame = pd.read_csv(path / "metrics.csv").sort_values("episode")
        np.testing.assert_array_equal(frame.episode, np.arange(spec["tracks_per_condition"]))
        np.testing.assert_array_equal(frame.track_seed, Partitions().seeds("development", spec["tracks_per_condition"]))
        if not np.isfinite(frame["return"]).all():
            raise ValueError("Nonfinite outcomes; do not silently drop failures")
        summary = json.loads((path / "summary.json").read_text())
        cost = json.loads((path / "cost.json").read_text())
        assert cost["native_frames"] == int(frame.steps.sum())
        assert cost["decisions"] == int(frame.decisions.sum())
        assert cost["internal_physics_frames"] == int(frame.model_transitions.sum()) * spec["action_repeat"]
        tables[condition["name"]] = frame
        rows.append({"condition": condition["name"], "episodes": len(frame),
            "mean_return": float(frame["return"].mean()), "return_std_tracks": float(frame["return"].std()),
            "return_iqm_tracks": float(trim_mean(frame["return"], .25)), "completion": float(frame.success.mean()),
            "mean_progress": float(frame.progress.mean()), "mean_offroad_frames": float(frame.off_track_steps.mean()),
            "playfield_exits": int(frame.crash.sum()), "completed_lap_time_mean_s": float(frame.loc[frame.success, "lap_time_s"].mean()) if frame.success.any() else None,
            "pipeline_p50_ms": summary["pipeline_latency_p50_ms"], "pipeline_p95_ms": summary["pipeline_latency_p95_ms"],
            "episode_mean_deadline_miss_fraction": float(frame.deadline_miss_fraction.mean()), **cost})
        provenance.append({"condition": condition["name"], "run": str(path), "config_sha256": sha256(path / "config.json"),
            "metrics_sha256": sha256(path / "metrics.csv"), "cost_sha256": sha256(path / "cost.json")})
    subsets, contrasts = [], []
    for panel in spec["analysis_panels"]:
        start, stop = panel["episode_slice"]
        selected = {name: frame.iloc[start:stop] for name, frame in tables.items()}
        for name, frame in selected.items():
            subsets.append({"panel": panel["name"], "condition": name, "episodes": len(frame),
                "mean_return": float(frame["return"].mean()), "completion": float(frame.success.mean()),
                "mean_offroad_frames": float(frame.off_track_steps.mean())})
        for contrast in spec["contrasts"]:
            for metric in ("return", "success"):
                a, b = selected[contrast["candidate"]], selected[contrast["reference"]]
                contrasts.append({"panel": panel["name"], **contrast, "metric": metric,
                    **paired_checkpoint_interval(a[metric].to_numpy(dtype=float)[None], b[metric].to_numpy(dtype=float)[None],
                        reps=spec["bootstrap_repetitions"], seed=spec["bootstrap_seed"]),
                    "uncertainty_scope": "conditional on one selected checkpoint and one fixed planning stream; tracks only"})
    table = pd.DataFrame(rows)
    native, internal = int(table.native_frames.sum()), int(table.internal_physics_frames.sum())
    if native > spec["native_frame_cap"] or internal > spec["internal_physics_frame_cap"]:
        raise ValueError("Declared resource cap exceeded")
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(subsets).to_csv(args.output / "subsets.csv", index=False)
    pd.concat([frame.assign(condition=name) for name, frame in tables.items()]).to_csv(args.output / "episodes.csv", index=False)
    write_json(args.output / "analysis.json", {"manifest_sha256": sha256(args.campaign / "config.json"),
        "checkpoint_sha256": spec["checkpoint_sha256"], "inputs": provenance, "paired_differences": contrasts,
        "native_frames": native, "internal_physics_frames": internal, "actor_forward_examples": int(table.actor_forward_examples.sum()),
        "training_cost": "zero new training; original policy development and all unsuccessful trials remain additional",
        "interpretation": "200-track development comparison; overlapping20/50 subsets and remaining150 are not final holdouts; one selected model and fixed planning stream",
        "multiplicity": "nominal unadjusted exploratory intervals; no confirmatory significance or SOTA claim",
        "held_out_evaluation": False, "user_only_oracle_values_read": False})
    labels = [condition["label"] for condition in spec["conditions"]]
    colors = [condition["color"] for condition in spec["conditions"]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6))
    for ax, field, title in zip(axes, ("mean_return", "completion", "pipeline_p95_ms"),
                               ("Mean official return", "Completion (%)", "Pipeline p95 (ms)"), strict=True):
        ax.bar(labels, table[field] * (100 if field == "completion" else 1), color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.spines[["top", "right"]].set_visible(False)
        if field == "completion":
            ax.set_ylim(0, 100)
        if field == "pipeline_p95_ms":
            ax.axhline(80, color="#a14124", linestyle="--", linewidth=1)
    fig.text(.02, .015, "One frozen checkpoint, 200 common development tracks. Sparse renderer for search. No deadline fallback or test-set claim.", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(args.output / "comparison.png", dpi=160)
    fig.savefig(args.output / "comparison.pdf")
    plt.close(fig)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
