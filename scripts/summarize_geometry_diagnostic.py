"""Public, explicitly privileged geometry comparison on matched development tracks."""
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
    manifest = Path("configs/campaigns/geometry-diagnostic-2026-10-28.json")
    declaration = json.loads(manifest.read_text())
    if declaration["partition"] != "development" or declaration["user_only_oracle_access"]:
        raise ValueError("Public development diagnostic only")
    output = Path("reports/geometry_diagnostic_2026-10-28")
    output.mkdir(parents=True, exist_ok=True)
    rows, episodes, comparisons, provenance = [], [], [], []
    for condition in declaration["conditions"]:
        root = Path("runs/global_physics_search") / condition["run_id"]
        reference = Path(condition["local_reference"])
        matrices = []
        for path, label in ((reference, "local"), (root, "global_geometry")):
            configuration = read_config(path)
            expected = {**declaration["protocol"], "view": "local" if label == "local" else "global"}
            if configuration["protocol"] != expected or configuration["settings"]["partition"] != "development":
                raise ValueError("Information/partition mismatch")
            params = configuration["settings"]["driver"]
            assert params["speeds"] == declaration["speeds"] and params["steering_gains"] == condition["gains"]
            assert params["horizon"] == declaration["horizon"]
            assert params["simulator"]["vehicle_source_sha256"] == declaration["vehicle_source_sha256"]
            frame = pd.read_csv(path / "metrics.csv").sort_values("track_seed")
            if len(frame) != declaration["tracks"] or not np.isfinite(frame["return"]).all():
                raise ValueError("Incomplete run")
            cost = json.loads((path / "cost.json").read_text())
            summary = json.loads((path / "summary.json").read_text())
            matrices.append(frame)
            rows.append({"run_id": path.name, "information": label, "candidates": 7 * len(condition["gains"]),
                "mean_return": frame["return"].mean(), "completion": frame.success.mean(),
                "mean_progress": frame.progress.mean(), "mean_offroad_frames": frame.off_track_steps.mean(),
                "latency_p50_ms": summary["pipeline_latency_p50_ms"], "latency_p95_ms": summary["pipeline_latency_p95_ms"],
                **cost})
            provenance.append({"path": str(path), "config_sha256": sha256(path / "config.json"),
                               "metrics_sha256": sha256(path / "metrics.csv"), "cost_sha256": sha256(path / "cost.json")})
            for _, episode in frame.iterrows():
                with np.load(path / "episodes" / f"{int(episode.episode):04d}.npz") as arrays:
                    diagnosis = describe_episode(episode.to_dict(), arrays,
                        {"tail_native_frames": 100, "low_speed_mps": 2., "low_progress_fraction": .5})
                episodes.append({"run_id": path.name, "information": label, "episode": int(episode.episode),
                    "track_seed": int(episode.track_seed), "return": episode["return"], "success": bool(episode.success), **diagnosis})
        local, full = matrices
        np.testing.assert_array_equal(local.track_seed, full.track_seed)
        comparisons.append({"candidates": 7 * len(condition["gains"]),
            "global_minus_local_return": difference_interval(full["return"].to_numpy()[None], local["return"].to_numpy()[None]),
            "global_minus_local_completion": difference_interval(full.success.to_numpy()[None], local.success.to_numpy()[None])})
    table = pd.DataFrame(rows)
    costs = table[table.information == "global_geometry"]
    native, internal = int(costs.native_frames.sum()), int(costs.internal_physics_frames.sum())
    assert native <= declaration["native_evaluation_frame_cap"] and internal <= declaration["internal_physics_frame_cap"]
    result = {"manifest_sha256": sha256(manifest), "inputs": provenance, "comparisons": comparisons,
        "new_cost": {"native_frames": native, "internal_physics_frames": internal, "training_frames": 0},
        "replay_check": json.loads(Path("runs/verification/geometry_local_compatibility_20261028/result.json").read_text()),
        "interpretation": "privileged geometry plus centreline-reconstruction intervention; same estimated vehicle state; adaptive 20-track diagnostic; no local/SOTA claim",
        "held_out_evaluation": False, "user_only_oracle_values_read": False}
    write_json(output / "analysis.json", result)
    table.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(episodes).to_csv(output / "episodes.csv", index=False)
    pd.DataFrame(episodes).groupby(["run_id", "outcome"]).size().rename("episodes").reset_index().to_csv(output / "outcomes.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    labels = [f"{row.candidates} / {'local' if row.information == 'local' else 'full map'}" for row in table.itertuples()]
    colors = ["#777777", "#217b8d", "#777777", "#217b8d"]
    axes[0].bar(labels, table.mean_return, color=colors)
    axes[0].set(ylabel="Mean official return", title="Same 20 development tracks")
    axes[1].bar(labels, table.completion * 100, color=colors)
    axes[1].set(ylabel="Completion (%)", ylim=(0, 105), title="Known road; estimated vehicle state")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=12, labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.02, .015, "Full map is privileged and also replaces centreline reconstruction. This is an adaptive diagnostic, not equal-input superiority.", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(output / "geometry_diagnostic.png", dpi=160)
    fig.savefig(output / "geometry_diagnostic.pdf")
    plt.close(fig)
    print(table.to_string(index=False))
    print(json.dumps({k: v for k, v in result.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
