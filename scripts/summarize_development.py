"""Rebuild public pilot tables/figure. Never traverses private diagnostics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cars.eval.paired import difference_interval
from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/campaigns/development-pilot-2026-10-28.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/pilot_2026-10-28"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    rows, episodes, provenance = [], [], []
    for entry in manifest["runs"]:
        path = Path(entry["path"])
        frame = pd.read_csv(path / "metrics.csv")
        assert len(frame) == manifest["tracks"]
        frame.insert(0, "variant", entry["label"])
        episodes.append(frame)
        rows.append({"variant": entry["label"], "run": entry["path"],
            "mean_return": frame["return"].mean(), "completion": frame.success.mean(),
            "mean_progress": frame.progress.mean(), "mean_offroad_frames": frame.off_track_steps.mean(),
            "information": frame.information.iloc[0], "episodes": len(frame)})
        provenance.append({"path": entry["path"], "metrics_sha256": sha256(path / "metrics.csv"),
                           "config_sha256": sha256(path / "config.json")})
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "summary.csv", index=False)
    pd.concat(episodes, ignore_index=True).to_csv(args.output / "episodes.csv", index=False)
    reference = pd.read_csv(Path(manifest["reference"]) / "metrics.csv").sort_values("track_seed")
    replicas = [pd.read_csv(Path(path) / "metrics.csv").sort_values("track_seed") for path in manifest["replicated_variant"]]
    for frame in replicas:
        np.testing.assert_array_equal(frame.track_seed, reference.track_seed)
    matrix = np.stack([frame["return"].to_numpy() for frame in replicas])
    result = {"manifest_sha256": sha256(args.manifest), "inputs": provenance,
        "replicated_variant": {"paths": manifest["replicated_variant"], "run_means": matrix.mean(axis=1).tolist(),
            "mean_return": float(matrix.mean()), "completion_by_run": [float(frame.success.mean()) for frame in replicas],
            "difference_from_fixed": difference_interval(matrix, reference["return"].to_numpy()[None])},
        "interpretation": "adaptive development evidence; not selection-adjusted or confirmatory; PPO pilot is not competitive",
        "user_only_oracle_values_read": False}
    write_json(args.output / "analysis.json", result)
    fig, ax = plt.subplots(figsize=(11, 6.8))
    for index, (row, frame) in enumerate(zip(rows, episodes, strict=True)):
        color = "#ad6b17" if row["information"] == "global_privileged" else "#23649a"
        ax.scatter(frame["return"], np.full(len(frame), index), s=15, color=color, alpha=.28)
        ax.scatter(row["mean_return"], index, marker="D", s=60, color=color, zorder=3)
    ax.set_yticks(np.arange(len(rows)), [row["variant"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Official episode return — dots: individual tracks; diamonds: means")
    ax.set_title("Development pilot: 20 common tracks", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=.2)
    ax.spines[["right", "top"]].set_visible(False)
    fig.text(.02, .015, "Adaptive pilot; one fitted model per displayed row. Full-map expert has privileged inputs. No held-out or SOTA claim.", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(args.output / "development_results.png", dpi=160)
    fig.savefig(args.output / "development_results.pdf")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
