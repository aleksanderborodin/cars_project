"""Analyze only the public development runs explicitly named in this campaign."""
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/campaigns/control-audit-2026-10-28.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/control_audit_2026-10-28"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["partition"] != "development" or manifest["user_only_oracle_access"]:
        raise ValueError("Ordinary development evidence only")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance, summaries, episodes = [], [], []
    before, after, completions = [], [], []
    fixed = {}
    common_tracks = None

    def read(path: Path) -> pd.DataFrame:
        nonlocal common_tracks
        config = read_config(path)
        if config["protocol"] != manifest["protocol"] or config["settings"]["partition"] != "development":
            raise ValueError("Incompatible protocol or partition")
        frame = pd.read_csv(path / "metrics.csv").sort_values("track_seed")
        if len(frame) != manifest["tracks"] or not np.isfinite(frame["return"]).all():
            raise ValueError("Incomplete campaign; do not silently omit runs")
        if not (path / "cost.json").exists() or not (path / "summary.json").exists():
            raise ValueError("Run has not completed")
        if common_tracks is None:
            common_tracks = frame.track_seed.to_numpy()
        np.testing.assert_array_equal(frame.track_seed, common_tracks)
        provenance.append({"path": str(path), **{f"{name}_sha256": sha256(path / name)
                          for name in ("config.json", "metrics.csv", "cost.json")}})
        return frame

    for job in manifest["jobs"]:
        method = "model_search" if job["kind"] == "corrected_search" else "fixed_behavior"
        path = Path("runs") / method / job["run_id"]
        frame = read(path)
        if sha256(path / "campaign.json") != sha256(args.manifest):
            raise ValueError("Run campaign differs from analysis manifest")
        config = read_config(path)
        driver = config["settings"]["driver"]
        if job["kind"] == "corrected_search":
            if (driver["model_sha256"] != job["model_sha256"]
                    or driver["road_query"] != "shared_road_query_v2"
                    or driver["speeds"] != manifest["speeds"] or driver["steering_gains"] != manifest["gains"]
                    or driver["horizon"] != manifest["horizon_decisions"]):
                raise ValueError("Search condition differs from declaration")
            original = read(Path(job["before"]))
            if read_config(Path(job["before"]))["settings"]["driver"]["model_sha256"] != job["model_sha256"]:
                raise ValueError("Before/after models are not identical")
            before.append(original["return"].to_numpy())
            after.append(frame["return"].to_numpy())
            completions.append(frame.success.to_numpy(dtype=float))
        else:
            if driver["speed"] != job["speed"] or driver["gain"] != job["gain"]:
                raise ValueError("Fixed parameters differ from declaration")
            fixed[job["candidate_index"]] = frame
        cost = json.loads((path / "cost.json").read_text())
        timing = json.loads((path / "summary.json").read_text())
        summaries.append({"kind": job["kind"], "run_id": job["run_id"],
            "candidate_index": job.get("candidate_index"), "speed": job.get("speed"), "gain": job.get("gain"),
            "mean_return": frame["return"].mean(), "completion": frame.success.mean(),
            "mean_progress": frame.progress.mean(), "mean_offroad_frames": frame.off_track_steps.mean(),
            "pipeline_latency_p95_ms": timing["pipeline_latency_p95_ms"], **cost})
        for _, row in frame.iterrows():
            episode_path = path / "episodes" / f'{int(row.episode):04d}.npz'
            with np.load(episode_path) as arrays:
                diagnosis = describe_episode(row.to_dict(), arrays, manifest["failure_definitions"])
            episodes.append({"run_id": job["run_id"], "kind": job["kind"],
                "episode": row.episode, "track_seed": row.track_seed, "return": row["return"],
                "success": row.success, "progress": row.progress, **diagnosis,
                "trajectory_sha256": sha256(episode_path)})
    table = pd.DataFrame(summaries)
    fixed_table = table[table.kind == "fixed"].sort_values(
        ["mean_return", "completion", "candidate_index"], ascending=[False, False, True])
    selected = fixed_table.iloc[0]
    candidate = int(selected.candidate_index)
    baseline = read(Path(manifest["reference"]))
    after, before = np.stack(after), np.stack(before)
    delta = after - before
    correction = difference_interval(delta, np.zeros((1, delta.shape[1])))
    correction["units"] = "identical fitted models paired before/after, then resampled; common paired track draws"
    analysis = {"manifest_sha256": sha256(args.manifest), "inputs": provenance,
        "corrected_search": {"run_returns": after.mean(axis=1).tolist(), "mean_return": float(after.mean()),
            "completion_by_run": [float(x.mean()) for x in completions]},
        "correction_difference": correction,
        "selected_fixed": {"candidate_index": candidate, "speed": float(selected.speed), "gain": float(selected.gain),
            "mean_return": float(selected.mean_return), "completion": float(selected.completion)},
        "search_minus_selected_fixed": difference_interval(after, fixed[candidate]["return"].to_numpy()[None]),
        "search_minus_fixed60": difference_interval(after, baseline["return"].to_numpy()[None]),
        "cost": {"native_frames": int(table.native_frames.sum()),
            "model_transitions": int(table.model_transitions.sum()),
            "sum_run_wall_time_s": float(table.wall_time_s.sum()), "new_training_frames": 0},
        "interpretation": "adaptive development; fixed controller selected on these same tracks; intervals not selection-adjusted; no confirmatory/SOTA claim",
        "user_only_oracle_values_read": False, "held_out_evaluation": False}
    if analysis["cost"]["native_frames"] > manifest["evaluation_native_frame_cap"]:
        raise ValueError("Campaign frame budget exceeded")
    table.to_csv(args.output / "summary.csv", index=False)
    fixed_table.to_csv(args.output / "fixed_candidates.csv", index=False)
    episode_table = pd.DataFrame(episodes)
    episode_table.to_csv(args.output / "episodes.csv", index=False)
    episode_table.groupby(["run_id", "outcome"]).size().rename("episodes").reset_index().to_csv(
        args.output / "outcomes.csv", index=False)
    write_json(args.output / "analysis.json", analysis)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.25, 1]})
    means = fixed_table.pivot(index="gain", columns="speed", values="mean_return").sort_index()
    success = fixed_table.pivot(index="gain", columns="speed", values="completion").sort_index()
    image = axes[0].imshow(means, cmap="Blues", aspect="auto")
    for i in range(len(means)):
        for j in range(len(means.columns)):
            axes[0].text(j, i, f'{means.iloc[i,j]:.0f}\n{success.iloc[i,j]:.0%}', ha="center", va="center",
                         color="white" if means.iloc[i,j] > (means.values.min() + means.values.max()) / 2 else "black", fontsize=9)
    axes[0].set(xticks=range(len(means.columns)), xticklabels=means.columns.astype(int),
                yticks=range(len(means)), yticklabels=means.index, xlabel="Fixed speed target", ylabel="Steering gain",
                title="All 21 constants: mean return / completion")
    fig.colorbar(image, ax=axes[0], shrink=.75, label="Mean official return")
    x = np.arange(len(after))
    axes[1].bar(x - .18, before.mean(axis=1), width=.36, label="Original query")
    axes[1].bar(x + .18, after.mean(axis=1), width=.36, label="Corrected query")
    axes[1].axhline(selected.mean_return, color="#8b321d", linestyle="--", label="Selected fixed candidate")
    axes[1].set(xticks=x, xticklabels=[f"Model {i}" for i in x], ylabel="Mean official return",
                title="Same fitted models, 20 common tracks")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.text(.02, .015, "Development only. Constant selected on these tracks; no test or private ranking evidence. Completion is reported separately.", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(args.output / "control_audit.png", dpi=160)
    fig.savefig(args.output / "control_audit.pdf")
    plt.close(fig)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
