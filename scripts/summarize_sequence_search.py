"""Report matched optimizer outcomes, historical context, failures and actual costs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from cars.eval.paired import paired_checkpoint_interval
from cars.training import sha256, write_json
from sequence_campaign_common import complete_table, verify_manifest, verify_run, verify_work


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads((args.campaign / "config.json").read_text())
    verify_manifest(spec)
    state = json.loads((args.campaign / "status.json").read_text())
    checks = json.loads((args.campaign / "checks.json").read_text())
    if state["status"] not in ("evaluations_complete", "complete") or len(state["completed"]) != len(spec["conditions"]):
        raise ValueError("Complete campaign required")
    sources = {mode: {"path": Path(path), "fresh": False} for mode, path in spec["historical_context"].items()}
    new_costs = {}
    for c in spec["conditions"]:
        path = Path("runs") / c["directory"] / c["run_id"]
        verify_run(path, spec, mode=c["mode"], fresh=True, episodes=c["episodes"])
        complete_table(path, spec, c["episodes"])
        if checks[c["name"]]["replayed_episodes"] != c.get("replay_episodes", 0):
            raise ValueError("Replay gate incomplete")
        verify_work(path, spec, c["mode"])
        new_costs[c["name"]] = json.loads((path / "cost.json").read_text())
        if c["mode"] in ("colored", "icem"):
            sources[c["mode"]] = {"path": path, "fresh": True, "episodes": c["episodes"]}
    native = sum(c["native_frames"] for c in new_costs.values())
    internal = sum(c["internal_physics_frames"] for c in new_costs.values())
    if native > spec["native_frame_cap"] or internal > spec["internal_physics_frame_cap"]:
        raise ValueError("Stage budget exceeded")
    tables, provenance, summaries, behavior_rows = {}, [], [], []
    for mode, source in sources.items():
        path = source["path"]
        config = verify_run(path, spec, mode=mode, fresh=source["fresh"], episodes=source.get("episodes"))
        table = complete_table(path, spec, len(spec["seeds"]))
        tables[mode] = table
        timings, traces = {}, []
        for row in table.itertuples():
            trace = path / f"episodes/{row.episode:04d}.npz"
            diagnostic = path / f"episodes/{row.episode:04d}_diagnostics.json"
            with np.load(trace) as data:
                if int(data["frames"].sum()) != row.steps or len(data["action"]) != row.decisions:
                    raise ValueError("Trajectory accounting mismatch")
                np.testing.assert_allclose(data["reward"].sum(), table.loc[table.episode == row.episode, "return"].item(), atol=1e-8)
                timings[row.episode] = data["latency_ms"].copy()
                behavior_rows.append({"condition": mode, "episode": row.episode,
                    "sampled_mean_speed_mps": float(np.average(np.linalg.norm(data["observation_ego"][1:, :2], axis=1), weights=data["frames"])),
                    "mean_brake": float(np.average(data["action"][:, 2], weights=data["frames"])),
                    "truncated": bool(data["truncated"][-1])})
            traces.append({"path": str(trace), "sha256": sha256(trace), "diagnostics_sha256": sha256(diagnostic)})
        behavior = pd.DataFrame([r for r in behavior_rows if r["condition"] == mode])
        for panel in spec["analysis_panels"]:
            start, stop = panel["episode_slice"]
            subset, executed = table.iloc[start:stop], behavior.iloc[start:stop]
            latency = np.concatenate([timings[e] for e in subset.episode])
            summaries.append({"panel": panel["name"], "condition": mode, "historical_reuse": not source["fresh"],
                "episodes": len(subset), "mean_return": float(subset["return"].mean()),
                "return_std_tracks": float(subset["return"].std()), "return_iqm_tracks": float(trim_mean(subset["return"], .25)),
                "completions": int(subset.success.sum()), "completion": float(subset.success.mean()),
                "mean_progress": float(subset.progress.mean()), "mean_offroad_frames": float(subset.off_track_steps.mean()),
                "playfield_exits": int(subset.crash.sum()), "completed_lap_time_mean_s": float(subset.loc[subset.success, "lap_time_s"].mean()) if subset.success.any() else None,
                "sampled_mean_speed_mps": float(executed.sampled_mean_speed_mps.mean()), "mean_brake": float(executed.mean_brake.mean()),
                "pipeline_p50_ms": float(np.median(latency)), "pipeline_p95_ms": float(np.quantile(latency, .95)),
                "pipeline_p99_ms": float(np.quantile(latency, .99)), "deadline_miss_fraction": float(np.mean(latency > 80)),
                "native_frames": int(subset.steps.sum()), "model_transitions": int(subset.model_transitions.sum()),
                "internal_physics_frames": int(subset.model_transitions.sum()) * 4})
        provenance.append({"condition": mode, "run": str(path), "config_sha256": sha256(path / "config.json"),
            "metrics_sha256": sha256(path / "metrics.csv"), "cost_sha256": sha256(path / "cost.json"),
            "source_sha256": config["source_sha256"], "trajectories": traces})
    contrasts = []
    for panel in spec["analysis_panels"]:
        start, stop = panel["episode_slice"]
        for contrast in spec["contrasts"]:
            a = tables[contrast["candidate"]].iloc[start:stop]
            b = tables[contrast["reference"]].iloc[start:stop]
            np.testing.assert_array_equal(a.track_seed, b.track_seed)
            for metric in ("return", "success", "off_track_steps"):
                x, y = a[metric].to_numpy(float), b[metric].to_numpy(float)
                delta = x - y
                result = {**contrast, "panel": panel["name"], "metric": metric, "mean_difference": float(delta.mean()),
                    "per_track": delta.tolist(), "positive": int((delta > 0).sum()), "negative": int((delta < 0).sum()), "ties": int((delta == 0).sum())}
                if panel["intervals"]:
                    result["nominal_track_bootstrap_95_percent"] = paired_checkpoint_interval(x[None], y[None], **spec["bootstrap"])["interval_95_percent"]
                contrasts.append(result)
    previous_cost = None
    if spec["stage"] == "comparison":
        previous_cost = json.loads((Path(spec["prerequisite_campaign"]) / "status.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.concat([t.assign(condition=mode) for mode, t in tables.items()]).to_csv(args.output / "episodes.csv", index=False)
    pd.DataFrame(behavior_rows).to_csv(args.output / "executed_behavior.csv", index=False)
    write_json(args.output / "analysis.json", {"manifest_sha256": sha256(args.campaign / "config.json"),
        "inputs": provenance, "paired_contrasts": contrasts, "new_condition_costs": new_costs, "checks": checks,
        "new_native_frames": native, "new_internal_physics_frames": internal, "new_training_frames": 0,
        "preceding_smoke_native_frames": previous_cost["completed_native_frames"] if previous_cost else 0,
        "preceding_smoke_internal_physics_frames": previous_cost["completed_internal_physics_frames"] if previous_cost else 0,
        "historical_context_is_not_new_work": True, "bootstrap": spec["bootstrap"],
        "uncertainty": "one frozen actor/stream; nominal paired development-track intervals only; no selection adjustment or training-seed uncertainty",
        "comparison_scope": "primary same full-sequence representation, bounds, local inputs, batch sizes and actual rollout work; optimizer package includes fit/mean injection/cache/shift",
        "context_scope": "knots differ in representation and batching; direct has no simulator work; both historical with historical timings",
        "speed_definition": "decision-end observed speed weighted by native action duration, then equal episode means",
        "latency": "environment waits synchronously; no deadline fallback; serial noninterleaved conditions",
        "held_out_evaluation": False, "private_oracle_outputs_read": False})
    if spec["stage"] == "comparison":
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
        x = np.arange(1, len(spec["seeds"]) + 1)
        for mode, label in (("direct", "Direct frozen policy (historical)"), ("knots", "Three-knot shooting, 21 candidates (historical)"),
                            ("colored", "Colored sequence shooting"), ("icem", "Policy-guided iCEM adaptation")):
            axes[0].plot(x, tables[mode]["return"], "o-", markersize=4, label=label)
        axes[0].set_ylabel("Official episode return")
        axes[0].legend(fontsize=8)
        delta = tables["icem"]["return"].to_numpy() - tables["colored"]["return"].to_numpy()
        axes[1].bar(x, delta, color=np.where(delta >= 0, "#157f78", "#b24b4b"))
        axes[1].axhline(0, color="black", linewidth=.8)
        axes[1].set_ylabel("iCEM adaptation minus shooting return")
        axes[1].set_xlabel("Development track in fixed manifest order (not final test)")
        for ax in axes:
            ax.axvline(2.5, color="gray", linestyle="--", linewidth=1)
            ax.set_xticks(x)
        fig.suptitle("Same frozen actor and local simulator: 21 evaluated trajectories per decision\nFirst two tracks replay the smoke check; all twenty are development", fontsize=11)
        fig.savefig(args.output / "paired_returns.png", dpi=160)
        fig.savefig(args.output / "paired_returns.pdf")
        plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
