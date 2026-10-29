"""Report every declared development panel, paired differences, timing and provenance."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cars.eval.paired import paired_checkpoint_interval
from cars.training import sha256, write_json
from reference_replication_common import CONDITIONS, verify_config, verify_declaration, verified_table


PANELS = {"all20": (0, 20), "original5": (0, 5), "additional15": (5, 20)}


def contrasts(tables: dict[str, pd.DataFrame], spec: dict) -> list[dict]:
    results = []
    for panel, (start, stop) in PANELS.items():
        candidate = tables["true_tail"].iloc[start:stop]
        for reference in ("tangent", "local_context"):
            control = tables[reference].iloc[start:stop]
            np.testing.assert_array_equal(candidate.track_seed, control.track_seed)
            for metric in ("return", "success", "progress", "off_track_steps"):
                a, b = candidate[metric].to_numpy(float), control[metric].to_numpy(float)
                delta = a - b
                result = {"panel": panel, "reference": reference, "metric": metric,
                    "role": "primary" if reference == "tangent" else "different_information_context",
                    "mean_difference": float(delta.mean()), "per_track": delta.tolist(),
                    "positive": int((delta > 0).sum()), "negative": int((delta < 0).sum()),
                    "ties": int((delta == 0).sum())}
                if panel != "original5":
                    interval = paired_checkpoint_interval(a[None], b[None], **spec["bootstrap"])
                    result["nominal_track_bootstrap_95_percent"] = interval["interval_95_percent"]
                results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((args.campaign / "config.json").read_text())
    verify_declaration(spec)
    state = json.loads((args.campaign / "status.json").read_text())
    replay = json.loads((args.campaign / "replay_check.json").read_text())
    if (state["status"] not in ("evaluations_complete", "complete") or not replay["exact_replay"]
            or replay["conditions"] != {name: 5 for name in CONDITIONS}):
        raise ValueError("Completed, replay-verified campaign required")
    tables, provenance, costs = {}, [], {}
    summary_rows, behavior_rows = [], []
    paths = {name: Path("runs/policy_geometry_intervention") / item["run_id"] for name, item in spec["conditions"].items()}
    paths["local_context"] = Path(spec["context_run"])
    for name, path in paths.items():
        config = verify_config(path, spec, CONDITIONS.get(name), fresh=name != "local_context")
        table = verified_table(path, spec)
        tables[name] = table
        costs[name] = json.loads((path / "cost.json").read_text())
        latencies, trajectories = {}, []
        for row in table.itertuples():
            source = path / f"episodes/{row.episode:04d}.npz"
            diagnostic = path / f"episodes/{row.episode:04d}_diagnostics.json"
            with np.load(source) as data:
                if int(data["frames"].sum()) != row.steps or len(data["action"]) != row.decisions:
                    raise ValueError("Trajectory frame/decision counts disagree")
                # Return is a Python keyword; use the named DataFrame column.
                np.testing.assert_allclose(data["reward"].sum(), table.loc[table.episode == row.episode, "return"].item(), atol=1e-8)
                latencies[row.episode] = data["latency_ms"].copy()
                behavior_rows.append({"condition": name, "episode": row.episode,
                    "sampled_mean_speed_mps": float(np.average(np.linalg.norm(data["observation_ego"][1:, :2], axis=1), weights=data["frames"])),
                    "mean_brake": float(np.average(data["action"][:, 2], weights=data["frames"])),
                    "truncated": bool(data["truncated"][-1]),
                    "reference_fallbacks": (sum(d["visible_reference_fallback"] for d in json.loads(diagnostic.read_text()))
                                             if name != "local_context" else 0)})
            trajectories.append({"path": str(source), "sha256": sha256(source), "diagnostics_sha256": sha256(diagnostic)})
        behavior = pd.DataFrame([r for r in behavior_rows if r["condition"] == name])
        for panel, (start, stop) in PANELS.items():
            subset = table.iloc[start:stop]
            executed = behavior.iloc[start:stop]
            timing = np.concatenate([latencies[e] for e in subset.episode])
            summary_rows.append({"panel": panel, "condition": name, "episodes": len(subset),
                "historical_reuse": name == "local_context", "mean_return": float(subset["return"].mean()),
                "completions": int(subset.success.sum()), "completion": float(subset.success.mean()),
                "mean_progress": float(subset.progress.mean()), "mean_offroad_frames": float(subset.off_track_steps.mean()),
                "playfield_exits": int(subset.crash.sum()), "sampled_mean_speed_mps": float(executed.sampled_mean_speed_mps.mean()),
                "mean_brake": float(executed.mean_brake.mean()), "reference_fallbacks": int(executed.reference_fallbacks.sum()),
                "pipeline_p95_ms": float(np.quantile(timing, .95)), "deadline_miss_fraction": float(np.mean(timing > 80)),
                "native_frames": int(subset.steps.sum()), "internal_physics_frames": int(subset.decisions.sum()) * 560})
        provenance.append({"condition": name, "run": str(path), "config_sha256": sha256(path / "config.json"),
            "metrics_sha256": sha256(path / "metrics.csv"), "cost_sha256": sha256(path / "cost.json"),
            "source_sha256": config["source_sha256"], "trajectories": trajectories})
    native = sum(costs[name]["native_frames"] for name in CONDITIONS)
    internal = sum(costs[name]["internal_physics_frames"] for name in CONDITIONS)
    if native > spec["native_frame_cap"] or internal > spec["internal_physics_frame_cap"]:
        raise ValueError("Resource cap exceeded")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.concat([table.assign(condition=name) for name, table in tables.items()]).to_csv(args.output / "episodes.csv", index=False)
    pd.DataFrame(behavior_rows).to_csv(args.output / "executed_behavior.csv", index=False)
    differences = contrasts(tables, spec)
    write_json(args.output / "analysis.json", {"manifest_sha256": sha256(args.campaign / "config.json"),
        "inputs": provenance, "paired_contrasts": differences, "new_native_frames": native,
        "new_internal_physics_frames": internal, "new_training_frames": 0,
        "new_condition_costs": {name: costs[name] for name in CONDITIONS}, "replay_check": replay,
        "historical_context_cost_not_new_work": True, "bootstrap": spec["bootstrap"],
        "uncertainty": "nominal paired-track intervals only, one frozen actor/stream, no selection adjustment or training-seed uncertainty; original5 descriptive only",
        "panels": "original5 informed this intervention; additional15 also development, previously used by the local context; all20 is not independent confirmation",
        "speed_definition": "decision-end observed speed weighted by action duration, then equal episode means",
        "latency": "pooled measured decisions per panel; historical local timings not paired; synchronous environment, no deadline fallback",
        "held_out_evaluation": False, "private_oracle_outputs_read": False})
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    x = np.arange(1, 21)
    for name, label in (("local_context", "Local search (historical; different information)"),
                        ("tangent", "Oracle road + tangent reference"), ("true_tail", "Oracle road + true reference tail")):
        axes[0].plot(x, tables[name]["return"], "o-", markersize=4, label=label)
    axes[0].set_ylabel("Official episode return")
    axes[0].legend(fontsize=8)
    delta = tables["true_tail"]["return"].to_numpy() - tables["tangent"]["return"].to_numpy()
    axes[1].bar(x, delta, color=np.where(delta >= 0, "#157f78", "#b24b4b"))
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set_ylabel("True tail minus tangent return")
    axes[1].set_xlabel("Development track in fixed manifest order (not final test)")
    for ax in axes:
        ax.axvline(5.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
    fig.suptitle("Frozen reference-tail replication: one actor, seven candidates, horizon 20\nTracks 1–5: original pilot; 6–20: additional development tracks", fontsize=11)
    fig.savefig(args.output / "paired_returns.png", dpi=160)
    fig.savefig(args.output / "paired_returns.pdf")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
