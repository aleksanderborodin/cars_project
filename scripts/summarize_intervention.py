"""All cells and interactions; no five-track significance or adequacy claim."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cars.env.partitions import Partitions
from cars.research.artifacts import read_config
from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((args.campaign / "config.json").read_text())
    state = json.loads((args.campaign / "status.json").read_text())
    if (state["status"] not in ("evaluations_complete", "complete") or spec["partition"] != "development"
            or spec["held_out_evaluation"] or spec["user_only_oracle_access"]):
        raise ValueError("Completed public development diagnostic required")
    tables, rows, inputs = {}, [], []
    for cell in spec["conditions"]:
        path = Path("runs/policy_geometry_intervention") / cell["run_id"]
        config = read_config(path)
        driver = config["settings"]["driver"]
        if (config["protocol"]["view"] != "global" or config["settings"]["partition"] != "development"
                or config["source_sha256"] != spec["source_sha256"] or config["versions"] != spec["versions"]
                or config["partition_sha256"] != spec["partition_sha256"]
                or driver["actor"]["checkpoint_sha256"] != spec["checkpoint_sha256"]
                or config["settings"]["user_only_oracle_monitor"]):
            raise ValueError("Protocol/source/checkpoint mismatch")
        for key in ("candidates", "horizon", "planning_seed", "observation_renderer", "steer_sigma", "gas_sigma", "brake_sigma",
                    "departure_cost", "centre_cost", "unknown_cost", "half_width_prior"):
            if driver[key] != spec[key]:
                raise ValueError(f"Search mismatch: {key}")
        assert driver["intervention"] == {"reference": cell["reference"], "continuation": cell["continuation"], "version": 1}
        assert driver["simulator"]["vehicle_source_sha256"] == spec["vehicle_source_sha256"]
        table = pd.read_csv(path / "metrics.csv").sort_values("episode")
        np.testing.assert_array_equal(table.episode, np.arange(spec["tracks_per_condition"]))
        np.testing.assert_array_equal(table.track_seed, Partitions().seeds("development", spec["tracks_per_condition"]))
        if not np.isfinite(table["return"]).all():
            raise ValueError("Invalid outcome; never drop a failed trial")
        cost = json.loads((path / "cost.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        assert cost["native_frames"] == int(table.steps.sum())
        assert cost["internal_physics_frames"] == int(table.decisions.sum()) * spec["candidates"] * spec["horizon"] * spec["action_repeat"]
        fallbacks = sum(sum(d["visible_reference_fallback"] for d in json.loads(
            (path / f"episodes/{episode:04d}_diagnostics.json").read_text())) for episode in range(len(table)))
        tables[cell["name"]] = table
        rows.append({"condition": cell["name"], "episodes": len(table), "mean_return": float(table["return"].mean()),
            "completion": float(table.success.mean()), "mean_offroad_frames": float(table.off_track_steps.mean()),
            "pipeline_p95_ms": summary["pipeline_latency_p95_ms"], "reference_fallback_decisions": fallbacks, **cost})
        inputs.append({"run": str(path), "config_sha256": sha256(path / "config.json"),
                       "metrics_sha256": sha256(path / "metrics.csv"), "cost_sha256": sha256(path / "cost.json")})
    contrasts = []
    for metric in ("return", "success", "off_track_steps"):
        data = {name: table[metric].to_numpy(float) for name, table in tables.items()}
        for contrast in spec["contrasts"]:
            delta = sum(coefficient * data[name] for name, coefficient in contrast["weights"].items())
            contrasts.append({"metric": metric, "name": contrast["name"], "mean": float(delta.mean()),
                              "per_track": delta.tolist(), "uncertainty": "descriptive five-track integration pilot; no significance/equivalence claim"})
    result = pd.DataFrame(rows)
    assert result.native_frames.sum() <= spec["native_frame_cap"]
    assert result.internal_physics_frames.sum() <= spec["internal_physics_frame_cap"]
    args.output.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.output / "summary.csv", index=False)
    pd.concat([table.assign(condition=name) for name, table in tables.items()]).to_csv(args.output / "episodes.csv", index=False)
    write_json(args.output / "analysis.json", {"manifest_sha256": sha256(args.campaign / "config.json"),
        "inputs": inputs, "contrasts": contrasts, "native_frames": int(result.native_frames.sum()),
        "internal_physics_frames": int(result.internal_physics_frames.sum()),
        "actor_forward_examples": int(result.actor_forward_examples.sum()), "new_experiment_training_frames": 0,
        "interpretation": "fixed local actor; path-reconstruction and hidden-road module substitutions, not trained V/L/F comparison",
        "private_oracle_outputs_accessed": False, "held_out_evaluation": False})
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
