"""Publish a completed device diagnostic without changing checkpoint selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.campaign / "status.json").read_text())
    assert status["status"] == "complete"
    spec = json.loads((args.campaign / "config.json").read_text())
    result = json.loads((args.campaign / "results.json").read_text())
    provenance, verified, tables = {}, 0, {}
    rng = np.random.default_rng(20261029)
    draws = rng.integers(0, 10, size=(10000, 10))
    for case, comparison in zip(spec["conditions"], result["comparisons"], strict=True):
        assert case["checkpoint"] == comparison["checkpoint"]
        for check in comparison["replays"]:
            assert check["passed"] and check["all_trace_fields_exact_except_timing"]
        cpu = pd.read_csv(Path(case["original_cpu_evaluation"]) / "episodes.csv").head(10)
        cuda = pd.read_csv(Path("runs/native_evaluation") / case["cuda_run_id"] / "episodes.csv")
        np.testing.assert_array_equal(cpu.track_seed, cuda.track_seed)
        delta = cuda["return"].to_numpy() - cpu["return"].to_numpy()
        assert np.isclose(delta.mean(), comparison["paired_return_change"])
        comparison["paired_track_bootstrap95"] = np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()
        tables[case["checkpoint"]] = {"cpu": cpu["return"].to_numpy(), "cuda": cuda["return"].to_numpy()}
        for kind in ("cpu_replay", "cuda", "cuda_replay"):
            run = Path("runs/native_evaluation") / case[kind + "_run_id"]
            config = json.loads((run / "config.json").read_text())
            assert config["settings"]["inference_device"] == ("cpu" if kind == "cpu_replay" else "cuda")
            assert config["settings"]["artifact_sha256"] == case["checkpoint_sha256"]
            assert json.loads((run / "summary.json").read_text())["normalization_frozen"]
            for name, expected in config["source_sha256"].items():
                assert sha256(run / "source" / name) == expected
                verified += 1
            provenance[str(run)] = {str(p.relative_to(run)): sha256(p) for p in run.rglob("*")
                if p.is_file() and "source" not in p.relative_to(run).parts}
    result["checkpoint_contrasts"] = {}
    for device in ("cpu", "cuda"):
        result["checkpoint_contrasts"][device] = {}
        for checkpoint in ("2m", "4m"):
            delta = tables[checkpoint][device] - tables["1m"][device]
            result["checkpoint_contrasts"][device][checkpoint + "_minus_1m"] = {
                "mean_change": float(delta.mean()),
                "paired_track_bootstrap95": np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()}
    result["cost_summary"] = {key: status[key] for key in ("native_frames", "reset_physics_ticks", "wall_seconds")}
    result.update(source_files_verified=verified, historical_cpu_replay_episodes=6, cuda_replay_episodes=6,
        original_checkpoint_selection_unchanged=True, training_native_frames=0,
        interpretation="Later regression persists on both devices; precision/device can still materially affect individual trajectories; ten-track development diagnostic only",
        uncertainty="10000 paired track bootstrap draws seed20261029; no independent training seeds, equivalence claim or confirmatory test")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "summary.json", result)
    write_json(args.output / "provenance.json", {"runs": provenance,
        "campaign": {p.name: sha256(p) for p in args.campaign.iterdir() if p.is_file()},
        "analysis_script_sha256": sha256(Path(__file__))})
    print(json.dumps({"comparisons": result["comparisons"], "cost": result["cost_summary"], "sources": verified}))


if __name__ == "__main__":
    main()
