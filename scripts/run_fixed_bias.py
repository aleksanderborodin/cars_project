"""Fit-declared fixed offsets versus the same frozen policy and completed search."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from cars.eval.paired import paired_checkpoint_interval
from cars.planning.policy_controls import FixedPolicyBias
from cars.research.artifacts import read_config
from cars.training import sha256, write_json


def main() -> None:
    manifest = Path("configs/campaigns/fixed-bias-2026-10-28.json")
    spec = json.loads(manifest.read_text())
    assert spec["partition"] == "development" and not spec["held_out_evaluation"] and not spec["user_only_oracle_access"]
    checkpoint = Path(spec["checkpoint"])
    assert sha256(checkpoint / "model.zip") == spec["checkpoint_sha256"]
    for source in spec["input_hashes"]:
        path = Path(source["trace"])
        assert sha256(path) == source["trace_sha256"]
        assert sha256(path.with_name(path.stem + "_diagnostics.json")) == source["diagnostics_sha256"]
    root = Path("runs/campaigns") / spec["id"]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest, root / "config.json")
    shutil.copy2(__file__, root / "runner.py")
    reference_spec = json.loads((Path(spec["reference_campaign"]) / "config.json").read_text())
    paths = {c["name"]: Path("runs") / c["directory"] / c["run_id"] for c in reference_spec["conditions"]
             if c["name"] in ("direct", "search7", "search21")}
    actor = FixedPolicyBias(checkpoint, (0., 0., 0.))
    checks = 0
    for path in sorted(paths["direct"].glob("episodes/*.npz")):
        actor.reset()
        with np.load(path) as data:
            for i in range(12):
                obs = {key.removeprefix("observation_"): value[i] for key, value in data.items() if key.startswith("observation_")}
                np.testing.assert_array_equal(actor.act(obs), data["action"][i])
                checks += 1
    assert checks == spec["verification_actor_forward_examples"]
    write_json(root / "verification.json", {"zero_bias_actions_exact": True, "actor_forward_examples": checks, "native_frames": 0})
    started = time.perf_counter()
    completed, native = [], 0
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    try:
        for condition in spec["conditions"]:
            command = [sys.executable, "-m", "cars.cli", "research", "evaluate", "--method", "policy-bias",
                "--model", str(checkpoint), "--action-bias", *map(str, condition["bias"]),
                "--episodes", str(spec["tracks"]), "--run-id", condition["run_id"]]
            write_json(root / (condition["name"] + "_command.json"), command)
            write_json(root / "status.json", {"status": "running", "current": condition["name"], "completed": completed})
            with (root / (condition["name"] + ".log")).open("x") as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            path = Path("runs/policy_bias") / condition["run_id"]
            shutil.copy2(manifest, path / "campaign.json")
            cost = json.loads((path / "cost.json").read_text())
            native += cost["native_frames"]
            assert cost["internal_physics_frames"] == 0 and native <= spec["native_frame_cap"]
            paths[condition["name"]] = path
            completed.append(condition["name"])
        tables, rows, provenance, tracks = {}, [], [], None
        for name, path in paths.items():
            config = read_config(path)
            driver = config["settings"]["driver"]
            assert driver.get("actor", driver)["checkpoint_sha256"] == spec["checkpoint_sha256"]
            assert config["protocol_id"] == spec["protocol_id"] and config["partition_sha256"] == spec["partition_sha256"]
            frame = pd.read_csv(path / "metrics.csv").sort_values("track_seed")
            assert len(frame) == spec["tracks"] and np.isfinite(frame["return"]).all()
            if tracks is None:
                tracks = frame.track_seed.to_numpy()
            np.testing.assert_array_equal(tracks, frame.track_seed)
            tables[name] = frame
            summary = json.loads((path / "summary.json").read_text())
            rows.append({"condition": name, "mean_return": float(frame["return"].mean()),
                "completion": float(frame.success.mean()), "mean_offroad_frames": float(frame.off_track_steps.mean()),
                "pipeline_p95_ms": summary["pipeline_latency_p95_ms"]})
            provenance.append({"condition": name, "config_sha256": sha256(path / "config.json"), "metrics_sha256": sha256(path / "metrics.csv")})
        contrasts = []
        for condition in spec["conditions"]:
            name = condition["name"]
            for reference in ("direct", "search21"):
                for metric in ("return", "success"):
                    contrasts.append({"candidate": name, "reference": reference, "metric": metric,
                        **paired_checkpoint_interval(tables[name][metric].to_numpy(dtype=float)[None],
                            tables[reference][metric].to_numpy(dtype=float)[None]),
                        "uncertainty_scope": "fixed selected checkpoint, same adaptive development fitting/evaluation panel; tracks only"})
        output = Path(spec["report_directory"])
        output.mkdir(parents=True, exist_ok=False)
        pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
        pd.concat([frame.assign(condition=name) for name, frame in tables.items()]).to_csv(output / "episodes.csv", index=False)
        write_json(output / "analysis.json", {"manifest_sha256": sha256(manifest), "inputs": provenance,
            "contrasts": contrasts, "new_native_frames": native, "new_internal_physics_frames": 0,
            "interpretation": "same-panel fitted cheap controls; diagnostic, no held-out/strong-baseline superiority claim",
            "user_only_oracle_values_read": False, "held_out_evaluation": False})
        write_json(root / "status.json", {"status": "complete", "completed": completed,
            "native_frames": native, "internal_physics_frames": 0, "wall_time_s": time.perf_counter() - started})
        print(pd.DataFrame(rows).to_string(index=False))
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "completed": completed, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
