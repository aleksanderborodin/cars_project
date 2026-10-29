"""Run the frozen four-case native training profile serially; never score-select."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from cars.research.native_config import NativeTrainingConfig
from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    declaration = parser.parse_args().manifest
    spec = json.loads(declaration.read_text())
    root = Path("runs/campaigns") / spec["id"]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(declaration, root / "config.json")
    shutil.copy2(__file__, root / "runner.py")
    write_json(root / "provenance.json", {"manifest_sha256": sha256(declaration),
        "runner_sha256": sha256(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "SDL_VIDEODRIVER": "dummy", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    results = []
    started = time.perf_counter()
    for case in spec["conditions"]:
        config = replace(NativeTrainingConfig(), seed=spec["training_seed"],
            device=case["device"], vector_backend=case["vector_backend"])
        path = root / (case["run_id"] + ".json")
        write_json(path, config.resolved())
        stop = spec["rollouts_per_case"] * config.workers * config.n_steps
        write_json(root / "status.json", {"status": "running", "active": case["run_id"],
                                          "completed": [r["run_id"] for r in results]})
        command = [sys.executable, "-m", "cars.cli", "research", "train-native", "--config", str(path),
                   "--stop-at", str(stop), "--run-id", case["run_id"]]
        with (root / (case["run_id"] + ".log")).open("x") as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        run = Path("runs/native_training") / case["run_id"]
        if process.returncode:
            failure = json.loads((run / "failure.json").read_text()) if (run / "failure.json").exists() else {
                "error": "See public process log; run failed before artifact setup", "native_frames": 0}
            row = {**case, "status": "failed", **failure}
        else:
            cost = json.loads((run / "cost.json").read_text())
            learning = [json.loads(line) for line in (run / "learning.jsonl").read_text().splitlines()]
            steady = learning[1:]
            mean_seconds = sum(row["learn_seconds"] for row in steady) / len(steady)
            row = {**case, "status": "complete", **cost,
                "first_learn_seconds": learning[0]["learn_seconds"],
                "steady_learn_seconds_mean": mean_seconds,
                "steady_native_frames_per_second": config.workers * config.n_steps / mean_seconds,
                "projected_977_rollouts_hours": mean_seconds * 977 / 3600,
                "initial_parameters_sha256": json.loads((run / "initial_policy.json").read_text())["parameters_sha256"],
                "artifact_sha256": {name: sha256(run / name) for name in (
                    "config.json", "cost.json", "learning.jsonl", "model.zip", "normalization.pkl", "reload_check.json")}}
        results.append(row)
        write_json(root / "results.json", results)
        print(json.dumps({key: row[key] for key in ("run_id", "status", "native_frames")}), flush=True)
        if sum(row.get("native_frames", 0) for row in results) > spec["training_native_frame_cap"]:
            raise RuntimeError("Declared profile interaction cap exceeded")
    write_json(root / "status.json", {"status": "complete" if all(r["status"] == "complete" for r in results) else "complete_with_failures",
        "wall_seconds": time.perf_counter() - started, "completed": [r["run_id"] for r in results],
        "native_frames": sum(r.get("native_frames", 0) for r in results)})


if __name__ == "__main__":
    main()
