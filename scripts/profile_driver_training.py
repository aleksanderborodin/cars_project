"""Declared six-case training profile; one complete PPO update per fresh run."""
from dataclasses import replace
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from cars.research.driver_config import DriverTrainingConfig
from cars.training import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/campaigns/driver-profile-2026-10-28.json"))
    declaration = parser.parse_args().manifest
    spec = json.loads(declaration.read_text())
    root = Path("runs/campaigns") / spec["id"]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(declaration, root / "config.json")
    shutil.copy2(__file__, root / "runner.py")
    geometry = json.loads(Path("runs/campaigns/geometry_diagnostic_20261028/status.json").read_text())
    if geometry["status"] != "complete":
        raise RuntimeError("Finish geometry work before profiling")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    results = []
    started = time.perf_counter()
    write_json(root / "status.json", {"status": "running", "completed": []})
    for case in spec["conditions"]:
        config = replace(DriverTrainingConfig(), seed=spec["training_seed"], workers=case["workers"],
                         device=case["device"], checkpoints=())
        path = root / (case["run_id"] + ".json")
        write_json(path, config.resolved())
        command = [sys.executable, "-m", "cars.cli", "research", "train-driver", "--config", str(path),
                   "--stop-at", "1", "--run-id", case["run_id"]]
        with (root / (case["run_id"] + ".log")).open("x") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        run = Path("runs/driver_training") / case["run_id"]
        if result.returncode:
            failure = json.loads((run / "failure.json").read_text()) if (run / "failure.json").exists() else {"error": "inspect public log", "native_frames": 0}
            row = {**case, "status": "failed", **failure}
        else:
            cost = json.loads((run / "cost.json").read_text())
            learning = [json.loads(line) for line in (run / "learning.jsonl").read_text().splitlines()]
            row = {**case, "status": "complete", **cost,
                   "update_native_frames_per_s": cost["native_frames"] / sum(x["update_wall_time_s"] for x in learning)}
        results.append(row)
        write_json(root / "results.json", results)
        write_json(root / "status.json", {"status": "running", "completed": [r["run_id"] for r in results]})
        print(json.dumps(row), flush=True)
        assert sum(r.get("native_frames", 0) for r in results) <= spec["training_native_frame_cap"]
    write_json(root / "status.json", {"status": "complete", "completed": [r["run_id"] for r in results],
        "wall_time_s": time.perf_counter() - started, "native_frames": sum(r.get("native_frames", 0) for r in results)})


if __name__ == "__main__":
    main()
