"""Execute a declared native baseline in serial training/development stages."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from cars.research.native_config import NativeTrainingConfig
from cars.env.partitions import Partitions
from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    declaration = parser.parse_args().manifest
    spec = json.loads(declaration.read_text())
    config = NativeTrainingConfig(**spec["training"])
    if config.total_decisions != 4_000_000 or spec["evaluation_episodes_per_checkpoint"] != 50:
        raise ValueError("This runner implements the declared 4M / first-50-development campaign")
    if spec["partition_sha256"] != Partitions().fingerprint:
        raise ValueError("Partition fingerprint changed")
    if [stage["stop_at"] for stage in spec["stages"]] != [500000, 1000000, 2000000, 4000000]:
        raise ValueError("Unexpected checkpoint schedule")
    if sha256(Path(spec["profile_summary"])) != spec["profile_summary_sha256"]:
        raise ValueError("Selected profile evidence changed")
    if shutil.disk_usage(".").free < spec["initial_free_disk_requirement_bytes"]:
        raise RuntimeError("Insufficient free disk for the declared campaign")
    root = Path("runs/campaigns") / spec["id"]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(declaration, root / "config.json")
    shutil.copy2(__file__, root / "runner.py")
    write_json(root / "provenance.json", {"manifest_sha256": sha256(declaration),
        "runner_sha256": sha256(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
    training_path = root / "training_config.json"
    write_json(training_path, config.resolved())
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "SDL_VIDEODRIVER": "dummy", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    results, resume = [], None
    started = time.perf_counter()
    try:
        for stage in spec["stages"]:
            if shutil.disk_usage(root).free < spec["minimum_free_disk_bytes"]:
                raise RuntimeError("Declared free-space floor reached; preserve data and stop")
            for mode in ("training", "evaluation"):
                write_json(root / "status.json", {"status": "running", "active": stage[mode + "_run_id"],
                    "phase": mode, "target_decisions": stage["stop_at"], "completed": results,
                    "pid": os.getpid()})
                command = [sys.executable, "-m", "cars.cli", "research"]
                if mode == "training":
                    command += ["train-native", "--config", str(training_path), "--stop-at", str(stage["stop_at"]),
                                "--run-id", stage["training_run_id"]]
                    if resume:
                        command += ["--resume", str(resume)]
                else:
                    command += ["evaluate-native", "--model", str(run),
                        "--episodes", str(spec["evaluation_episodes_per_checkpoint"]),
                        "--run-id", stage["evaluation_run_id"]]
                with (root / (stage[mode + "_run_id"] + ".log")).open("x") as log:
                    process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                if process.returncode:
                    raise RuntimeError(f"{mode} failed: {stage[mode + '_run_id']}; see saved log/artifacts")
                if mode == "training":
                    run = Path("runs/native_training") / stage["training_run_id"]
                else:
                    evaluated = Path("runs/native_evaluation") / stage["evaluation_run_id"]
            state = json.loads((run / "training_state.json").read_text())
            row = {**stage, "actual_cumulative_decisions": state["decisions"],
                "training_cost": json.loads((run / "cost.json").read_text()),
                "evaluation_cost": json.loads((evaluated / "cost.json").read_text()),
                "development": json.loads((evaluated / "summary.json").read_text())}
            results.append(row)
            write_json(root / "results.json", results)
            resume = run
            print(json.dumps({"completed": stage["stop_at"], "development": row["development"]}), flush=True)
        selected = max(results, key=lambda row: (row["development"]["mean_return"],
                       row["development"]["success_rate"], -row["actual_cumulative_decisions"]))
        write_json(root / "selection.json", {"rule": spec["checkpoint_selection"],
            "training_run_id": selected["training_run_id"], "evaluation_run_id": selected["evaluation_run_id"],
            "native_protocol_only": True, "not_final_test_or_sota_evidence": True})
        write_json(root / "status.json", {"status": "complete", "completed": results,
            "wall_seconds": time.perf_counter() - started})
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "error": str(error), "completed": results,
            "wall_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
