"""Execute the explicitly declared physics-budget and fixed-controller follow-up."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from cars.training import write_json


def main() -> None:
    manifest = Path("configs/campaigns/physics-budget-2026-10-28.json")
    data = json.loads(manifest.read_text())
    if data["partition"] != "development" or data["user_only_oracle_access"]:
        raise ValueError("Only ordinary development evaluation is allowed")
    root = Path("runs/campaigns") / data["id"]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest, root / "config.json")
    shutil.copy2(__file__, root / "runner.py")
    condition = data["physics_condition"]
    commands = [(condition["run_id"], ["--method", "physics", "--horizon", str(condition["horizon"]),
        "--speed-grid", *map(str, condition["speeds"]), "--gain-grid", *map(str, condition["gains"])])]
    commands += [(job["run_id"], ["--method", "fixed", "--speed", str(job["speed"]), "--gain", str(job["gain"])])
                 for job in data["new_fixed_jobs"]]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    write_json(root / "commands.json", commands)
    started = time.perf_counter()
    completed = []

    def execute(entry: tuple) -> str:
        run_id, arguments = entry
        with (root / f"{run_id}.log").open("x") as output:
            subprocess.run([sys.executable, "-m", "cars.cli", "research", "evaluate", *arguments,
                            "--episodes", str(data["tracks"]), "--run-id", run_id],
                           stdout=output, stderr=subprocess.STDOUT, env=environment, check=True)
        method = "physics_search" if run_id == condition["run_id"] else "fixed_behavior"
        shutil.copy2(manifest, Path("runs") / method / run_id / "campaign.json")
        return run_id

    write_json(root / "status.json", {"status": "running", "completed": []})
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(execute, command) for command in commands]
            for future in concurrent.futures.as_completed(futures):
                completed.append(future.result())
                write_json(root / "status.json", {"status": "running", "completed": completed,
                    "wall_time_s": time.perf_counter() - started})
                print(f"{len(completed)}/{len(commands)} completed: {completed[-1]}", flush=True)
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "completed": completed, "error": str(error)})
        raise
    write_json(root / "status.json", {"status": "complete", "completed": completed,
        "wall_time_s": time.perf_counter() - started})


if __name__ == "__main__":
    main()
