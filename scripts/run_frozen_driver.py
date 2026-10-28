"""Execute the declared same-checkpoint development comparison sequentially."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from cars.env.partitions import Partitions
from cars.research.artifacts import read_config
from cars.training import sha256, write_json


def main() -> None:
    manifest = Path("configs/campaigns/frozen-driver-2026-10-28.json")
    spec = json.loads(manifest.read_text())
    checkpoint = Path(spec["checkpoint"])
    config = read_config(checkpoint)
    if (spec["partition"] != "development" or spec["held_out_evaluation"]
            or spec["user_only_oracle_access"] or spec["training_during_campaign"]):
        raise ValueError("Only the declared public development evaluation is authorized")
    if (sha256(checkpoint / "model.zip") != spec["checkpoint_sha256"]
            or config["protocol_id"] != spec["protocol_id"]
            or Partitions().fingerprint != spec["partition_sha256"]
            or sha256(Path(spec["checkpoint_selection_artifact"])) != spec["checkpoint_selection_sha256"]):
        raise ValueError("Checkpoint, selection or protocol declaration changed")
    root = Path("runs/campaigns") / spec["id"]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest, root / "config.json")
    shutil.copy2(__file__, root / "runner.py")
    shutil.copy2("scripts/summarize_frozen_driver.py", root / "summarize.py")
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    completed, native, internal = [], 0, 0
    started = time.perf_counter()

    def status(state: str, **extra) -> None:
        write_json(root / "status.json", {"status": state, "completed": completed,
            "native_frames": native, "internal_physics_frames": internal,
            "wall_time_s": time.perf_counter() - started, **extra})

    status("running")
    try:
        for condition in spec["conditions"]:
            if sha256(checkpoint / "model.zip") != spec["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint changed")
            command = [sys.executable, "-m", "cars.cli", "research", "evaluate",
                "--method", condition["method"], "--model", str(checkpoint),
                "--episodes", str(spec["tracks_per_condition"]), "--run-id", condition["run_id"],
                "--candidates", str(condition["candidates"]), "--planning-seed", str(condition["planning_seed"]),
                "--horizon", str(spec["horizon"]), "--steer-sigma", str(spec["steer_sigma"]),
                "--gas-sigma", str(spec["gas_sigma"]), "--brake-sigma", str(spec["brake_sigma"])]
            write_json(root / (condition["name"] + "_command.json"), command)
            with (root / (condition["name"] + ".log")).open("x") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                status("running", current=condition["name"], child_pid=process.pid)
                if process.wait() != 0:
                    raise RuntimeError(f"Condition {condition['name']} failed; log retained")
            run = Path("runs") / condition["directory"] / condition["run_id"]
            shutil.copy2(manifest, run / "campaign.json")
            cost = json.loads((run / "cost.json").read_text())
            native += cost["native_frames"]
            internal += cost["internal_physics_frames"]
            completed.append(condition["name"])
            status("running")
            print(f"Completed {condition['name']}: {json.loads((run / 'summary.json').read_text())}", flush=True)
            if native > spec["native_frame_cap"] or internal > spec["internal_physics_frame_cap"]:
                raise ValueError("Declared resource cap exceeded")
        status("evaluations_complete")
        subprocess.run([sys.executable, str(root / "summarize.py"), "--campaign", str(root),
                        "--output", spec["report_directory"]], env=environment, check=True)
        status("complete", report=spec["report_directory"])
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
