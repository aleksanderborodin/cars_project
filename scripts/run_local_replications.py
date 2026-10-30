"""Run independently seeded local baselines serially from frozen production source."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from cars.env.partitions import Partitions
from cars.research.driver_config import DriverTrainingConfig
from cars.research.frozen_code import freeze_sources, source_hashes
from cars.training import sha256, write_json


def validate(spec: dict) -> None:
    if spec["partition_sha256"] != Partitions().fingerprint:
        raise ValueError("Partition commitment changed")
    if sha256(Path(spec["reference_config"])) != spec["reference_config_sha256"]:
        raise ValueError("Reference recipe changed")
    reference = DriverTrainingConfig(**json.loads(Path(spec["reference_config"]).read_text())).resolved()
    if [arm["seed"] for arm in spec["arms"]] != [1, 2]:
        raise ValueError("The declaration must contain both fixed independent seeds")
    for arm in spec["arms"]:
        declaration = Path(arm["manifest"])
        if sha256(declaration) != arm["manifest_sha256"]:
            raise ValueError("Child declaration changed")
        child = json.loads(declaration.read_text())
        config = Path(child["training_config"])
        if sha256(config) != child["training_config_sha256"]:
            raise ValueError("Child training config changed")
        trial = DriverTrainingConfig(**json.loads(config.read_text())).resolved()
        if {k for k in reference if reference[k] != trial[k]} != {"seed"} or trial["seed"] != arm["seed"]:
            raise ValueError("Only the training seed may change from the retained recipe")
        if child["stop_at_native_frames"] != 2_000_000 or child["held_out_evaluation"] or child["user_only_oracle_access"]:
            raise ValueError("Unexpected training/access scope")
    if source_hashes(Path.cwd()) != spec["source_sha256"]:
        raise ValueError("Production source changed before launch")
    for name, expected in spec["script_sha256"].items():
        if sha256(name) != expected:
            raise ValueError("Frozen runner/reporter changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    validate(spec)
    if args.check:
        print("Valid: only seeds 1/2 differ; source, declarations, config and access scope match")
        return
    root = Path("runs/campaigns") / spec["id"]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.manifest, root / "config.json")
    shutil.copy2(__file__, root / "runner.py")
    environment = freeze_sources(Path.cwd(), root / "source", spec["source_sha256"])
    for source, name in (("scripts/run_driver_block.py", "block_runner.py"),
                         ("scripts/summarize_driver_block.py", "reporter.py")):
        shutil.copy2(source, root / name)
    # Freeze both arms now: later edits in the working tree cannot change seed 2.
    executions = {}
    for arm in spec["arms"]:
        child = json.loads(Path(arm["manifest"]).read_text())
        config = root / f"seed{arm['seed']}_training.json"
        shutil.copy2(child["training_config"], config)
        child["training_config"] = str(config.resolve())
        execution = root / f"seed{arm['seed']}_manifest.json"
        write_json(execution, child)
        executions[arm["seed"]] = (child, execution)
    rows = []
    started = time.perf_counter()
    process = None
    try:
        for arm in spec["arms"]:
            if shutil.disk_usage(root).free < spec["minimum_free_disk_bytes"]:
                raise RuntimeError("Storage reserve reached")
            child, execution = executions[arm["seed"]]
            with (root / f"seed{arm['seed']}.log").open("x") as log:
                process = subprocess.Popen([sys.executable, str(root / "block_runner.py"),
                    "--manifest", str(execution), "--reporter", str(root / "reporter.py")],
                    env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    write_json(root / "status.json", {"status": "running", "active_seed": arm["seed"],
                        "active_campaign": child["id"], "child_pid": process.pid,
                        "completed": rows, "wall_seconds": time.perf_counter() - started})
                    time.sleep(5)
                if process.returncode:
                    raise RuntimeError(f"Seed {arm['seed']} failed; see saved log")
            if source_hashes(root / "source") != spec["source_sha256"]:
                raise RuntimeError("Frozen production source changed")
            campaign = Path("runs/campaigns") / child["id"]
            state = json.loads((campaign / "status.json").read_text())
            if state["status"] != "complete":
                raise RuntimeError("Child campaign did not complete")
            rows.append({"seed": arm["seed"], "campaign": str(campaign), "status": state,
                         "selection": json.loads((campaign / "selection.json").read_text())})
            write_json(root / "results.json", rows)
        write_json(root / "status.json", {"status": "complete", "completed": rows,
                                         "wall_seconds": time.perf_counter() - started})
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        write_json(root / "status.json", {"status": "failed", "error": str(error), "completed": rows})
        raise


if __name__ == "__main__":
    main()
