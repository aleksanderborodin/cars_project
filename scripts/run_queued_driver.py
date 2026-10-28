"""Freeze a declared training probe now; run it after the broad timing campaign."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from cars.env.partitions import Partitions
from cars.research.artifacts import read_config
from cars.research.driver_config import DriverTrainingConfig
from cars.research.frozen_code import freeze_sources, source_hashes
from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate declaration only, without creating runs")
    args = parser.parse_args()
    repository = Path.cwd()
    spec = json.loads(args.manifest.read_text())
    config = Path(spec["training_config"])
    reference = json.loads(Path("configs/training/recurrent-local-v2-s0.json").read_text())
    trial = json.loads(config.read_text())
    DriverTrainingConfig(**trial)
    if (spec["training_partition"] != "train" or spec["evaluation_partition"] != "development"
            or spec["held_out_evaluation"] or spec["user_only_oracle_access"]
            or spec["partition_sha256"] != Partitions().fingerprint
            or sha256(config) != spec["training_config_sha256"]):
        raise ValueError("Development training declaration mismatch")
    differences = {key for key in reference.keys() | trial.keys() if reference.get(key) != trial.get(key)}
    if differences != {"learning_rate"} or trial["learning_rate"] != 3e-5:
        raise ValueError("This queue only authorizes the declared learning-rate-only probe")
    if source_hashes(repository) != spec["new_source_sha256"]:
        raise ValueError("Source changed before preparing queued probe")
    dependency = Path("runs/campaigns/broad_driver_20261028")
    dependency_spec = json.loads((dependency / "config.json").read_text())
    if dependency_spec["source_sha256"] != spec["new_source_sha256"]:
        raise ValueError("Expected tested broad-campaign source")
    root = Path("runs/queues") / spec["id"]
    if root.exists() or (Path("runs/campaigns") / spec["id"]).exists():
        raise FileExistsError("Queue/campaign already exists; do not overwrite")
    if args.check:
        print("Declaration valid; only learning rate changes; no run created")
        return
    if shutil.disk_usage(repository).free < 1_000_000_000:
        raise RuntimeError("Insufficient storage reserve")
    root.mkdir(parents=True, exist_ok=False)
    environment = freeze_sources(repository, root / "source", spec["new_source_sha256"])
    shutil.copytree("tests", root / "source/tests", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for original, name in ((args.manifest, "declaration.json"), (config, "training_config.json"),
            (Path(__file__), "queue_runner.py"), (Path("scripts/run_driver_block.py"), "driver_runner.py"),
            (Path("scripts/summarize_driver_block.py"), "reporter.py")):
        shutil.copy2(original, root / name)
    execution = {**spec, "training_config": str((root / "training_config.json").resolve()),
                 "original_declaration_sha256": sha256(args.manifest)}
    write_json(root / "execution_manifest.json", execution)
    hashes = {str(path.relative_to(root)): sha256(path) for path in root.rglob("*")
              if path.is_file() and "__pycache__" not in path.parts}
    write_json(root / "prepared_hashes.json", hashes)
    started = time.perf_counter()
    process = None

    def status(state: str, **extra) -> None:
        temporary = root / "status.tmp.json"
        write_json(temporary, {"status": state, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "queue_pid": os.getpid(), "dependency": str(dependency),
            "wall_time_s": time.perf_counter() - started, **extra})
        temporary.replace(root / "status.json")

    def verify_frozen() -> None:
        if source_hashes(root / "source") != spec["new_source_sha256"]:
            raise RuntimeError("Queued source snapshot changed")
        for name, expected in hashes.items():
            if sha256(root / name) != expected:
                raise RuntimeError(f"Prepared artifact changed: {name}")

    def run(command: list[str], log_name: str, stage: str) -> None:
        nonlocal process
        with (root / log_name).open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                       start_new_session=True)
            while process.poll() is None:
                status(stage, child_pid=process.pid, log=str(root / log_name))
                time.sleep(10)
            if process.returncode:
                raise RuntimeError(f"{stage} failed; inspect public {log_name}")

    try:
        while True:
            state = json.loads((dependency / "status.json").read_text())
            if state["status"] == "failed":
                raise RuntimeError("Preceding broad campaign failed; stop queue for review")
            if state["status"] == "complete":
                break
            updated = datetime.fromisoformat(state["updated_utc"])
            if (datetime.now(timezone.utc) - updated).total_seconds() > 600:
                raise RuntimeError("Preceding campaign heartbeat is stale; do not start competing work")
            status("waiting_for_broad_campaign", dependency_stage=state["status"])
            time.sleep(30)
        verify_frozen()
        # Full regression checks are serialized too: several tests train/race.
        run(["uv", "run", "--locked", "--no-sync", "pytest", "-q", str(root / "source/tests"),
             "--rootdir", str(root / "source")], "tests.log", "testing_frozen_source")
        verify_frozen()
        if shutil.disk_usage(repository).free < 1_000_000_000:
            raise RuntimeError("Insufficient storage reserve before training")
        run([sys.executable, str(root / "driver_runner.py"), "--manifest", str(root / "execution_manifest.json"),
             "--reporter", str(root / "reporter.py")], "campaign.log", "running_training_probe")
        verify_frozen()
        campaign = Path("runs/campaigns") / spec["id"]
        result = json.loads((campaign / "status.json").read_text())
        if result["status"] != "complete":
            raise RuntimeError("Training campaign did not complete")
        records = json.loads((campaign / "checkpoint_results.json").read_text())
        for path in [Path("runs/driver_training") / spec["training_run_id"],
                     *(Path(record["evaluation"]) for record in records)]:
            actual = read_config(path)
            if (actual["source_sha256"] != spec["new_source_sha256"]
                    or actual["versions"] != dependency_spec["versions"]):
                raise RuntimeError("Completed run differs from frozen source/dependencies")
        status("complete", campaign=str(campaign), report=spec["report_directory"])
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
