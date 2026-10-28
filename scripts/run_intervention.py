"""Run a declared four-cell public geometry intervention from frozen sources."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from cars.env.partitions import Partitions
from cars.research.artifacts import read_config
from cars.research.frozen_code import freeze_sources, source_hashes
from cars.training import sha256, write_json


def verify_negative_control(reference: Path, current: Path, count: int) -> None:
    for episode in range(count):
        with np.load(reference / f"episodes/{episode:04d}.npz") as old, np.load(current / f"episodes/{episode:04d}.npz") as new:
            keys = ["action", "reward", "frames", "terminated", "truncated", "evaluation_offroad_frame_indices"]
            keys += [key for key in old if key.startswith("observation_")]
            for key in keys:
                np.testing.assert_array_equal(old[key], new[key], err_msg=f"episode {episode}: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    if (spec["partition"] != "development" or spec["held_out_evaluation"]
            or spec["user_only_oracle_access"] or spec["training_during_campaign"]):
        raise ValueError("Public development-only diagnostic required")
    if spec["partition_sha256"] != Partitions().fingerprint:
        raise ValueError("Partition changed")
    count = spec["tracks_per_condition"]
    if not 1 <= count <= Partitions().development_count:
        raise ValueError("Invalid development count")
    cells = [(c["reference"], c["continuation"]) for c in spec["conditions"]]
    if len(cells) != 4 or set(cells) != {(a, b) for a in ("reconstructed", "oracle_visible") for b in ("prior", "oracle")}:
        raise ValueError("Four distinct preregistered cells required")
    if cells[0] != ("reconstructed", "prior"):
        raise ValueError("Negative-control replay must pass first")
    # The native TimeLimit plus fixed rollout budget is an a priori work bound.
    maximum_decisions = (1000 + spec["action_repeat"] - 1) // spec["action_repeat"]
    bound = count * len(cells) * maximum_decisions * spec["candidates"] * spec["horizon"] * spec["action_repeat"]
    if 1000 * count * len(cells) > spec["native_frame_cap"] or bound > spec["internal_physics_frame_cap"]:
        raise ValueError("Manifest cannot fit its declared resource cap")
    for artifact in spec["required_artifacts"]:
        if sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError(f"Required artifact changed: {artifact['path']}")
    if sha256(Path(spec["checkpoint"]) / "model.zip") != spec["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    if shutil.disk_usage(".").free < spec["minimum_free_bytes"]:
        raise RuntimeError("Insufficient storage reserve")
    root = Path("runs/campaigns") / spec["id"]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.manifest, root / "config.json")
    shutil.copy2(__file__, root / "runner.py")
    shutil.copy2("scripts/summarize_intervention.py", root / "summarize.py")
    environment = freeze_sources(Path.cwd(), root / "source", spec["source_sha256"])
    imported = subprocess.check_output([sys.executable, "-c", "import cars.planning.intervention; print(cars.planning.intervention.__file__)"],
                                       env=environment, text=True).strip()
    if not Path(imported).is_relative_to((root / "source").resolve()):
        raise RuntimeError("Child did not import frozen source")
    write_json(root / "import_check.json", {"module": imported, "frozen_source": True})
    started = time.perf_counter()
    completed, native, internal, process = [], 0, 0, None

    def status(state: str, **extra) -> None:
        temporary = root / "status.tmp.json"
        write_json(temporary, {"status": state, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "completed": completed, "completed_condition_native_frames": native,
            "completed_condition_internal_physics_frames": internal,
            "wall_time_s": time.perf_counter() - started, **extra})
        temporary.replace(root / "status.json")

    try:
        for cell in spec["conditions"]:
            if source_hashes(root / "source") != spec["source_sha256"]:
                raise RuntimeError("Frozen source changed")
            if sha256(Path(spec["checkpoint"]) / "model.zip") != spec["checkpoint_sha256"]:
                raise RuntimeError("Checkpoint changed")
            command = [sys.executable, "-m", "cars.cli", "research", "evaluate", "--method", "policy-intervention",
                "--model", spec["checkpoint"], "--episodes", str(count), "--run-id", cell["run_id"],
                "--reference", cell["reference"], "--continuation", cell["continuation"],
                "--candidates", str(spec["candidates"]), "--horizon", str(spec["horizon"]),
                "--planning-seed", str(spec["planning_seed"]), "--observation-renderer", spec["observation_renderer"],
                "--steer-sigma", str(spec["steer_sigma"]), "--gas-sigma", str(spec["gas_sigma"]),
                "--brake-sigma", str(spec["brake_sigma"])]
            write_json(root / (cell["name"] + "_command.json"), command)
            path = Path("runs/policy_geometry_intervention") / cell["run_id"]
            if path.exists():
                raise FileExistsError(path)
            with (root / (cell["name"] + ".log")).open("x") as log:
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                while True:
                    metrics = path / "metrics.csv"
                    rows = []
                    if metrics.exists():
                        with metrics.open() as stream:
                            rows = list(csv.DictReader(stream))
                    status("running", current=cell["name"], child_pid=process.pid,
                           current_completed_episodes=len(rows), episodes_per_condition=count)
                    code = process.poll()
                    if code is not None:
                        if code != 0:
                            raise RuntimeError(f"Condition {cell['name']} failed; log retained")
                        break
                    time.sleep(5)
            config = read_config(path)
            if config["source_sha256"] != spec["source_sha256"] or config["versions"] != spec["versions"]:
                raise RuntimeError("Executed source/dependencies differ")
            if cell == spec["conditions"][0]:
                verify_negative_control(Path(spec["negative_control_reference"]), path, count)
                write_json(root / "negative_control_replay.json", {"episodes": count,
                    "exact_observation_action_reward_replay": True, "extra_environment_frames": 0})
            shutil.copy2(args.manifest, path / "campaign.json")
            cost = json.loads((path / "cost.json").read_text())
            native += cost["native_frames"]
            internal += cost["internal_physics_frames"]
            completed.append(cell["name"])
            print(f"Completed {cell['name']}: {count} development episodes", flush=True)
        if native > spec["native_frame_cap"] or internal > spec["internal_physics_frame_cap"]:
            raise RuntimeError("Resource cap exceeded")
        status("evaluations_complete")
        subprocess.run([sys.executable, str(root / "summarize.py"), "--campaign", str(root),
                        "--output", spec["report_directory"]], env=environment, check=True)
        status("complete", report=spec["report_directory"])
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
