"""Execute the frozen 200-track comparison from an isolated source snapshot."""
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


def completed_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as stream:
        return [row for row in csv.DictReader(stream) if row.get("steps", "").isdigit()
                and row.get("decisions", "").isdigit() and row.get("model_transitions", "").isdigit()]


def verify_pilot(reference: Path, current: Path, count: int = 20) -> None:
    for i in range(count):
        with np.load(reference / f"episodes/{i:04d}.npz") as old, np.load(current / f"episodes/{i:04d}.npz") as new:
            keys = ["action", "reward", "frames", "terminated", "truncated", "evaluation_offroad_frame_indices"]
            keys += [key for key in old if key.startswith("observation_")]
            for key in keys:
                np.testing.assert_array_equal(new[key], old[key], err_msg=f"pilot episode {i}, {key}")


def main() -> None:
    manifest = Path("configs/campaigns/broad-driver-2026-10-28.json")
    spec = json.loads(manifest.read_text())
    if (spec["partition"] != "development" or spec["held_out_evaluation"] or
            spec["user_only_oracle_access"] or spec["training_during_campaign"]):
        raise ValueError("Declared development-only evaluation required")
    checkpoint = Path(spec["checkpoint"])
    if sha256(checkpoint / "model.zip") != spec["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    if Partitions().fingerprint != spec["partition_sha256"] or spec["tracks_per_condition"] != 200:
        raise ValueError("Unexpected development manifest")
    for item in spec["required_artifacts"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Required artifact changed: {item['path']}")
    if shutil.disk_usage(".").free < spec["minimum_free_bytes"]:
        raise RuntimeError("Insufficient declared storage reserve")
    root = Path("runs/campaigns") / spec["id"]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest, root / "config.json")
    shutil.copy2(__file__, root / "runner.py")
    shutil.copy2("scripts/summarize_broad_driver.py", root / "summarize.py")
    environment = freeze_sources(Path.cwd(), root / "source", spec["source_sha256"])
    imported = subprocess.check_output([sys.executable, "-c", "import cars.env.protocol; print(cars.env.protocol.__file__)"],
                                       env=environment, text=True).strip()
    if not Path(imported).is_relative_to((root / "source").resolve()):
        raise RuntimeError("Child import did not use frozen source")
    write_json(root / "import_check.json", {"protocol_module": imported, "frozen_source": True})
    started = time.perf_counter()
    completed, native, internal, process = [], 0, 0, None

    def status(state: str, **extra) -> None:
        temporary = root / "status.tmp.json"
        write_json(temporary, {"status": state, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "completed": completed, "completed_condition_native_frames": native,
            "completed_condition_internal_physics_frames": internal, "wall_time_s": time.perf_counter() - started, **extra})
        temporary.replace(root / "status.json")

    try:
        for condition in spec["conditions"]:
            if source_hashes(root / "source") != spec["source_sha256"]:
                raise RuntimeError("Campaign source snapshot changed")
            if sha256(checkpoint / "model.zip") != spec["checkpoint_sha256"]:
                raise RuntimeError("Frozen checkpoint changed")
            path = Path("runs") / condition["directory"] / condition["run_id"]
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
            command = [sys.executable, "-m", "cars.cli", "research", "evaluate", "--method", condition["method"],
                "--model", str(checkpoint), "--episodes", str(spec["tracks_per_condition"]), "--run-id", condition["run_id"]]
            if condition["method"] == "policy-search":
                command += ["--candidates", str(condition["candidates"]), "--planning-seed", str(spec["planning_seed"]),
                    "--horizon", str(spec["horizon"]), "--steer-sigma", str(spec["steer_sigma"]),
                    "--gas-sigma", str(spec["gas_sigma"]), "--brake-sigma", str(spec["brake_sigma"]),
                    "--observation-renderer", spec["observation_renderer"]]
            if condition["method"] == "policy-bias":
                command += ["--action-bias", *map(str, condition["bias"])]
            write_json(root / (condition["name"] + "_command.json"), command)
            parity = False
            with (root / (condition["name"] + ".log")).open("x") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                while True:
                    rows = completed_rows(path / "metrics.csv")
                    if len(rows) >= 20 and not parity:
                        verify_pilot(Path(condition["pilot_reference"]), path)
                        parity = True
                        write_json(root / (condition["name"] + "_pilot_replay.json"),
                                   {"episodes": 20, "exact_observation_action_reward_replay": True, "extra_environment_frames": 0})
                    current_native = sum(int(row["steps"]) for row in rows)
                    current_internal = sum(int(row["model_transitions"]) for row in rows) * spec["action_repeat"]
                    status("running", current=condition["name"], child_pid=process.pid,
                        current_completed_episodes=len(rows), episodes_per_condition=spec["tracks_per_condition"],
                        current_completed_episode_native_frames=current_native,
                        current_completed_episode_internal_physics_frames=current_internal,
                        pilot_replay_passed=parity)
                    if native + current_native > spec["native_frame_cap"] or internal + current_internal > spec["internal_physics_frame_cap"]:
                        raise RuntimeError("Declared resource cap exceeded")
                    code = process.poll()
                    if code is not None:
                        if code != 0:
                            raise RuntimeError(f"Condition {condition['name']} failed; log retained")
                        break
                    time.sleep(10)
            if not parity:
                verify_pilot(Path(condition["pilot_reference"]), path)
            config = read_config(path)
            if config["source_sha256"] != spec["source_sha256"] or config["versions"] != spec["versions"]:
                raise RuntimeError("Evaluated source or dependencies differ from declaration")
            shutil.copy2(manifest, path / "campaign.json")
            cost = json.loads((path / "cost.json").read_text())
            native += cost["native_frames"]
            internal += cost["internal_physics_frames"]
            completed.append(condition["name"])
            print(f"Completed {condition['name']}; {spec['tracks_per_condition']} tracks", flush=True)
        if native > spec["native_frame_cap"] or internal > spec["internal_physics_frame_cap"]:
            raise RuntimeError("Final resource cap exceeded")
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
