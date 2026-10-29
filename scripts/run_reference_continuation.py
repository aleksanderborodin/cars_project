"""Freeze and execute a replay-gated, five-track reference-continuation pilot."""
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
from cars.env.protocol import ProtocolConfig
from cars.planning.reference_extension import extend_true_reference
from cars.planning.visible_reference import visible_reference
from cars.research.artifacts import read_config
from cars.research.frozen_code import freeze_sources, source_hashes
from cars.training import sha256, write_json


def verify_geometry(spec: dict, root: Path) -> None:
    """Recorded public geometry only; no environment steps or candidate values."""
    protocol = ProtocolConfig(**spec["protocol"])
    states = fallbacks = 0
    for episode in range(spec["episodes"]):
        with np.load(Path(spec["control_run"]) / f"episodes/{episode:04d}.npz") as data:
            for centre in data["observation_map_centre"]:
                visible = visible_reference(centre, protocol.radius, protocol.resolution)
                states += 1
                if visible is None:
                    fallbacks += 1
                    continue
                extended = extend_true_reference(visible, centre)
                np.testing.assert_array_equal(extended[:len(visible)], visible)
                assert len(extended) == len(visible) + 100 and np.isfinite(extended).all()
    write_json(root / "geometry_check.json", {"recorded_states": states, "visible_reference_fallbacks": fallbacks,
        "all_prefixes_and_lengths_match": True, "new_environment_frames": 0, "internal_physics_frames": 0})


def verify_replay(reference: Path, replay: Path) -> None:
    with np.load(reference / "episodes/0000.npz") as old, np.load(replay / "episodes/0000.npz") as new:
        keys = ["action", "reward", "frames", "terminated", "truncated", "evaluation_offroad_frame_indices"]
        keys += [key for key in old if key.startswith("observation_")]
        for key in keys:
            np.testing.assert_array_equal(old[key], new[key], err_msg=key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    if (spec["partition"] != "development" or spec["held_out_evaluation"] or spec["private_oracle_access"]
            or spec["training_frames"] != 0 or spec["partition_sha256"] != Partitions().fingerprint):
        raise ValueError("Fixed public development evaluation only")
    if spec["episodes"] != 5 or spec["candidates"] != 7 or spec["horizon"] != 20 or spec["action_repeat"] != 4:
        raise ValueError("This pilot has a bounded fixed design")
    if spec["native_frame_cap"] != 6000 or spec["internal_physics_frame_cap"] != 840000:
        raise ValueError("Include both replay and new-condition caps")
    for item in spec["required_artifacts"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Required artifact changed: {item['path']}")
    actor = read_config(Path(spec["checkpoint"]))
    if actor["protocol"] != spec["actor_protocol"]:
        raise ValueError("Wrong actor inputs")
    for name in ("control_run", "context_run"):
        config = read_config(Path(spec[name]))
        driver = config["settings"]["driver"]
        if (config["protocol"] != spec["protocol"] or config["versions"] != spec["versions"]
                or config["settings"]["partition"] != "development"
                or driver["actor"]["checkpoint_sha256"] != spec["checkpoint_sha256"]
                or config["settings"]["user_only_oracle_monitor"]):
            raise ValueError("Unmatched historical control")
        for key in spec["search"]:
            if driver[key] != spec["search"][key]:
                raise ValueError(f"Changed common parameter: {key}")
    if read_config(Path(spec["control_run"]))["settings"]["driver"]["intervention"] != {
            "reference": "oracle_visible", "continuation": "oracle", "version": 1}:
        raise ValueError("Expected the earlier both-oracles control")
    root = Path("runs/campaigns") / spec["id"]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.manifest, root / "config.json")
    shutil.copy2(__file__, root / "runner.py")
    shutil.copy2("scripts/summarize_reference_continuation.py", root / "summarize.py")
    environment = freeze_sources(Path.cwd(), root / "source", spec["source_sha256"])
    imported = subprocess.check_output([sys.executable, "-c", "import cars.planning.reference_extension; print(cars.planning.reference_extension.__file__)"],
                                       env=environment, text=True).strip()
    if not Path(imported).is_relative_to((root / "source").resolve()):
        raise RuntimeError("Child import is not frozen")
    write_json(root / "import_check.json", {"module": imported, "frozen_source": True})
    started = time.perf_counter()
    completed, native, internal, process = [], 0, 0, None

    def status(state: str, **extra) -> None:
        temporary = root / "status.tmp.json"
        write_json(temporary, {"status": state, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "completed": completed, "completed_native_frames": native, "completed_internal_physics_frames": internal,
            "wall_time_s": time.perf_counter() - started, **extra})
        temporary.replace(root / "status.json")

    try:
        status("checking_recorded_geometry")
        verify_geometry(spec, root)
        for phase, reference, episodes in (("replay", "oracle_visible", 1), ("new", "oracle_continued", spec["episodes"])):
            if source_hashes(root / "source") != spec["source_sha256"]:
                raise RuntimeError("Frozen source changed")
            if sha256(Path(spec["checkpoint"]) / "model.zip") != spec["checkpoint_sha256"]:
                raise RuntimeError("Checkpoint changed")
            path = Path("runs/policy_geometry_intervention") / spec[phase + "_run_id"]
            if path.exists():
                raise FileExistsError(path)
            command = [sys.executable, "-m", "cars.cli", "research", "evaluate", "--method", "policy-intervention",
                "--model", spec["checkpoint"], "--run-id", spec[phase + "_run_id"], "--episodes", str(episodes),
                "--reference", reference, "--continuation", "oracle"]
            for key in ("candidates", "horizon", "planning_seed", "observation_renderer", "steer_sigma", "gas_sigma", "brake_sigma"):
                command += ["--" + key.replace("_", "-"), str(spec["search"][key])]
            write_json(root / (phase + "_command.json"), command)
            with (root / (phase + ".log")).open("x") as log:
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                while True:
                    rows = []
                    if (path / "metrics.csv").exists():
                        with (path / "metrics.csv").open() as stream:
                            rows = list(csv.DictReader(stream))
                    status("running", phase=phase, child_pid=process.pid,
                           current_completed_episodes=len(rows), phase_episodes=episodes)
                    code = process.poll()
                    if code is not None:
                        if code:
                            raise RuntimeError(f"{phase} failed; log retained")
                        break
                    time.sleep(5)
            config = read_config(path)
            if (config["source_sha256"] != spec["source_sha256"] or config["versions"] != spec["versions"]
                    or config["protocol"] != spec["protocol"]):
                raise RuntimeError("Executed source/dependencies/protocol mismatch")
            for key, expected in spec["search"].items():
                if config["settings"]["driver"][key] != expected:
                    raise RuntimeError(f"Executed parameter mismatch: {key}")
            if phase == "replay":
                verify_replay(Path(spec["control_run"]), path)
                write_json(root / "replay_check.json", {"episodes": 1, "exact_observation_action_reward_replay": True})
            shutil.copy2(args.manifest, path / "campaign.json")
            cost = json.loads((path / "cost.json").read_text())
            native += cost["native_frames"]
            internal += cost["internal_physics_frames"]
            if native > spec["native_frame_cap"] or internal > spec["internal_physics_frame_cap"]:
                raise RuntimeError("Resource cap exceeded")
            completed.append(phase)
            print(f"Completed {phase}: {episodes} episode(s)", flush=True)
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
