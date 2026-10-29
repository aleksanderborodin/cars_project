"""Execute the frozen 20-track reference-tail replication; replay checks use no extra races."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from cars.research.frozen_code import freeze_sources, source_hashes
from cars.training import sha256, write_json
from reference_replication_common import CONDITIONS, verify_config, verify_declaration, verify_replay, verified_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    verify_declaration(spec)
    for name, reference in CONDITIONS.items():
        verify_config(Path(spec["conditions"][name]["pilot_run"]), spec, reference)
    verify_config(Path(spec["context_run"]), spec, None)
    root = Path("runs/campaigns") / spec["id"]
    for condition in spec["conditions"].values():
        if (Path("runs/policy_geometry_intervention") / condition["run_id"]).exists():
            raise FileExistsError(condition["run_id"])
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.manifest, root / "config.json")
    for script in ("run_reference_replication.py", "summarize_reference_replication.py", "reference_replication_common.py"):
        shutil.copy2(Path(__file__).parent / script, root / script)
    environment = freeze_sources(Path.cwd(), root / "source", spec["source_sha256"])
    imported = subprocess.check_output([sys.executable, "-c", "import cars.planning.intervention; print(cars.planning.intervention.__file__)"],
                                       env=environment, text=True).strip()
    if not Path(imported).is_relative_to((root / "source").resolve()):
        raise RuntimeError("Child import is not frozen")
    write_json(root / "import_check.json", {"module": imported, "frozen_source": True})
    started = time.perf_counter()
    completed, replay_checks = [], {}
    native = internal = 0
    process = None

    def status(state: str, **extra) -> None:
        temporary = root / "status.tmp.json"
        write_json(temporary, {"status": state, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "completed": completed, "replay_checks": replay_checks,
            "completed_native_frames": native, "completed_internal_physics_frames": internal,
            "wall_time_s": time.perf_counter() - started, **extra})
        temporary.replace(root / "status.json")

    try:
        for name, reference in CONDITIONS.items():
            condition = spec["conditions"][name]
            if (source_hashes(root / "source") != spec["source_sha256"]
                    or sha256(Path(spec["checkpoint"]) / "model.zip") != spec["checkpoint_sha256"]):
                raise RuntimeError("Frozen sources/checkpoint changed")
            path = Path("runs/policy_geometry_intervention") / condition["run_id"]
            command = [sys.executable, "-m", "cars.cli", "research", "evaluate", "--method", "policy-intervention",
                "--model", spec["checkpoint"], "--run-id", condition["run_id"], "--episodes", str(spec["episodes"]),
                "--reference", reference, "--continuation", "oracle"]
            for key in ("candidates", "horizon", "planning_seed", "observation_renderer", "steer_sigma", "gas_sigma", "brake_sigma"):
                command += ["--" + key.replace("_", "-"), str(spec["search"][key])]
            write_json(root / (name + "_command.json"), command)
            replay_checks[name] = 0
            phase_started = time.perf_counter()
            with (root / (name + ".log")).open("x") as log:
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                while True:
                    rows = []
                    if (path / "metrics.csv").exists():
                        with (path / "metrics.csv").open() as stream:
                            rows = list(csv.DictReader(stream))
                    # metrics are published after each complete trajectory; a writer
                    # can briefly truncate the CSV, so never infer failure from zero rows.
                    for episode in range(replay_checks[name], min(len(rows), spec["replay_episodes"])):
                        verify_replay(Path(condition["pilot_run"]), path, episode)
                        replay_checks[name] += 1
                    if (path / "config.json").exists():
                        verify_config(path, spec, reference, fresh=True)
                    status("running", phase=name, child_pid=process.pid, current_completed_episodes=len(rows),
                        phase_episodes=spec["episodes"], phase_wall_time_s=time.perf_counter() - phase_started)
                    code = process.poll()
                    if code is not None:
                        if code:
                            raise RuntimeError(f"{name} failed; log retained")
                        break
                    time.sleep(5)
            verified_table(path, spec)
            if replay_checks[name] != spec["replay_episodes"]:
                raise RuntimeError("Missing replay checks")
            shutil.copy2(root / "config.json", path / "campaign.json")
            cost = json.loads((path / "cost.json").read_text())
            native += cost["native_frames"]
            internal += cost["internal_physics_frames"]
            if native > spec["native_frame_cap"] or internal > spec["internal_physics_frame_cap"]:
                raise RuntimeError("Resource cap exceeded")
            completed.append(name)
            print(f"Completed {name}: {spec['episodes']} episodes, five exact replays", flush=True)
        verify_declaration(spec)
        write_json(root / "replay_check.json", {"conditions": replay_checks, "exact_replay": True,
            "extra_environment_frames": 0, "excluded_fields": ["latency_ms", "sensor_ms", "policy_ms"]})
        status("evaluations_complete")
        subprocess.run([sys.executable, str(root / "summarize_reference_replication.py"), "--campaign", str(root),
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
