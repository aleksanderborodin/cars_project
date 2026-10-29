"""Freeze and run a replay-gated sequence-search smoke or comparison campaign."""
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
from sequence_campaign_common import complete_table, verify_episode_replay, verify_manifest, verify_run, verify_work


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    verify_manifest(spec)
    if spec["stage"] == "comparison":
        previous = Path(spec["prerequisite_campaign"])
        if (json.loads((previous / "status.json").read_text())["status"] != "complete"
                or sha256(previous / "config.json") != spec["prerequisite_manifest_sha256"]
                or json.loads((previous / "config.json").read_text())["source_sha256"] != spec["source_sha256"]):
            raise ValueError("Unchanged implementation and completed smoke campaign required")
    for mode, path in spec["historical_context"].items():
        verify_run(Path(path), spec, mode=mode)
    root = Path("runs/campaigns") / spec["id"]
    for condition in spec["conditions"]:
        if (Path("runs") / condition["directory"] / condition["run_id"]).exists():
            raise FileExistsError(condition["run_id"])
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.manifest, root / "config.json")
    for name in ("run_sequence_search.py", "summarize_sequence_search.py", "sequence_campaign_common.py"):
        shutil.copy2(Path(__file__).parent / name, root / name)
    environment = freeze_sources(Path.cwd(), root / "source", spec["source_sha256"])
    imported = subprocess.check_output([sys.executable, "-c", "import cars.planning.policy_sequence_search; print(cars.planning.policy_sequence_search.__file__)"],
        env=environment, text=True).strip()
    if not Path(imported).is_relative_to((root / "source").resolve()):
        raise RuntimeError("Child does not import frozen sources")
    write_json(root / "import_check.json", {"module": imported, "frozen_source": True})
    started = time.perf_counter()
    completed, checks = [], {}
    native = internal = 0
    child = None

    def status(state: str, **extra) -> None:
        temporary = root / "status.tmp.json"
        write_json(temporary, {"status": state, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "completed": completed, "checks": checks, "completed_native_frames": native,
            "completed_internal_physics_frames": internal, "wall_time_s": time.perf_counter() - started, **extra})
        temporary.replace(root / "status.json")

    try:
        for condition in spec["conditions"]:
            name, episodes = condition["name"], condition["episodes"]
            if (source_hashes(root / "source") != spec["source_sha256"]
                    or sha256(Path(spec["checkpoint"]) / "model.zip") != spec["checkpoint_sha256"]):
                raise RuntimeError("Frozen code/model changed")
            path = Path("runs") / condition["directory"] / condition["run_id"]
            command = [sys.executable, "-m", "cars.cli", "research", "evaluate", "--method", condition["method"],
                "--model", spec["checkpoint"], "--run-id", condition["run_id"], "--episodes", str(episodes)]
            for key in ("candidates", "horizon", "planning_seed", "steer_sigma", "gas_sigma", "brake_sigma", "observation_renderer"):
                command += ["--" + key.replace("_", "-"), str(spec["search"][key])]
            if condition["mode"] in ("colored", "icem"):
                optimizer = spec["optimizers"][condition["mode"]]
                command += ["--optimizer-noise-beta", str(optimizer["noise_beta"]), "--residual-bound-sigma", str(optimizer["residual_bound_sigma"])]
            write_json(root / f"{name}_command.json", command)
            checks[name] = {"replayed_episodes": 0}
            phase_start = time.perf_counter()
            with (root / f"{name}.log").open("x") as log:
                child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                while True:
                    rows = []
                    if (path / "metrics.csv").exists():
                        with (path / "metrics.csv").open() as stream:
                            rows = list(csv.DictReader(stream))
                    if (path / "config.json").exists():
                        verify_run(path, spec, mode=condition["mode"], fresh=True, episodes=episodes)
                    for episode in range(checks[name]["replayed_episodes"], min(len(rows), condition.get("replay_episodes", 0))):
                        verify_episode_replay(Path(condition["replay_reference"]), path, episode)
                        checks[name]["replayed_episodes"] += 1
                    status("running", phase=name, child_pid=child.pid, current_completed_episodes=len(rows),
                        phase_episodes=episodes, phase_wall_time_s=time.perf_counter() - phase_start)
                    code = child.poll()
                    if code is not None:
                        if code:
                            raise RuntimeError(f"{name} failed; logs retained")
                        break
                    time.sleep(5)
            complete_table(path, spec, episodes)
            if checks[name]["replayed_episodes"] != condition.get("replay_episodes", 0):
                raise RuntimeError("Missing replay gate")
            checks[name].update(verify_work(path, spec, condition["mode"]))
            shutil.copy2(root / "config.json", path / "campaign.json")
            cost = json.loads((path / "cost.json").read_text())
            native += cost["native_frames"]
            internal += cost["internal_physics_frames"]
            if native > spec["native_frame_cap"] or internal > spec["internal_physics_frame_cap"]:
                raise RuntimeError("Resource cap exceeded")
            completed.append(name)
            print(f"Completed {name}: {episodes} episodes; replay/work checks passed", flush=True)
        verify_manifest(spec)
        write_json(root / "checks.json", checks)
        status("evaluations_complete")
        subprocess.run([sys.executable, str(root / "summarize_sequence_search.py"), "--campaign", str(root),
                        "--output", spec["report_directory"]], env=environment, check=True)
        status("complete", report=spec["report_directory"])
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
