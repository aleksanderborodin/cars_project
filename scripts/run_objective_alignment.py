"""Freeze a score-only pilot, wait for the existing matrix, then execute serially."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from cars.env.partitions import Partitions
from cars.research.artifacts import read_config
from cars.research.frozen_code import freeze_sources, source_hashes
from cars.training import sha256, write_json


def replay(reference: Path, current: Path, episodes: int) -> None:
    for i in range(episodes):
        with np.load(reference / f"episodes/{i:04d}.npz") as a, np.load(current / f"episodes/{i:04d}.npz") as b:
            for name in a.files:
                if name.startswith("observation_") or name in ("action", "reward", "frames", "terminated", "truncated", "evaluation_offroad_frame_indices"):
                    np.testing.assert_array_equal(a[name], b[name], err_msg=f"Legacy parity {i}/{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    if (spec["tracks"] != 20 or spec["partition"] != "development" or spec["actor_seeds"] != [0, 1]
            or spec["penalty_scales"] != [1., .25, 0.] or spec["native_frame_cap"] != 124000
            or spec["internal_physics_frame_cap"] != 16800000 or spec["held_out_evaluation"]
            or spec["private_oracle_access"] or spec["partition_sha256"] != Partitions().fingerprint
            or (spec["candidates"], spec["horizon"], spec["planning_seed"], spec["observation_renderer"]) != (7, 20, 0, "sparse")
            or [a["seed"] for a in spec["actors"]] != [0, 1]):
        raise ValueError("Wrong fixed development pilot")
    if source_hashes(Path.cwd()) != spec["source_sha256"] or sha256(Path(__file__)) != spec["runner_sha256"]:
        raise ValueError("Declared source/controller changed")
    if sha256(Path(spec["reporter"])) != spec["reporter_sha256"]:
        raise ValueError("Declared report changed")
    for name, expected in spec["required_sha256"].items():
        if sha256(Path(name)) != expected: raise ValueError(f"Frozen input changed: {name}")
    for actor in spec["actors"]:
        cfg = read_config(Path(actor["checkpoint"]))
        chosen = json.loads(Path(actor["selection"]).read_text())["selected"]["checkpoint"]
        if Path(chosen).resolve() != Path(actor["checkpoint"]).resolve():
            raise ValueError("Standalone checkpoint selection differs")
        if cfg["protocol_id"] != spec["protocol_id"] or cfg["versions"] != spec["versions"] or cfg["settings"]["training"]["seed"] != actor["seed"]:
            raise ValueError("Wrong frozen standalone-selected actor")
    if args.check:
        print("Validated: two actors, three score settings, legacy gates, fixed budgets; no driving launched")
        return
    workspace = Path(spec["workspace"])
    workspace.mkdir(parents=True, exist_ok=False)
    root = workspace / "queue"
    root.mkdir()
    shutil.copy2(args.manifest, root / "config.json")
    shutil.copy2(__file__, root / "runner.py")
    shutil.copy2(spec["reporter"], root / "reporter.py")
    env = freeze_sources(Path.cwd(), root / "source", spec["source_sha256"])
    for name in ("tmp", "cache"): (workspace / name).mkdir()
    env.update(PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(workspace / "tmp"),
        XDG_CACHE_HOME=str(workspace / "cache"), MPLCONFIGDIR=str(workspace / "cache/matplotlib"),
        CUBLAS_WORKSPACE_CONFIG=":4096:8")
    completed, costs = [], []
    process = None
    started = time.perf_counter()

    def status(state: str, **extra) -> None:
        target = root / "status.tmp.json"
        write_json(target, {"status": state, "queue_pid": os.getpid(), "updated_epoch": time.time(),
            "completed": completed, "native_frames": sum(c["native_frames"] for c in costs),
            "internal_physics_frames": sum(c["internal_physics_frames"] for c in costs),
            "wall_seconds_including_dependency": time.perf_counter() - started, **extra})
        target.replace(root / "status.json")

    def execute(actor: dict, label: str, scale: float | None, episodes: int) -> Path:
        nonlocal process
        if shutil.disk_usage(workspace).free < spec["minimum_free_disk_bytes"]:
            raise RuntimeError("Pilot free-space floor reached before child launch")
        if source_hashes(root / "source") != spec["source_sha256"]:
            raise ValueError("Frozen pilot source changed")
        if sha256(Path(actor["checkpoint"]) / "model.zip") != actor["model_sha256"]:
            raise ValueError("Frozen actor changed")
        run_id = f"objective_20261030_s{actor['seed']}_{label}"
        method = "recurrent" if scale is None else "policy-objective"
        destination = workspace / "runs" / ("recurrent_ppo" if scale is None else "policy_objective_search") / run_id
        if destination.exists(): raise FileExistsError(destination)
        command = [sys.executable, "-m", "cars.cli", "research", "evaluate", "--method", method,
            "--model", actor["checkpoint"], "--episodes", str(episodes), "--run-id", run_id]
        if scale is not None:
            command += ["--penalty-scale", str(scale), "--candidates", "7", "--horizon", "20",
                "--planning-seed", "0", "--steer-sigma", "0.1", "--gas-sigma", "0.15",
                "--brake-sigma", "0.15", "--observation-renderer", "sparse"]
        write_json(root / (run_id + "_command.json"), command)
        with (root / (run_id + ".log")).open("x") as log:
            process = subprocess.Popen(command, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
            while process.poll() is None:
                if shutil.disk_usage(workspace).free < spec["minimum_free_disk_bytes"]:
                    raise RuntimeError("Pilot free-space floor reached")
                status("evaluating", active=run_id, child_pid=process.pid)
                time.sleep(5)
            if process.returncode: raise RuntimeError(f"Pilot child failed: {run_id}")
        costs.append(json.loads((destination / "cost.json").read_text()))
        if sum(c["native_frames"] for c in costs) > spec["native_frame_cap"] or sum(c["internal_physics_frames"] for c in costs) > spec["internal_physics_frame_cap"]:
            raise RuntimeError("Pilot budget exceeded")
        cfg = read_config(destination)
        drv = cfg["settings"]["driver"]
        if (cfg["versions"] != spec["versions"] or cfg["protocol_id"] != spec["protocol_id"]
                or drv.get("actor", drv).get("checkpoint_sha256") != actor["model_sha256"]):
            raise ValueError("Wrong evaluated identity")
        metrics = pd.read_csv(destination / "metrics.csv")
        if metrics.track_seed.tolist() != Partitions().seeds("development", episodes):
            raise ValueError("Wrong development subset")
        return destination

    try:
        dependency = Path(spec["dependency_status"])
        while True:
            state = json.loads(dependency.read_text())
            if state["status"] == "failed": raise RuntimeError("Existing matrix failed; preserve pilot without driving")
            if state["status"] == "complete": break
            if time.time() - state["updated_epoch"] > 600: raise RuntimeError("Dependency heartbeat stale")
            status("waiting_for_frozen_matrix", dependency=str(dependency))
            time.sleep(10)
        for actor in spec["actors"]:
            checked = execute(actor, "direct_parity", None, 2)
            reference = Path(actor["direct_reference"])
            replay(reference, checked, 2)
            completed.append({"seed": actor["seed"], "condition": "direct", "penalty_scale": None,
                "evaluation": str(reference), "reused": True, "metrics_sha256": sha256(reference / "metrics.csv"),
                "config_sha256": sha256(reference / "config.json")})
            for label, scale in (("scale1", 1.), ("scale025", .25), ("scale0", 0.)):
                destination = execute(actor, label, scale, 20)
                if scale == 1.: replay(Path(actor["search7_reference"]), destination, 20)
                completed.append({"seed": actor["seed"], "condition": label, "penalty_scale": scale,
                    "evaluation": str(destination), "reused": False,
                    "metrics_sha256": sha256(destination / "metrics.csv"), "config_sha256": sha256(destination / "config.json")})
                write_json(root / "results.json", {"conditions": completed, "new_costs": costs})
        status("evaluations_complete")
        subprocess.run([sys.executable, str(root / "reporter.py"), "--campaign", str(root),
            "--output", str(workspace / "report")], cwd=workspace, env=env, check=True)
        status("complete", report=str(workspace / "report"))
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        status("failed", error=str(error))
        raise


if __name__ == "__main__": main()
