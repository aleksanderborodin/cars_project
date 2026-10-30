"""Frozen, dependency-gated multi-seed programmed-search development campaign."""
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
                if name.startswith("observation_") or name in (
                    "action", "reward", "frames", "terminated", "truncated", "evaluation_offroad_frame_indices"):
                    np.testing.assert_array_equal(a[name], b[name], err_msg=f"Episode {i}: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    if (spec["partition_sha256"] != Partitions().fingerprint or spec["partition"] != "development"
            or spec["tracks"] != 200 or spec["held_out_evaluation"] or spec["user_only_oracle_access"]):
        raise ValueError("Only the declared 200-track development protocol is allowed")
    if source_hashes(Path.cwd()) != spec["source_sha256"] or sha256(Path(__file__)) != spec["runner_sha256"]:
        raise ValueError("Frozen source/runner changed before preparation")
    for name, expected in spec["required_sha256"].items():
        if sha256(name) != expected:
            raise ValueError(f"Declared input changed: {name}")
    if sha256(Path(spec["reporter"])) != spec["reporter_sha256"]:
        raise ValueError("Frozen report analysis changed")
    if [c["name"] for c in spec["conditions"]] != ["direct", "fixed", "random7_s0", "random7_s1", "random7_s2",
                                                  "random21_s0", "random21_s1", "random21_s2", "search7", "search21"]:
        raise ValueError("Unexpected condition matrix")
    if args.check:
        print("Valid: fixed development matrix, immutable source, standalone selections and historical reuse")
        return
    root = Path("runs/queues") / spec["id"]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.manifest, root / "config.json")
    shutil.copy2(__file__, root / "runner.py")
    shutil.copy2(spec["reporter"], root / "reporter.py")
    env = freeze_sources(Path.cwd(), root / "source", spec["source_sha256"])
    env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8")
    completed, costs = [], []
    process = None
    started = time.perf_counter()

    def status(state: str, **extra) -> None:
        temporary = root / "status.tmp.json"
        write_json(temporary, {"status": state, "queue_pid": os.getpid(), "updated_epoch": time.time(),
            "completed": completed, "new_native_frames": sum(c["native_frames"] for c in costs),
            "new_internal_physics_frames": sum(c["internal_physics_frames"] for c in costs),
            "wall_seconds": time.perf_counter() - started, **extra})
        temporary.replace(root / "status.json")

    def execute(seed: int, checkpoint: Path, condition: dict, episodes: int, suffix: str) -> Path:
        nonlocal process
        if source_hashes(root / "source") != spec["source_sha256"]:
            raise ValueError("Frozen evaluation source changed")
        if shutil.disk_usage(root).free < spec["minimum_free_disk_bytes"]:
            raise RuntimeError("Storage floor reached; preserve artifacts")
        run_id = f"paper_search_20261030_s{seed}_{condition['name']}{suffix}"
        destination = Path("runs") / condition["directory"] / run_id
        if destination.exists():
            raise FileExistsError(destination)
        command = [sys.executable, "-m", "cars.cli", "research", "evaluate", "--method", condition["method"],
                   "--model", str(checkpoint), "--episodes", str(episodes), "--run-id", run_id]
        if condition["method"] in ("policy-search", "policy-random"):
            command += ["--candidates", str(condition["candidates"]), "--planning-seed", str(condition["planning_seed"]),
                        "--horizon", "20", "--steer-sigma", "0.1", "--gas-sigma", "0.15", "--brake-sigma", "0.15",
                        "--observation-renderer", "reference"]
        if condition["method"] == "policy-bias":
            command += ["--action-bias", *map(str, spec["fixed_bias"])]
        write_json(root / (run_id + "_command.json"), command)
        with (root / (run_id + ".log")).open("x") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                status("evaluating", active=run_id, child_pid=process.pid)
                time.sleep(5)
            if process.returncode:
                raise RuntimeError(f"{run_id} failed; inspect saved public log")
        costs.append(json.loads((destination / "cost.json").read_text()))
        if (sum(c["native_frames"] for c in costs) > spec["new_native_frame_cap"] or
                sum(c["internal_physics_frames"] for c in costs) > spec["new_internal_physics_frame_cap"]):
            raise RuntimeError("Declared campaign cost cap exceeded")
        return destination

    try:
        dependency = Path(spec["dependency"])
        while True:
            try:
                state = json.loads((dependency / "status.json").read_text())
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            if state["status"] == "failed":
                raise RuntimeError("Training dependency failed; evaluation queue stops")
            if state["status"] == "complete":
                break
            if time.time() - (dependency / "status.json").stat().st_mtime > 600:
                raise RuntimeError("Training dependency heartbeat stale")
            status("waiting_for_training", dependency=str(dependency), active_seed=state.get("active_seed"))
            time.sleep(10)
        for seed in (0, 1, 2):
            selection_path = Path(spec["selection_artifacts"][str(seed)])
            selection = json.loads(selection_path.read_text())
            selected = selection["selected"]
            checkpoint = Path(selected["checkpoint"])
            config = read_config(checkpoint)
            if config["protocol_id"] != spec["protocol_id"] or config["settings"]["training"]["seed"] != seed:
                raise ValueError("Wrong selected local actor")
            if config["versions"] != spec["versions"]:
                raise ValueError("Checkpoint software versions differ from frozen comparison")
            # Check the standalone rule from all four scheduled evaluations; never select by search gain.
            candidates = json.loads((selection_path.parent / "checkpoint_results.json").read_text())
            best = max(candidates, key=lambda c: (c["mean_return"], c["completion"], -c["training_native_frames"]))
            if Path(best["checkpoint"]).resolve() != checkpoint.resolve():
                raise ValueError("Standalone checkpoint-selection rule mismatch")
            model_hash = sha256(checkpoint / "model.zip")
            write_json(root / f"seed{seed}_frozen_checkpoint.json", {"checkpoint": str(checkpoint),
                "model_sha256": model_hash, "selection_sha256": sha256(selection_path), "standalone_selection": selected})
            for condition in spec["conditions"]:
                if sha256(checkpoint / "model.zip") != model_hash:
                    raise ValueError("Frozen policy changed")
                key = condition["name"]
                reused = spec["seed0_reuse"].get(key) if seed == 0 else None
                if reused:
                    destination = Path(reused)
                    preflight = execute(seed, checkpoint, condition, 2, "_parity")
                    replay(destination, preflight, 2)
                else:
                    destination = execute(seed, checkpoint, condition, spec["tracks"], "")
                    if key == "direct":
                        replay(Path(selected["evaluation"]), destination, 50)
                    if seed == 0 and key.startswith("random7_"):
                        replay(Path(spec["seed0_random7_reference"][key]), destination, 20)
                evaluated = read_config(destination)
                if evaluated["protocol_id"] != spec["protocol_id"]:
                    raise ValueError("Evaluation protocol mismatch")
                if evaluated["versions"] != spec["versions"]:
                    raise ValueError("Evaluation software versions changed")
                actor = evaluated["settings"]["driver"]
                if actor.get("actor", actor).get("checkpoint_sha256") != model_hash:
                    raise ValueError("Evaluated actor differs from standalone-selected checkpoint")
                metrics = pd.read_csv(destination / "metrics.csv")
                if metrics.track_seed.tolist() != Partitions().seeds("development", 200):
                    raise ValueError("Wrong full development panel")
                record = {"seed": seed, "condition": key, "evaluation": str(destination), "reused": bool(reused),
                    "mean_return": float(metrics["return"].mean()), "completion": float(metrics.success.mean()),
                    "model_sha256": model_hash, "metrics_sha256": sha256(destination / "metrics.csv"),
                    "config_sha256": sha256(destination / "config.json")}
                completed.append(record)
                write_json(root / "results.json", {"conditions": completed, "new_costs": costs})
                print(json.dumps(record), flush=True)
        status("evaluations_complete")
        subprocess.run([sys.executable, str(root / "reporter.py"), "--campaign", str(root),
                        "--output", spec["report_directory"]], env=env, check=True)
        status("complete", report=spec["report_directory"])
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
