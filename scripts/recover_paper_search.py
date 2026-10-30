"""Recover the frozen development matrix on another disk; preserve prior artifacts."""
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
from cars.research.frozen_code import source_hashes
from cars.training import sha256, write_json


def replay(reference: Path, current: Path, episodes: int) -> None:
    for i in range(episodes):
        with np.load(reference / f"episodes/{i:04d}.npz") as a, np.load(current / f"episodes/{i:04d}.npz") as b:
            for name in a.files:
                if name.startswith("observation_") or name in (
                    "action", "reward", "frames", "terminated", "truncated", "evaluation_offroad_frame_indices"):
                    np.testing.assert_array_equal(a[name], b[name], err_msg=f"Recovered prefix {i}: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()
    root = args.campaign.resolve() if args.campaign else Path(__file__).resolve().parent
    spec = json.loads((root / "config.json").read_text())
    rec = spec["recovery"]
    origin, workspace = Path(rec["original_repository"]), Path(rec["workspace"])
    prior = json.loads((root / "prior_results.json").read_text())
    completed, costs = list(prior["conditions"]), list(prior["new_costs"])
    if (len(completed) != 18 or len({(r["seed"], r["condition"]) for r in completed}) != 18
            or spec["partition_sha256"] != Partitions().fingerprint or spec["tracks"] != 200
            or spec["held_out_evaluation"] or spec["user_only_oracle_access"]):
        raise ValueError("Wrong fixed development recovery declaration")
    original_config = Path(rec["original_campaign"]) / "config.json"
    if (sha256(original_config) != rec["original_config_sha256"]
            or {k: v for k, v in spec.items() if k != "recovery"} != json.loads(original_config.read_text())):
        raise ValueError("Frozen scientific declaration changed")
    if sha256(root / "runner.py") != rec["runner_sha256"] or sha256(Path(__file__)) != rec["runner_sha256"]:
        raise ValueError("Recovery controller changed")
    if source_hashes(root / "source") != spec["source_sha256"]:
        raise ValueError("Frozen evaluator/source changed")
    if sha256(root / "reporter.py") != spec["reporter_sha256"]:
        raise ValueError("Frozen analysis changed")
    for row in completed:
        path = Path(row["evaluation"])
        if sha256(path / "metrics.csv") != row["metrics_sha256"] or sha256(path / "config.json") != row["config_sha256"]:
            raise ValueError("Previously completed result changed")
        table = pd.read_csv(path / "metrics.csv")
        if table.track_seed.tolist() != Partitions().seeds("development", 200):
            raise ValueError("Previously completed panel mismatch")
    if shutil.disk_usage(workspace).free < rec["storage_floor_bytes"]:
        raise RuntimeError("Recovery disk below declared free-space floor")
    if args.check:
        print("Validated: 18 complete panels, same frozen methods/source/analysis, recovery caps and storage")
        return
    costs.append({"native_frames": rec["wasted_native_frames_upper_bound"],
        "internal_physics_frames": rec["wasted_internal_physics_frames_upper_bound"],
        "cost_scope": "conservative upper bound for failed original attempt; not additional scored episodes",
        "failed_run": rec["partial_failed_run"], "native_frames_exact": False})
    env = {**os.environ, "PYTHONPATH": str(root / "source/src"), "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TMPDIR": str(workspace / "tmp"),
        "XDG_CACHE_HOME": str(workspace / "cache"), "MPLCONFIGDIR": str(workspace / "cache/matplotlib")}
    started, process = time.perf_counter(), None

    def status(state: str, **extra) -> None:
        target = root / "status.tmp.json"
        write_json(target, {"status": state, "queue_pid": os.getpid(), "updated_epoch": time.time(),
            "completed": completed, "recovery_wall_seconds": time.perf_counter() - started,
            "native_frames_including_failed_attempt_bound": sum(c["native_frames"] for c in costs),
            "internal_frames_including_failed_attempt_bound": sum(c["internal_physics_frames"] for c in costs),
            "free_bytes": shutil.disk_usage(workspace).free, **extra})
        target.replace(root / "status.json")

    try:
        for seed in (0, 1, 2):
            selection_file = origin / spec["selection_artifacts"][str(seed)]
            selection = json.loads(selection_file.read_text())["selected"]
            checkpoint = Path(selection["checkpoint"])
            if not checkpoint.is_absolute(): checkpoint = origin / checkpoint
            config = read_config(checkpoint)
            if (config["protocol_id"] != spec["protocol_id"] or config["versions"] != spec["versions"]
                    or config["settings"]["training"]["seed"] != seed):
                raise ValueError("Selected actor/protocol/version mismatch")
            candidates = json.loads((selection_file.parent / "checkpoint_results.json").read_text())
            best = max(candidates, key=lambda r: (r["mean_return"], r["completion"], -r["training_native_frames"]))
            best_path = Path(best["checkpoint"])
            if not best_path.is_absolute(): best_path = origin / best_path
            if best_path.resolve() != checkpoint.resolve():
                raise ValueError("Standalone selection changed")
            model_hash = sha256(checkpoint / "model.zip")
            for condition in spec["conditions"]:
                if any(r["seed"] == seed and r["condition"] == condition["name"] for r in completed):
                    continue
                if sha256(checkpoint / "model.zip") != model_hash or source_hashes(root / "source") != spec["source_sha256"]:
                    raise ValueError("Frozen inputs changed")
                run_id = f"paper_search_20261030_s{seed}_{condition['name']}_recovery1"
                destination = workspace / "runs" / condition["directory"] / run_id
                if destination.exists(): raise FileExistsError(destination)
                command = [sys.executable, "-m", "cars.cli", "research", "evaluate", "--method", condition["method"],
                    "--model", str(checkpoint), "--episodes", "200", "--run-id", run_id]
                if condition["method"] in ("policy-search", "policy-random"):
                    command += ["--candidates", str(condition["candidates"]), "--planning-seed", str(condition["planning_seed"]),
                        "--horizon", "20", "--steer-sigma", "0.1", "--gas-sigma", "0.15", "--brake-sigma", "0.15",
                        "--observation-renderer", spec["observation_renderer"]]
                if condition["method"] == "policy-bias": command += ["--action-bias", *map(str, spec["fixed_bias"])]
                write_json(root / (run_id + "_command.json"), command)
                with (root / (run_id + ".log")).open("x") as log:
                    process = subprocess.Popen(command, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=True)
                    while process.poll() is None:
                        if shutil.disk_usage(workspace).free < rec["storage_floor_bytes"]:
                            raise RuntimeError("Recovery disk reached free-space floor during evaluation")
                        status("evaluating", active=run_id, child_pid=process.pid)
                        time.sleep(5)
                    if process.returncode: raise RuntimeError(f"Recovery child failed: {run_id}")
                costs.append(json.loads((destination / "cost.json").read_text()))
                if (sum(c["native_frames"] for c in costs) > spec["new_native_frame_cap"] + rec["added_native_frame_cap"]
                        or sum(c["internal_physics_frames"] for c in costs) > spec["new_internal_physics_frame_cap"] + rec["added_internal_physics_frame_cap"]):
                    raise RuntimeError("Declared recovery budget exceeded")
                if seed == 1 and condition["name"] == "search7":
                    replay(Path(rec["partial_failed_run"]), destination, rec["partial_complete_episodes"])
                    write_json(root / "recovered_prefix_check.json", {"episodes": 58, "exact": True,
                        "excluded_from_parity": "latency and infrastructure paths; method outputs/actions/rewards/observations identical"})
                if condition["name"] == "direct":
                    reference = Path(selection["evaluation"])
                    replay(reference if reference.is_absolute() else origin / reference, destination, 50)
                evaluated = read_config(destination)
                driver = evaluated["settings"]["driver"]
                if (evaluated["versions"] != spec["versions"] or evaluated["protocol_id"] != spec["protocol_id"]
                        or driver.get("actor", driver).get("checkpoint_sha256") != model_hash):
                    raise ValueError("Recovered evaluation identity mismatch")
                metrics = pd.read_csv(destination / "metrics.csv")
                if metrics.track_seed.tolist() != Partitions().seeds("development", 200):
                    raise ValueError("Wrong development panel")
                completed.append({"seed": seed, "condition": condition["name"], "evaluation": str(destination), "reused": False,
                    "mean_return": float(metrics['return'].mean()), "completion": float(metrics.success.mean()),
                    "model_sha256": model_hash, "metrics_sha256": sha256(destination / "metrics.csv"),
                    "config_sha256": sha256(destination / "config.json")})
                write_json(root / "results.json", {"conditions": completed, "new_costs": costs})
        status("evaluations_complete")
        subprocess.run([sys.executable, str(root / "reporter.py"), "--campaign", str(root),
            "--output", rec["report_directory"]], cwd=workspace, env=env, check=True)
        status("complete", report=rec["report_directory"])
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        status("failed", error=str(error))
        raise


if __name__ == "__main__": main()
