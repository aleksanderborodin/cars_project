"""Serial native CPU/CUDA comparison with exact historical and same-device replays."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from cars.training import sha256, write_json


def replay(first: Path, second: Path, episodes: int) -> dict:
    pd.testing.assert_frame_equal(pd.read_csv(first / "episodes.csv").head(episodes),
                                  pd.read_csv(second / "episodes.csv"), check_exact=True)
    for index in range(episodes):
        with np.load(first / f"episodes/{index:04d}.npz") as a, np.load(second / f"episodes/{index:04d}.npz") as b:
            assert set(a.files) == set(b.files)
            for name in a.files:
                if name != "policy_seconds":
                    np.testing.assert_array_equal(a[name], b[name])
    return {"passed": True, "episodes": episodes, "all_trace_fields_exact_except_timing": True,
            "original": str(first), "replay": str(second)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    declaration = parser.parse_args().manifest
    spec = json.loads(declaration.read_text())
    root = Path("runs/campaigns") / spec["id"]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(declaration, root / "config.json")
    shutil.copy2(__file__, root / "runner.py")
    write_json(root / "provenance.json", {"manifest_sha256": sha256(declaration),
        "runner_sha256": sha256(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "SDL_VIDEODRIVER": "dummy", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    results, costs = [], []
    started = time.perf_counter()
    try:
        for case in spec["conditions"]:
            checkpoint = Path(case["model"])
            for name, expected in case["checkpoint_sha256"].items():
                assert sha256(checkpoint / name) == expected
            replays = []
            for kind, device, count in (("cpu_replay", "cpu", spec["replay_episodes"]),
                                        ("cuda", "cuda", spec["comparison_episodes"]),
                                        ("cuda_replay", "cuda", spec["replay_episodes"])):
                run_id = case[kind + "_run_id"]
                write_json(root / "status.json", {"status": "running", "active": run_id,
                                                  "completed": results, "pid": os.getpid()})
                command = [sys.executable, "-m", "cars.cli", "research", "evaluate-native", "--model",
                           str(checkpoint), "--episodes", str(count), "--device", device, "--run-id", run_id]
                with (root / (run_id + ".log")).open("x") as log:
                    process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                if process.returncode:
                    raise RuntimeError(f"Evaluation failed: {run_id}; inspect saved public log")
                run = Path("runs/native_evaluation") / run_id
                costs.append({"run_id": run_id, **json.loads((run / "cost.json").read_text())})
                if kind.endswith("replay"):
                    original = (Path(case["original_cpu_evaluation"]) if kind == "cpu_replay" else
                                Path("runs/native_evaluation") / case["cuda_run_id"])
                    replays.append(replay(original, run, count))
            cpu = pd.read_csv(Path(case["original_cpu_evaluation"]) / "episodes.csv").head(spec["comparison_episodes"])
            gpu = pd.read_csv(Path("runs/native_evaluation") / case["cuda_run_id"] / "episodes.csv")
            np.testing.assert_array_equal(cpu.track_seed, gpu.track_seed)
            row = {"checkpoint": case["checkpoint"], "cpu_mean_return": float(cpu["return"].mean()),
                   "cuda_mean_return": float(gpu["return"].mean()), "cpu_completions": int(cpu.success.sum()),
                   "cuda_completions": int(gpu.success.sum()), "episodes": len(cpu),
                   "paired_return_change": float((gpu["return"] - cpu["return"]).mean()), "replays": replays}
            results.append(row)
            write_json(root / "results.json", {"comparisons": results, "costs": costs})
            print(json.dumps(row), flush=True)
        total = sum(cost["native_frames"] for cost in costs)
        assert total <= spec["native_frame_cap"]
        write_json(root / "status.json", {"status": "complete", "completed": results,
            "native_frames": total, "reset_physics_ticks": sum(c["reset_physics_ticks"] for c in costs),
            "wall_seconds": time.perf_counter() - started})
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "error": str(error),
            "completed": results, "completed_run_costs": costs, "wall_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
