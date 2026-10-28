"""Train a declared block and evaluate each ready post-update checkpoint once."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from cars.training import sha256, write_json


def main() -> None:
    manifest = Path("configs/campaigns/driver-first-block-2026-10-28.json")
    spec = json.loads(manifest.read_text())
    config = Path(spec["training_config"])
    if sha256(config) != spec["training_config_sha256"] or spec["user_only_oracle_access"] or spec["held_out_evaluation"]:
        raise ValueError("Declaration mismatch")
    root = Path("runs/campaigns") / spec["id"]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest, root / "config.json")
    shutil.copy2(config, root / "training_config.json")
    shutil.copy2(__file__, root / "runner.py")
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    command = [sys.executable, "-m", "cars.cli", "research", "train-driver", "--config", str(config),
               "--stop-at", str(spec["stop_at_native_frames"]), "--run-id", spec["training_run_id"]]
    write_json(root / "training_command.json", command)
    training_root = Path("runs/driver_training") / spec["training_run_id"]
    started = time.perf_counter()
    jobs, rows = {}, []

    def evaluate(path: Path) -> dict:
        state = json.loads((path / "training_state.json").read_text())
        run_id = f"driver_s0_dev_{state['native_frames']:09d}"
        arguments = [sys.executable, "-m", "cars.cli", "research", "evaluate", "--method", "recurrent",
                     "--model", str(path), "--episodes", str(spec["development_tracks_per_checkpoint"]), "--run-id", run_id]
        with (root / (run_id + ".log")).open("x") as log:
            subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
        evaluation = Path("runs/recurrent_ppo") / run_id
        shutil.copy2(manifest, evaluation / "campaign.json")
        return {"checkpoint": str(path), "training_native_frames": state["native_frames"], "evaluation": str(evaluation),
                **json.loads((evaluation / "summary.json").read_text()),
                "evaluation_native_frames": json.loads((evaluation / "cost.json").read_text())["native_frames"]}

    with (root / "training.log").open("x") as log, ThreadPoolExecutor(max_workers=1) as pool:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        write_json(root / "status.json", {"status": "running", "training_pid": process.pid, "evaluations": []})
        final_scan = False
        try:
            while True:
                for path in sorted((training_root / "checkpoints").glob("frames_*")):
                    if path in jobs or not (path / "training_state.json").exists():
                        continue
                    try:
                        json.loads((path / "training_state.json").read_text())
                    except json.JSONDecodeError:
                        continue  # final checkpoint marker is still being written
                    jobs[path] = pool.submit(evaluate, path)
                for future in jobs.values():
                    if future.done() and not getattr(future, "recorded", False):
                        rows.append(future.result())
                        future.recorded = True
                        write_json(root / "checkpoint_results.json", rows)
                        print(f"Checkpoint evaluated: {rows[-1]['training_native_frames']} -> {rows[-1]['mean_return']:.1f}, completion={rows[-1]['completion']:.2f}", flush=True)
                code = process.poll()
                write_json(root / "status.json", {"status": "running", "training_pid": process.pid,
                    "training_exit_code": code, "evaluations": len(rows), "queued": len(jobs),
                    "wall_time_s": time.perf_counter() - started})
                if code is not None and code != 0:
                    raise RuntimeError(f"Training failed with exit code {code}; public training.log retained")
                if code == 0 and len(jobs) < len(spec["scheduled_checkpoints"]):
                    if final_scan:
                        raise RuntimeError("Training completed without all scheduled checkpoints")
                    final_scan = True
                    continue
                if code == 0 and all(future.done() for future in jobs.values()):
                    break
                time.sleep(5)
            if len(rows) != len(spec["scheduled_checkpoints"]):
                raise RuntimeError("Missing scheduled checkpoint evaluations")
            assert sum(row["evaluation_native_frames"] for row in rows) <= spec["evaluation_native_frame_cap"]
            cost = json.loads((training_root / "cost.json").read_text())
            assert cost["native_frames"] <= spec["training_frame_cap_including_maximum_rollout_overshoot"]
            selected = sorted(rows, key=lambda row: (-row["mean_return"], -row["completion"], row["training_native_frames"]))[0]
            write_json(root / "selection.json", {"selected": selected, "rule": spec["checkpoint_selection"],
                "interpretation": "standalone development selection; not evidence of convergence or test performance"})
            shutil.copy2(manifest, training_root / "campaign.json")
            write_json(root / "status.json", {"status": "complete", "evaluations": len(rows),
                "training_native_frames": cost["native_frames"],
                "evaluation_native_frames": sum(row["evaluation_native_frames"] for row in rows),
                "wall_time_s": time.perf_counter() - started})
        except Exception as error:
            if process.poll() is None:
                process.terminate()
                process.wait()
            write_json(root / "status.json", {"status": "failed", "error": str(error), "evaluations": len(rows)})
            raise


if __name__ == "__main__":
    main()
