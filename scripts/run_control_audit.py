"""Execute an explicitly listed development campaign; never enumerate run storage."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import json
from pathlib import Path
import shutil
import time

from cars.env.protocol import ProtocolConfig
from cars.models.dynamics import Dynamics
from cars.planning.local import FixedBehavior
from cars.planning.search import BehaviorSearch, SearchConfig
from cars.research.artifacts import read_config
from cars.research.evaluation import evaluate_driver
from cars.training import sha256, write_json


def execute(job: dict, manifest_path: str, logs: str) -> str:
    manifest = json.loads(Path(manifest_path).read_text())
    protocol = ProtocolConfig(**manifest["protocol"])
    if job["kind"] == "corrected_search":
        model = Path(job["model"])
        model_config = read_config(model)
        if model_config["protocol"] != manifest["protocol"] or sha256(model / "model.npz") != job["model_sha256"]:
            raise ValueError("Model/protocol changed since campaign declaration")
        driver = BehaviorSearch(protocol, Dynamics.load(model / "model.npz"),
            SearchConfig(horizon=manifest["horizon_decisions"], speeds=tuple(manifest["speeds"]),
                         steering_gains=tuple(manifest["gains"])), model_artifact=model)
    else:
        driver = FixedBehavior(protocol, job["speed"], job["gain"])
    with (Path(logs) / f'{job["run_id"]}.log').open("x") as file, contextlib.redirect_stdout(file):
        root = evaluate_driver(driver, job["run_id"], episodes=manifest["tracks"], protocol=protocol)
    shutil.copy2(manifest_path, root / "campaign.json")
    return str(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["partition"] != "development" or manifest["user_only_oracle_access"]:
        raise ValueError("Only ordinary development evaluation is supported")
    root = Path("runs/campaigns") / manifest["id"]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.manifest, root / "manifest.json")
    shutil.copy2(__file__, root / "runner.py")
    started = time.perf_counter()
    completed = []
    write_json(root / "status.json", {"status": "running", "completed": completed})
    try:
        with ProcessPoolExecutor(max_workers=manifest["workers"]) as pool:
            # All source/config changes finish before the pool starts.
            futures = [pool.submit(execute, job, str(args.manifest), str(root)) for job in manifest["jobs"]]
            for job, future in zip(manifest["jobs"], futures, strict=True):
                result = future.result()
                completed.append(result)
                write_json(root / "status.json", {"status": "running", "completed": completed,
                    "wall_time_s": time.perf_counter() - started})
                print(f'{len(completed)}/{len(futures)} completed: {job["run_id"]}', flush=True)
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "completed": completed,
            "error": str(error), "wall_time_s": time.perf_counter() - started})
        raise
    write_json(root / "status.json", {"status": "complete", "completed": completed,
        "wall_time_s": time.perf_counter() - started, "manifest_sha256": sha256(args.manifest)})


if __name__ == "__main__":
    main()
