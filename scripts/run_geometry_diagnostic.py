"""Execute the declared full-map diagnostic after checking local replay compatibility."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from cars.env.protocol import DrivingEnv, ProtocolConfig
from cars.planning.physics_search import PhysicsSearch
from cars.planning.search import SearchConfig
from cars.research.artifacts import begin_run, read_config
from cars.training import write_json


def main() -> None:
    manifest = Path("configs/campaigns/geometry-diagnostic-2026-10-28.json")
    declaration = json.loads(manifest.read_text())
    if declaration["partition"] != "development" or declaration["user_only_oracle_access"]:
        raise ValueError("Development-only public diagnostic")
    root = Path("runs/campaigns") / declaration["id"]
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest, root / "config.json")
    shutil.copy2(__file__, root / "runner.py")
    started = time.perf_counter()
    source = Path(declaration["conditions"][0]["local_reference"])
    config = read_config(source)
    protocol = ProtocolConfig(**config["protocol"])
    search = {f.name: config["settings"]["driver"][f.name] for f in fields(SearchConfig)}
    check = begin_run("verification", "geometry_local_compatibility_20261028", protocol,
                      {"partition": "development", "source": str(source), "decisions": 20})
    driver, env = PhysicsSearch(protocol, SearchConfig(**search)), DrivingEnv(protocol, "development")
    frames = work = 0
    try:
        obs, _ = env.reset()
        with np.load(source / "episodes/0000.npz") as data:
            for i in range(20):
                for key, value in obs.items():
                    np.testing.assert_array_equal(value, data["observation_" + key][i])
                action = driver.act(obs)
                np.testing.assert_array_equal(action, data["action"][i])
                obs, reward, terminated, truncated, info = env.step(action)
                assert reward == data["reward"][i]
                assert terminated == data["terminated"][i] and truncated == data["truncated"][i]
                frames += info["native_frames"]
                work += driver.diagnostics["internal_physics_frames"]
            for key, value in obs.items():
                np.testing.assert_array_equal(value, data["observation_" + key][20])
        write_json(check / "result.json", {"identical": True, "native_frames": frames, "internal_physics_frames": work})
    finally:
        env.close()
        driver.reset()
    print("Local replay matched exactly; starting the two declared global conditions", flush=True)
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    commands = []
    for condition in declaration["conditions"]:
        command = [sys.executable, "-m", "cars.cli", "research", "evaluate", "--method", "physics-global",
                   "--speed-grid", *map(str, declaration["speeds"]), "--gain-grid", *map(str, condition["gains"]),
                   "--horizon", str(declaration["horizon"]), "--episodes", str(declaration["tracks"]),
                   "--run-id", condition["run_id"]]
        commands.append((condition["run_id"], command))
    write_json(root / "commands.json", commands)
    completed = []

    def run(entry: tuple) -> str:
        run_id, command = entry
        with (root / (run_id + ".log")).open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
        shutil.copy2(manifest, Path("runs/global_physics_search") / run_id / "campaign.json")
        return run_id

    write_json(root / "status.json", {"status": "running", "completed": completed})
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(run, command) for command in commands]):
                completed.append(future.result())
                write_json(root / "status.json", {"status": "running", "completed": completed})
                print(f"Completed {completed[-1]}", flush=True)
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "completed": completed, "error": str(error)})
        raise
    costs = [json.loads((Path("runs/global_physics_search") / item / "cost.json").read_text()) for item in completed]
    native, internal = sum(c["native_frames"] for c in costs), sum(c["internal_physics_frames"] for c in costs)
    assert native <= declaration["native_evaluation_frame_cap"] and internal <= declaration["internal_physics_frame_cap"]
    write_json(root / "status.json", {"status": "complete", "completed": completed,
        "native_frames": native, "internal_physics_frames": internal, "wall_time_s": time.perf_counter() - started})


if __name__ == "__main__":
    main()
