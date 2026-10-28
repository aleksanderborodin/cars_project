"""Small train-only integration/timing check; not a racing performance comparison."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from cars.env.protocol import DrivingEnv, ProtocolConfig
from cars.planning.policy_search import PolicyPhysicsSearch, PolicySearchConfig
from cars.research.artifacts import begin_run, read_config
from cars.research.recurrent import RecurrentDriver
from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runtime-context", default="unspecified; do not interpret as isolated timing")
    args = parser.parse_args()
    config = read_config(args.model)
    if config["settings"].get("partition") != "train":
        raise ValueError("Use a train-fitted checkpoint")
    protocol = ProtocolConfig(**config["protocol"])
    settings = {"partition": "train", "collection_seed": 99002, "checkpoint": str(args.model),
        "checkpoint_sha256": sha256(args.model / "model.zip"), "snapshot_decisions": [0, 40, 80],
        "candidates": [7, 21], "horizon": 20, "planning_seed": 0, "native_frame_cap": 320,
        "internal_physics_frame_cap": 6720, "concurrency": args.runtime_context,
        "interpretation": "reset-state snapshot timing/integration only; no race or ranking claim"}
    root = begin_run("verification", args.run_id, protocol, settings)
    env, actor = DrivingEnv(protocol, "train", seed=99002), RecurrentDriver(args.model)
    snapshots, frames = [], 0
    try:
        obs, _ = env.reset()
        for i in range(81):
            if i in settings["snapshot_decisions"]:
                snapshots.append({key: value.copy() for key, value in obs.items()})
            if i == 80:
                break
            obs, _, done, truncated, info = env.step(actor.act(obs))
            frames += info["native_frames"]
            if done or truncated:
                raise RuntimeError("Probe episode ended before declared snapshots; preserve attempt")
        write_json(root / "collection.json", {"native_frames": frames, "track_seeds": env.track_seeds})
    finally:
        env.close()
    np.savez_compressed(root / "observations.npz", **{
        key: np.asarray([obs[key] for obs in snapshots]) for key in snapshots[0]})
    rows = []
    for count in settings["candidates"]:
        planner = PolicyPhysicsSearch(args.model, protocol, PolicySearchConfig(candidates=count))
        try:
            for i, observation in enumerate(snapshots):
                planner.reset()
                started = time.perf_counter()
                action = planner.act(observation)
                elapsed = (time.perf_counter() - started) * 1000
                assert np.isfinite(action).all() and np.all(action >= [-1, 0, 0]) and np.all(action <= 1)
                rows.append({"candidates": count, "snapshot": settings["snapshot_decisions"][i],
                    "policy_ms": elapsed, "action": action.tolist(), **planner.diagnostics})
        finally:
            planner.reset()
    assert frames <= settings["native_frame_cap"]
    assert sum(row["internal_physics_frames"] for row in rows) <= settings["internal_physics_frame_cap"]
    write_json(root / "result.json", {"rows": rows, "native_frames": frames,
        "internal_physics_frames": sum(row["internal_physics_frames"] for row in rows),
        "note": settings["interpretation"]})
    print(json.dumps({"run": str(root), "timings_ms": {str(n): [r["policy_ms"] for r in rows if r["candidates"] == n]
                      for n in settings["candidates"]}}, indent=2))


if __name__ == "__main__":
    main()
