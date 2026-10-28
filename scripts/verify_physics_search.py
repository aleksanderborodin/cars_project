"""Deterministic replay and isolated timing from public development observations."""
import json
from pathlib import Path
import time

import numpy as np

from cars.env.protocol import DrivingEnv, ProtocolConfig
from cars.planning.physics_search import PhysicsSearch
from cars.planning.search import SearchConfig
from cars.research.artifacts import begin_run
from cars.training import sha256, write_json


def main() -> None:
    source = Path("runs/physics_search/physics_budget_20261028_gain12")
    protocol = ProtocolConfig()
    speeds = (30., 40., 50., 60., 70., 80., 90.)
    root = begin_run("verification", "physics_search_20261028", protocol, {
        "partition": "development", "source_run": str(source), "replay_episode": 0,
        "timing_episodes": [0, 1, 2, 3], "positions_per_episode": 5, "repetitions": 3,
        "timing_history": "observer reset for each recorded observation; timing only, not accuracy",
        "timing_scope": "policy plus world construction; sensor, environment and disk excluded",
        "additional_training_frames": 0})
    driver = PhysicsSearch(protocol, SearchConfig(speeds=speeds, steering_gains=(1.2,)))
    environment = DrivingEnv(protocol, "development")
    started = time.perf_counter()
    try:
        with np.load(source / "episodes/0000.npz") as data:
            observation, _ = environment.reset()
            driver.reset()
            for i in range(len(data["action"])):
                for key, value in observation.items():
                    np.testing.assert_array_equal(value, data["observation_" + key][i])
                action = driver.act(observation)
                np.testing.assert_array_equal(action, data["action"][i])
                observation, reward, done, truncated, info = environment.step(action)
                assert reward == data["reward"][i] and info["native_frames"] == data["frames"][i]
                assert done == data["terminated"][i] and truncated == data["truncated"][i]
            for key, value in observation.items():
                np.testing.assert_array_equal(value, data["observation_" + key][-1])
            replay = {"identical_actions_rewards_observations_and_end_flags": True,
                "native_frames": int(data["frames"].sum()), "decisions": len(data["action"]),
                "internal_physics_frames": len(data["action"]) * 7 * 20 * 4,
                "wall_time_s": time.perf_counter() - started,
                "trajectory_sha256": sha256(source / "episodes/0000.npz")}
    finally:
        environment.close()
        driver.reset()
    cases = []
    for episode in range(4):
        with np.load(source / "episodes" / f"{episode:04d}.npz") as data:
            for index in np.linspace(0, len(data["action"]) - 1, 5, dtype=int):
                observation = {key.removeprefix("observation_"): data[key][index].copy()
                               for key in data.files if key.startswith("observation_")}
                cases.append((episode, int(index), observation))
    timings = []
    drivers = {7: PhysicsSearch(protocol, SearchConfig(speeds=speeds, steering_gains=(1.2,))),
               21: PhysicsSearch(protocol, SearchConfig(speeds=speeds))}
    for repetition in range(3):
        # Alternate method order to avoid assigning every first measurement to one method.
        order = (7, 21) if repetition % 2 == 0 else (21, 7)
        for episode, index, observation in cases:
            for candidates in order:
                policy = drivers[candidates]
                policy.reset()
                start = time.perf_counter()
                policy.act(observation)
                elapsed = (time.perf_counter() - start) * 1000
                timings.append({"repetition": repetition, "episode": episode, "decision": index,
                                "candidates": candidates, "policy_ms": elapsed})
    for policy in drivers.values():
        policy.reset()
    summary = {}
    for candidates in (7, 21):
        samples = [row["policy_ms"] for row in timings if row["candidates"] == candidates]
        summary[str(candidates)] = {"samples": len(samples), "p50_ms": float(np.median(samples)),
            "p95_ms": float(np.quantile(samples, .95)), "max_ms": max(samples)}
    result = {"run": str(root), "config_sha256": sha256(root / "config.json"),
        "replay": replay, "timing": summary,
        "timing_internal_physics_frames": sum(row["candidates"] * 20 * 4 for row in timings),
        "caveat": "reset-observer policy timing on recorded observations; not full online pipeline latency or independent racing evidence"}
    write_json(root / "timings.json", timings)
    write_json(root / "results.json", result)
    write_json(Path("reports/physics_budget_2026-10-28/verification.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
