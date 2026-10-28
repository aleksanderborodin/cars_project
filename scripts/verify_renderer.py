"""Compare renderer outputs and replay complete recorded development episodes."""
import json
from pathlib import Path
import time

import numpy as np

from cars.env.protocol import DrivingEnv, ProtocolConfig
from cars.planning.policy_search import PolicyPhysicsSearch, PolicySearchConfig
from cars.research.artifacts import begin_run, read_config
from cars.training import sha256, write_json


def main() -> None:
    declaration = Path("configs/campaigns/renderer-verification-2026-10-28.json")
    spec = json.loads(declaration.read_text())
    assert spec["partition"] == "development" and not spec["held_out_evaluation"] and not spec["user_only_oracle_access"]
    campaign = Path(spec["reference_campaign"])
    manifest = json.loads((campaign / "config.json").read_text())
    assert json.loads((campaign / "status.json").read_text())["status"] == "complete"
    checkpoint, protocol = Path(manifest["checkpoint"]), ProtocolConfig()
    assert sha256(checkpoint / "model.zip") == manifest["checkpoint_sha256"]
    root = begin_run("verification", spec["id"], protocol, spec)
    source = Path(spec["cached_train_snapshots"])
    assert read_config(source)["settings"]["partition"] == "train"
    rows, snapshot_work = [], 0
    with np.load(source / "observations.npz") as data:
        for count in spec["candidate_counts"]:
            policies = {renderer: PolicyPhysicsSearch(checkpoint, protocol,
                PolicySearchConfig(candidates=count, observation_renderer=renderer)) for renderer in spec["renderers"]}
            try:
                for i in range(3):
                    results = {}
                    order = spec["renderers"] if i % 2 == 0 else list(reversed(spec["renderers"]))
                    for renderer in order:
                        policy = policies[renderer]
                        policy.reset()
                        started = time.perf_counter()
                        action = policy.act({key: value[i].copy() for key, value in data.items()})
                        rows.append({"candidates": count, "snapshot": i, "renderer": renderer,
                                     "policy_ms": (time.perf_counter() - started) * 1000})
                        results[renderer] = (action, policy.candidate_actions.copy(), policy.candidate_costs.copy())
                        snapshot_work += policy.diagnostics["internal_physics_frames"]
                    for a, b in zip(results["reference"], results["sparse"], strict=True):
                        np.testing.assert_array_equal(a, b)
            finally:
                for policy in policies.values():
                    policy.reset()
    assert snapshot_work <= spec["snapshot_internal_physics_frame_cap"]
    write_json(root / "snapshots.json", rows)
    print(json.dumps({"snapshots_match_exactly": True, "timings": rows}), flush=True)
    native, internal, replays = 0, 0, []
    for count in spec["candidate_counts"]:
        condition = next(item for item in manifest["conditions"] if item["name"] == f"search{count}")
        path = Path("runs") / condition["directory"] / condition["run_id"]
        policy = PolicyPhysicsSearch(checkpoint, protocol, PolicySearchConfig(candidates=count, observation_renderer="sparse"))
        env = DrivingEnv(protocol, "development")
        timing = []
        try:
            observation, _ = env.reset()
            diagnostic = json.loads((path / "episodes/0000_diagnostics.json").read_text())
            with np.load(path / "episodes/0000.npz") as data:
                for i, expected in enumerate(data["action"]):
                    for key, value in observation.items():
                        np.testing.assert_array_equal(value, data["observation_" + key][i])
                    start = time.perf_counter()
                    action = policy.act(observation)
                    timing.append((time.perf_counter() - start) * 1000)
                    np.testing.assert_array_equal(action, expected)
                    assert policy.diagnostics["selected_candidate"] == diagnostic[i]["selected_candidate"]
                    assert policy.diagnostics["predicted_cost"] == diagnostic[i]["predicted_cost"]
                    observation, reward, done, truncated, info = env.step(action)
                    assert reward == data["reward"][i] and done == data["terminated"][i] and truncated == data["truncated"][i]
                    native += info["native_frames"]
                    internal += policy.diagnostics["internal_physics_frames"]
                    write_json(root / "progress.json", {"status": "running", "candidates": count, "decision": i + 1,
                        "episode_decisions": len(data["action"]), "native_frames": native, "internal_physics_frames": internal})
                for key, value in observation.items():
                    np.testing.assert_array_equal(value, data["observation_" + key][-1])
                assert done or truncated
        finally:
            policy.reset()
            env.close()
        replays.append({"candidates": count, "episode": 0, "decisions": len(timing),
            "policy_p50_ms": float(np.median(timing)), "policy_p95_ms": float(np.quantile(timing, .95)),
            "exact_replay": True})
        print(f"Exact full-episode replay: {count} candidates", flush=True)
    assert native <= spec["native_frame_cap"] and internal <= spec["replay_internal_physics_frame_cap"]
    result = {"declaration_sha256": sha256(declaration), "snapshots_match_exactly": True, "timings": rows,
        "full_episode_replays": replays, "native_frames": native, "replay_internal_physics_frames": internal,
        "snapshot_internal_physics_frames": snapshot_work,
        "interpretation": "computational refactor with sampled exact parity; two full replayed races, not a new performance campaign; timings overlap KL training",
        "user_only_oracle_values_read": False, "held_out_evaluation": False}
    write_json(root / "result.json", result)
    write_json(Path("reports/frozen_driver_2026-10-28/renderer_verification.json"), result)
    write_json(root / "progress.json", {"status": "complete", "native_frames": native, "internal_physics_frames": internal})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
