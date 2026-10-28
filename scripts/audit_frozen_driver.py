"""Audit public traces and replay short prefixes; never use candidate-oracle data."""
import cProfile
import json
from pathlib import Path
import pstats
import shutil

import numpy as np
import pandas as pd

from cars.env.protocol import DrivingEnv, ProtocolConfig
from cars.planning.policy_search import PolicyPhysicsSearch, PolicySearchConfig
from cars.research.artifacts import begin_run, read_config
from cars.training import sha256, write_json


def main() -> None:
    declaration = Path("configs/campaigns/frozen-driver-verification-2026-10-28.json")
    spec = json.loads(declaration.read_text())
    campaign = Path(spec["campaign"])
    manifest = json.loads((campaign / "config.json").read_text())
    if (spec["partition"] != "development" or spec["held_out_evaluation"]
            or spec["user_only_oracle_access"] or json.loads((campaign / "status.json").read_text())["status"] != "complete"):
        raise ValueError("Only the completed public development experiment may be audited")
    protocol = ProtocolConfig()
    root = begin_run("verification", spec["id"], protocol, spec)
    shutil.copy2(declaration, root / "campaign.json")
    rows, sources, reference_source = [], [], None
    for condition in manifest["conditions"]:
        path = Path("runs") / condition["directory"] / condition["run_id"]
        config, costs = read_config(path), json.loads((path / "cost.json").read_text())
        if reference_source is None:
            reference_source = config["source_sha256"]
        assert config["source_sha256"] == reference_source
        assert config["protocol_id"] == protocol.identifier
        assert config["settings"]["partition"] == "development" and not config["settings"]["user_only_oracle_monitor"]
        driver = config["settings"]["driver"]
        assert driver.get("actor", driver)["checkpoint_sha256"] == manifest["checkpoint_sha256"]
        table = pd.read_csv(path / "metrics.csv").sort_values("episode")
        native = decisions = work = examples = 0
        all_actions, all_ego = [], []
        for item in table.to_dict("records"):
            number = item["episode"]
            diagnostic = json.loads((path / f"episodes/{number:04d}_diagnostics.json").read_text())
            with np.load(path / f"episodes/{number:04d}.npz") as data:
                action, ego = data["action"], data["observation_ego"]
                assert len(action) == len(diagnostic) == item["decisions"]
                assert len(ego) == len(action) + 1
                assert np.all(action >= [-1, 0, 0]) and np.all(action <= 1)
                assert np.isclose(data["reward"].sum(), item["return"], rtol=0, atol=1e-8)
                assert data["frames"].sum() == item["steps"]
                assert data["terminated"][-1] or data["truncated"][-1]
                native += int(data["frames"].sum())
                decisions += len(action)
                all_actions.append(action.copy())
                all_ego.append(ego[:-1].copy())
                errors = [np.asarray(d["predicted_next_ego"]) - ego[i + 1] for i, d in enumerate(diagnostic)
                          if "predicted_next_ego" in d]
                rows.append({"condition": condition["name"], "episode": number,
                    "return": item["return"], "completion": item["success"],
                    "outcome": "completion" if item["success"] else ("playfield_exit" if item["crash"] else
                               ("timeout_with_offroad_frames" if item["off_track_steps"] else "timeout_no_offroad_frames")),
                    "selected_step_ego_rmse": np.sqrt(np.mean(np.square(errors), axis=0)).tolist() if errors else None})
                for i, d in enumerate(diagnostic):
                    work += d.get("internal_physics_frames", 0)
                    examples += d["actor_forward_examples"]
                    if "base_action" in d:
                        assert d["overrode_base"] == bool(np.max(abs(action[i] - d["base_action"])) > 1e-6)
        assert native == costs["native_frames"] and decisions == costs["decisions"]
        assert work == costs["internal_physics_frames"] and examples == costs["actor_forward_examples"]
        control, states = np.concatenate(all_actions), np.concatenate(all_ego)
        sources.append({"condition": condition["name"], "config_sha256": sha256(path / "config.json"),
            "decision_weighted_mean_gas": float(control[:, 1].mean()),
            "decision_weighted_mean_brake": float(control[:, 2].mean()),
            "decision_weighted_mean_absolute_steer": float(abs(control[:, 0]).mean()),
            "decision_weighted_mean_forward_speed": float(states[:, 0].mean())})
    frames = internal = 0
    for condition in manifest["conditions"]:
        if condition["name"] not in spec["replay_conditions"]:
            continue
        path = Path("runs") / condition["directory"] / condition["run_id"]
        driver = PolicyPhysicsSearch(Path(manifest["checkpoint"]), protocol,
            PolicySearchConfig(candidates=condition["candidates"], planning_seed=condition["planning_seed"]))
        env = DrivingEnv(protocol, "development")
        try:
            observation, _ = env.reset()
            with np.load(path / "episodes/0000.npz") as data:
                for i in range(spec["replay_decisions"]):
                    for key, value in observation.items():
                        np.testing.assert_array_equal(value, data["observation_" + key][i])
                    action = driver.act(observation)
                    np.testing.assert_array_equal(action, data["action"][i])
                    observation, reward, done, truncated, info = env.step(action)
                    assert reward == data["reward"][i] and done == data["terminated"][i] and truncated == data["truncated"][i]
                    frames += info["native_frames"]
                    internal += driver.diagnostics["internal_physics_frames"]
        finally:
            driver.reset()
            env.close()
    assert frames <= spec["native_frame_cap"] and internal <= spec["internal_physics_frame_cap"]
    profile_source = Path("runs/verification/policy_search_selected_preflight_20261028")
    assert read_config(profile_source)["settings"]["partition"] == "train"
    timings, profile_work = [], 0
    with np.load(profile_source / "observations.npz") as snapshots:
        for count in (7, 21):
            policy = PolicyPhysicsSearch(Path(manifest["checkpoint"]), protocol, PolicySearchConfig(candidates=count))
            profile = cProfile.Profile()
            try:
                for i in range(3):
                    policy.reset()
                    profile.runcall(policy.act, {key: value[i].copy() for key, value in snapshots.items()})
                    profile_work += policy.diagnostics["internal_physics_frames"]
            finally:
                policy.reset()
            profile.dump_stats(root / f"profile_{count}.prof")
            stats = pstats.Stats(profile)
            entries = [{"file": file, "line": line, "function": function, "primitive_calls": values[0],
                        "calls": values[1], "self_seconds": values[2], "cumulative_seconds": values[3]}
                       for (file, line, function), values in stats.stats.items()]
            timings.append({"candidates": count, "total_seconds": stats.total_tt,
                "top_cumulative": sorted(entries, key=lambda row: -row["cumulative_seconds"])[:35]})
    assert profile_work <= spec["profile_internal_physics_frame_cap"]
    report = {"declaration_sha256": sha256(declaration), "trace_checks_passed": True,
        "identical_production_sources": True, "sources": sources, "episode_outcomes": rows,
        "replay_exact": True, "replay_native_frames": frames, "replay_internal_physics_frames": internal,
        "profile_internal_physics_frames": profile_work, "profile": timings,
        "profile_context": "three instrumented train-only cached snapshots; KL-training trial running concurrently; attribution only",
        "user_only_oracle_values_read": False, "held_out_evaluation": False}
    write_json(root / "result.json", report)
    write_json(Path("reports/frozen_driver_2026-10-28/audit.json"), report)
    print(json.dumps({"verification": str(root), "replay_native_frames": frames,
        "replay_internal_physics_frames": internal, "profile_internal_physics_frames": profile_work}, indent=2))


if __name__ == "__main__":
    main()
