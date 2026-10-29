"""Integrity checks for the declared, public development replication only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cars.env.partitions import Partitions
from cars.research.artifacts import read_config
from cars.training import sha256


CONDITIONS = {"tangent": "oracle_visible", "true_tail": "oracle_continued"}


def verify_declaration(spec: dict) -> None:
    if (spec["partition"] != "development" or spec["held_out_evaluation"]
            or spec["private_oracle_access"] or spec["training_frames"] != 0
            or spec["partition_sha256"] != Partitions().fingerprint):
        raise ValueError("Public development evaluation only")
    if (spec["episodes"] != 20 or spec["replay_episodes"] != 5
            or spec["native_frame_cap"] != 40000 or spec["internal_physics_frame_cap"] != 5600000
            or spec["search"]["candidates"] != 7 or spec["search"]["horizon"] != 20
            or spec["protocol"]["action_repeat"] != 4 or set(spec["conditions"]) != set(CONDITIONS)
            or spec["seeds"] != Partitions().seeds("development", 20)):
        raise ValueError("This campaign has a fixed 20-track, two-condition design")
    for name, reference in CONDITIONS.items():
        if spec["conditions"][name]["reference"] != reference:
            raise ValueError("Reference condition changed")
    for item in spec["required_artifacts"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Pinned artifact changed: {item['path']}")
    if read_config(Path(spec["checkpoint"]))["protocol"] != spec["actor_protocol"]:
        raise ValueError("Actor protocol changed")


def verify_config(path: Path, spec: dict, reference: str | None, *, fresh: bool = False) -> dict:
    config = read_config(path)
    driver = config["settings"]["driver"]
    expected_protocol = spec["actor_protocol"] if reference is None else spec["protocol"]
    if (config["protocol"] != expected_protocol or config["versions"] != spec["versions"]
            or config["partition_sha256"] != spec["partition_sha256"]
            or config["settings"]["partition"] != "development"
            or config["settings"]["user_only_oracle_monitor"]
            or driver["actor"]["checkpoint_sha256"] != spec["checkpoint_sha256"]
            or driver["actor"]["evaluation_normalization_updates"]
            or driver["simulator"]["vehicle_source_sha256"] != spec["vehicle_source_sha256"]
            or driver["simulator"]["evaluation_environment_state_access"]
            or driver["training_at_inference"]):
        raise ValueError(f"Unmatched configuration: {path}")
    for key, expected in spec["search"].items():
        if driver[key] != expected:
            raise ValueError(f"Changed search setting: {key}")
    if reference is not None:
        if driver["intervention"] != {"reference": reference, "continuation": "oracle", "version": 1}:
            raise ValueError("Wrong intervention")
        if driver["private_candidate_oracle_access"] or driver["oracle_vehicle_state_access"]:
            raise ValueError("Forbidden state access")
    elif config["method"] != "policy_physics_search" or "intervention" in driver:
        raise ValueError("Expected historical local search context")
    if fresh and (config["source_sha256"] != spec["source_sha256"] or config["settings"]["episodes"] != 20):
        raise ValueError("Executed source/count mismatch")
    return config


def verify_replay(reference: Path, replay: Path, episode: int) -> None:
    """Compare complete public episode records; only elapsed timing may differ."""
    with np.load(reference / f"episodes/{episode:04d}.npz") as old, np.load(replay / f"episodes/{episode:04d}.npz") as new:
        if set(old.files) != set(new.files):
            raise ValueError("Episode schemas differ")
        for key in set(old.files) - {"latency_ms", "sensor_ms", "policy_ms"}:
            np.testing.assert_array_equal(old[key], new[key], err_msg=f"episode {episode}, {key}")


def verified_table(path: Path, spec: dict) -> pd.DataFrame:
    table = pd.read_csv(path / "metrics.csv").sort_values("episode")
    count = read_config(path)["settings"]["episodes"]
    np.testing.assert_array_equal(table.episode, np.arange(count))
    np.testing.assert_array_equal(table.track_seed, Partitions().seeds("development", count))
    if (not (table.partition == "development").all()
            or not (table.partition_sha256 == spec["partition_sha256"]).all()
            or not np.isfinite(table[["return", "progress", "steps", "decisions"]]).all().all()):
        raise ValueError("Invalid/incomplete outcomes; do not drop failures")
    cost = json.loads((path / "cost.json").read_text())
    if (cost["native_frames"] != int(table.steps.sum())
            or cost["internal_physics_frames"] != int(table.decisions.sum()) * 7 * 20 * 4):
        raise ValueError("Recorded cost disagrees with episode counts")
    return table.iloc[:spec["episodes"]].copy()
