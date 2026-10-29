"""Public development integrity checks shared by sequence-search campaign tools."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cars.env.partitions import Partitions
from cars.research.artifacts import read_config
from cars.training import sha256


def verify_manifest(spec: dict) -> None:
    if (spec["stage"] not in ("smoke", "comparison") or spec["partition"] != "development"
            or spec["private_oracle_access"] or spec["held_out_evaluation"] or spec["training_frames"]
            or spec["partition_sha256"] != Partitions().fingerprint
            or spec["protocol"]["view"] != "local" or spec["protocol"]["action_repeat"] != 4
            or spec["search"]["candidates"] != 21 or spec["search"]["horizon"] != 20
            or spec["search"]["observation_renderer"] != "sparse"):
        raise ValueError("Declared local development experiment only")
    design = [(c["name"], c["episodes"]) for c in spec["conditions"]]
    expected = ([("legacy_replay", 1), ("colored", 2), ("icem", 2)] if spec["stage"] == "smoke"
                else [("colored", 20), ("icem", 20)])
    if design != expected or spec["seeds"] != Partitions().seeds("development", 2 if spec["stage"] == "smoke" else 20):
        raise ValueError("Changed fixed design or track order")
    cap = sum(n for _, n in expected) * 1000
    if spec["native_frame_cap"] != cap or spec["internal_physics_frame_cap"] != cap * 420:
        raise ValueError("Budget must include every newly executed condition/replay")
    for artifact in spec["required_artifacts"]:
        if sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError(f"Pinned artifact changed: {artifact['path']}")
    if read_config(Path(spec["checkpoint"]))["protocol"] != spec["protocol"]:
        raise ValueError("Wrong actor protocol")


def verify_run(path: Path, spec: dict, *, mode: str, fresh: bool = False, episodes: int | None = None) -> dict:
    config = read_config(path)
    driver = config["settings"]["driver"]
    actor = driver if mode == "direct" else driver["actor"]
    if (config["protocol"] != spec["protocol"] or config["versions"] != spec["versions"]
            or config["partition_sha256"] != spec["partition_sha256"]
            or config["settings"]["partition"] != "development" or config["settings"]["user_only_oracle_monitor"]
            or config["settings"]["privileged_reference"]
            or actor["checkpoint_sha256"] != spec["checkpoint_sha256"] or actor["evaluation_normalization_updates"]):
        raise ValueError(f"Mismatched protocol/actor: {path}")
    if mode != "direct":
        for key, value in spec["search"].items():
            if driver[key] != value:
                raise ValueError(f"Search setting changed: {key}")
        if (driver["simulator"]["vehicle_source_sha256"] != spec["vehicle_source_sha256"]
                or driver["simulator"]["evaluation_environment_state_access"] or driver["training_at_inference"]
                or "intervention" in driver):
            raise ValueError("Wrong model/information access")
        if mode in ("colored", "icem"):
            if driver["optimizer"] != spec["optimizers"][mode]:
                raise ValueError("Optimizer configuration changed")
        elif mode != "knots" or "optimizer" in driver:
            raise ValueError("Expected historical knot search")
    if fresh and (config["source_sha256"] != spec["source_sha256"] or config["settings"]["episodes"] != episodes):
        raise ValueError("Executed source/count mismatch")
    return config


def complete_table(path: Path, spec: dict, count: int) -> pd.DataFrame:
    table = pd.read_csv(path / "metrics.csv").sort_values("episode")
    recorded = read_config(path)["settings"]["episodes"]
    np.testing.assert_array_equal(table.episode, np.arange(recorded))
    np.testing.assert_array_equal(table.track_seed, Partitions().seeds("development", recorded))
    if (not (table.partition == "development").all() or not (table.information == "local").all()
            or not np.isfinite(table[["return", "progress", "steps", "decisions"]]).all().all()
            or not (table.partition_sha256 == spec["partition_sha256"]).all() or count > recorded):
        raise ValueError("Invalid/incomplete outcomes; failures cannot be dropped")
    cost = json.loads((path / "cost.json").read_text())
    if (cost["native_frames"] != int(table.steps.sum()) or cost["decisions"] != int(table.decisions.sum())
            or cost["internal_physics_frames"] != int(table.model_transitions.sum()) * 4):
        raise ValueError("Cost/table mismatch")
    return table.iloc[:count].copy()


def verify_episode_replay(reference: Path, replay: Path, episode: int) -> None:
    with np.load(reference / f"episodes/{episode:04d}.npz") as old, np.load(replay / f"episodes/{episode:04d}.npz") as new:
        if set(old.files) != set(new.files):
            raise ValueError("Replay schema mismatch")
        for key in set(old.files) - {"latency_ms", "policy_ms", "sensor_ms"}:
            np.testing.assert_array_equal(old[key], new[key], err_msg=f"episode {episode}/{key}")


def verify_work(path: Path, spec: dict, mode: str) -> dict:
    """Check declared physical/actor work in every decision; public technical logs only."""
    config = read_config(path)
    total_decisions = 0
    cached = shifted = first_batch_winners = 0
    for episode in range(config["settings"]["episodes"]):
        diagnostics = json.loads((path / f"episodes/{episode:04d}_diagnostics.json").read_text())
        for step, d in enumerate(diagnostics):
            if (d["candidates"] != 21 or d["model_transitions"] != 420
                    or d["internal_physics_frames"] != 1680 or d["actor_forward_examples"] != 400):
                raise ValueError("Per-decision work differs from declared budget")
            if mode in ("colored", "icem"):
                if (d["batch_sizes"] != [13, 8] or d["actor_forward_calls"] != 39
                        or d["cached_rollouts_reused"] != int(mode == "icem")
                        or d["shifted_elites_reevaluated"] != int(mode == "icem" and step > 0)):
                    raise ValueError("Adaptive work/memory bookkeeping changed")
                cached += d["cached_rollouts_reused"]
                shifted += d["shifted_elites_reevaluated"]
                first_batch_winners += d["selected_batch"] == 0
        total_decisions += len(diagnostics)
    cost = json.loads((path / "cost.json").read_text())
    if cost["actor_forward_examples"] != 400 * total_decisions or cost["decisions"] != total_decisions:
        raise ValueError("Total actor count mismatch")
    return {"checked_decisions": total_decisions, "cached_rollouts_reused": cached,
            "shifted_elites_reevaluated": shifted, "first_batch_winners": first_batch_winners}
