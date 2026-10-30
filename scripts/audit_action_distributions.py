"""Execute a declared offline gSDE audit using only permitted recorded inputs."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
import torch

from cars.env.native_recipe import NativeRecipeProtocol
from cars.env.partitions import Partitions
from cars.env.protocol import ProtocolConfig
from cars.research.action_distribution import replay_distribution, summarize_distribution
from cars.research.artifacts import begin_run, read_config
from cars.training import sha256, write_json


def load_history(root: Path, episode: int, native: bool) -> tuple:
    # Explicit allowlist: no truth fields or evaluator state enter actor inference.
    with np.load(root / f"episodes/{episode:04d}.npz") as trace:
        actions = trace["action"].copy()
        n = len(actions)
        observations = (trace["observation"].copy() if native else
            {key: trace["observation_" + key][:n].copy()
             for key in ("road", "visible", "ego", "previous_action")})
    return observations, actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    declaration = parser.parse_args().manifest
    spec = json.loads(declaration.read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    campaign = Path("runs/campaigns") / spec["id"]
    campaign.mkdir(parents=True, exist_ok=False)
    shutil.copy2(declaration, campaign / "config.json")
    shutil.copy2(__file__, campaign / "runner.py")
    rows, cases_done = [], []
    decisions = actor_examples = 0
    started = time.perf_counter()
    try:
        for case in spec["cases"]:
            checkpoint = Path(case["checkpoint"])
            for name, expected in case["checkpoint_sha256"].items():
                if sha256(checkpoint / name) != expected:
                    raise ValueError("Frozen checkpoint changed")
            config = read_config(checkpoint)
            native = case["family"] == "native"
            protocol = (NativeRecipeProtocol(**config["protocol"]) if native else ProtocolConfig(**config["protocol"]))
            settings = {"partition": "development", "kind": "offline_actor_observation_audit", "case": case,
                "episodes_per_panel": spec["episodes_per_panel"], "new_environment_frames": 0,
                "clipped_normal": "analytic per-history marginal; gSDE temporal correlations not simulated",
                "gas_brake_overlap_threshold": .05, "episode_seeds": Partitions().seeds("development", 5)}
            root = begin_run("action_audit", case["run_id"], protocol, settings)
            model = RecurrentPPO.load(checkpoint / "model.zip", device="cpu")
            parameter_scale = float(model.policy.action_dist.get_std(model.policy.log_std).detach().mean())
            for panel, key in (("own", "evaluation"), ("common", "anchor_evaluation")):
                original = Path(case[key])
                evaluation = read_config(original)
                if evaluation["protocol_id"] != config["protocol_id"] or evaluation["settings"]["partition"] != "development":
                    raise ValueError("Matched protocol/development history required")
                metrics = pd.read_csv(original / ("episodes.csv" if native else "metrics.csv"))
                if metrics.track_seed.head(5).tolist() != Partitions().seeds("development", 5):
                    raise ValueError("Wrong recorded development panel")
                for name, expected in case["history_sha256"][key].items():
                    if sha256(original / name) != expected:
                        raise ValueError("Frozen recorded observations changed")
                episodes = []
                for episode in range(spec["episodes_per_panel"]):
                    history, recorded = load_history(original, episode, native)
                    arrays = replay_distribution(model, history, recorded if panel == "own" else None)
                    n = len(arrays["raw_mean"])
                    decisions += n
                    actor_examples += int(arrays["actor_forward_examples"])
                    if decisions > spec["replayed_decision_cap"] or actor_examples > spec["actor_forward_cap"]:
                        raise RuntimeError("Declared offline inference budget exceeded")
                    np.savez_compressed(root / f"{panel}_{episode:02d}.npz", **arrays)
                    episodes.append(arrays)
                combined = {key: np.concatenate([e[key] for e in episodes]) for key in (
                    "raw_mean", "marginal_std", "deterministic_action", "clipped_action_std",
                    "expected_clipped_action", "lower_clip_probability", "upper_clip_probability",
                    "gas_brake_overlap_probability")}
                row = {"family": case["family"], "checkpoint_label": case["label"], "panel": panel,
                    "run_id": case["run_id"], "parameter_scale_mean": parameter_scale,
                    "own_recorded_action_max_error": max(float(e["recorded_action_max_error"]) for e in episodes) if panel == "own" else None,
                    "predictor_parity_max_error": max(float(e["predictor_parity_max_error"]) for e in episodes),
                    **summarize_distribution(combined, model.action_space.low, model.action_space.high)}
                rows.append(row)
                write_json(root / f"{panel}_summary.json", row)
                print(json.dumps({k: row[k] for k in ("family", "checkpoint_label", "panel", "decisions")}), flush=True)
            training = pd.read_json(case["learning_log"], lines=True)
            clock = "decisions" if native else "native_frames"
            window = training[training[clock] <= case["training_clock"]].tail(20)
            keys = [k for k in ("train/approx_kl", "train/clip_fraction", "train/std", "train/explained_variance") if k in window]
            write_json(root / "update_window.json", {"rows": len(window), "median": window[keys].median().to_dict(),
                "note": "Last up-to-20 completed rollout logs before checkpoint; observational, not a causal update ablation"})
            write_json(root / "status.json", {"status": "complete", "environment_steps": 0})
            cases_done.append(str(root))
            write_json(campaign / "status.json", {"status": "running", "completed": cases_done,
                                                  "replayed_decisions": decisions})
        write_json(campaign / "results.json", rows)
        write_json(campaign / "status.json", {"status": "complete", "completed": cases_done,
            "replayed_decisions": decisions, "actor_forward_examples": actor_examples,
            "new_environment_frames": 0, "internal_physics_steps": 0, "optimizer_updates": 0,
            "wall_seconds": time.perf_counter() - started})
    except Exception as error:
        write_json(campaign / "status.json", {"status": "failed", "error": str(error),
            "completed": cases_done, "replayed_decisions": decisions, "actor_forward_examples": actor_examples})
        raise


if __name__ == "__main__":
    main()
