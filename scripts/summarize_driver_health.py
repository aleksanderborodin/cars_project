"""Record the first-block health review from ordinary development/training logs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cars.training import sha256, write_json


def main() -> None:
    campaign = Path("runs/campaigns/driver_first_block_20261028")
    selection = json.loads((campaign / "selection.json").read_text())["selected"]
    training = Path("runs/driver_training/driver_main_20261028_s0_to2m")
    learning = pd.read_json(training / "learning.jsonl", lines=True)
    columns = ["train/approx_kl", "train/clip_fraction", "train/explained_variance", "train/std"]
    phases = []
    for low, high in [(0, 500000), (500000, 1007379), (1007379, 2006414)]:
        phase = learning[(learning.native_frames > low) & (learning.native_frames <= high)]
        phases.append({"after_frames": low, "through_frames": high, "rollout_updates": len(phase),
                       "diagnostics": phase[columns].agg(["median", "max"]).to_dict()})
    roots = {"selected_1m": Path(selection["evaluation"]),
             "final_2m": Path("runs/recurrent_ppo/driver_s0_dev_002006414")}
    tables, summaries, sources = {}, {}, []
    for name, root in roots.items():
        config = json.loads((root / "config.json").read_text())
        assert config["settings"]["partition"] == "development"
        table = pd.read_csv(root / "metrics.csv").sort_values("track_seed")
        action, ego = [], []
        for episode in sorted(root.glob("episodes/*.npz")):
            with np.load(episode) as data:
                action.append(data["action"])
                ego.append(data["observation_ego"][:-1])
        controls, states = np.concatenate(action), np.concatenate(ego)
        tables[name] = table
        summaries[name] = {"mean_return": float(table["return"].mean()), "completion": float(table.success.mean()),
            "mean_progress": float(table.progress.mean()), "mean_offroad_frames": float(table.off_track_steps.mean()),
            "mean_offroad_events": float(table.off_track_events.mean()), "playfield_exits": int(table.crash.sum()),
            "decision_weighted_mean_forward_speed": float(states[:, 0].mean()),
            "decision_weighted_mean_absolute_steering": float(abs(controls[:, 0]).mean()),
            "decision_weighted_mean_gas": float(controls[:, 1].mean()),
            "decision_weighted_mean_brake": float(controls[:, 2].mean()),
            "decision_fraction_brake_over_0_1": float((controls[:, 2] > .1).mean())}
        sources.append({"run": str(root), "config_sha256": sha256(root / "config.json"),
                        "metrics_sha256": sha256(root / "metrics.csv")})
    a, b = tables["final_2m"], tables["selected_1m"]
    np.testing.assert_array_equal(a.track_seed, b.track_seed)
    # Different stages of one training run: resample common tracks only.
    rng = np.random.default_rng(0)
    indices = rng.integers(len(a), size=(5000, len(a)))
    differences = {}
    for key in ["return", "success", "off_track_steps"]:
        delta = a[key].to_numpy(dtype=float) - b[key].to_numpy(dtype=float)
        differences[key] = {"mean_difference": float(delta.mean()),
            "interval_95_percent": np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist(),
            "common_tracks": len(delta), "reps": 5000, "seed": 0, "selection_adjusted": False}
    write_json(Path("reports/driver_first_block_2026-10-28/health_review.json"), {
        "inputs": sources, "learning_log_sha256": sha256(training / "learning.jsonl"),
        "phase_diagnostics": phases, "driving": summaries, "final_minus_selected": differences,
        "uncertainty_scope": "one training run, adaptive development checkpoint selection, track-only unadjusted intervals",
        "decision_author": "assistant, executing the user's next-moves instruction",
        "decision": "stop this block at its declared 2M review; preserve standalone-selected 1M checkpoint for the planned exploratory simulation comparison; do not extend unchanged to 8M now",
        "reason": "later return/completion regression and increased off-road time; larger approximate KL and clipping coincide with deterioration",
        "causal_limit": "these logs do not establish that large PPO updates caused the regression; controlled repair still required",
        "future_probe": "separately declare conservative-update trial, such as a target-KL cap, from a recorded checkpoint or fresh seed with all restart costs; no silent training-config change",
        "held_out_evaluation": False, "user_only_oracle_values_read": False})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
