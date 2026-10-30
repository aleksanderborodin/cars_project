"""Complete, explicitly adaptive two-actor score-only development pilot."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cars.training import sha256, write_json


def analyze(campaign: Path, output: Path) -> dict:
    records = json.loads((campaign / "results.json").read_text())
    rows = records["conditions"]
    expected = {(s, key) for s in (0, 1) for key in ("direct", "scale1", "scale025", "scale0")}
    if len(rows) != 8 or {(r["seed"], r["condition"]) for r in rows} != expected:
        raise ValueError("Both actors and every declared condition must be retained")
    tables, summaries, provenance = {}, [], {}
    for row in rows:
        root = Path(row["evaluation"])
        if sha256(root / "metrics.csv") != row["metrics_sha256"] or sha256(root / "config.json") != row["config_sha256"]:
            raise ValueError("Evaluation changed after recording")
        cfg = json.loads((root / "config.json").read_text())
        if row["condition"] != "direct" and cfg["settings"]["driver"]["penalty_scale"] != row["penalty_scale"]:
            raise ValueError("Score label differs from executed config")
        table = pd.read_csv(root / "metrics.csv").iloc[:20]
        if len(table) != 20 or not np.isfinite(table['return']).all():
            raise ValueError("Twenty complete development episodes are required")
        tables[row["seed"], row["condition"]] = table
        provenance[str(root)] = {name: sha256(root / name) for name in ("config.json", "metrics.csv")}
        frames = stopped = 0
        for episode in range(20):
            with np.load(root / f"episodes/{episode:04d}.npz") as trace:
                weights = trace["frames"]
                speed = np.linalg.norm(trace["observation_ego"][:-1, :2], axis=1)
                frames += int(weights.sum())
                stopped += int(weights[speed < 2.].sum())
        summaries.append({**row, "mean_return": float(table['return'].mean()),
            "completion": float(table.success.mean()), "mean_progress": float(table.progress.mean()),
            "mean_offroad_steps": float(table.off_track_steps.mean()),
            "frame_weighted_decision_speed_below2mps": stopped / frames})
    reference = tables[0, "direct"].track_seed.to_numpy()
    for table in tables.values(): np.testing.assert_array_equal(table.track_seed, reference)
    draws = np.random.default_rng(20261030).integers(0, 20, (5000, 20))
    contrasts = []
    for condition in ("scale025", "scale0"):
        for control in ("scale1", "direct"):
            for seed in (0, 1):
                delta = (tables[seed, condition]['return'] - tables[seed, control]['return']).to_numpy()
                contrasts.append({"seed": seed, "condition": condition, "control": control,
                    "mean_return_change": float(delta.mean()),
                    "descriptive_paired_track95": np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()})
    eligible = [condition for condition in ("scale025", "scale0")
                if all(c["mean_return_change"] > 0 for c in contrasts if c["condition"] == condition)]
    result = {"conditions": summaries, "contrasts": contrasts, "new_costs": records["new_costs"],
        "primary": "scale025-minus-scale1 mean official return, separately for both frozen actors",
        "screened_for_separately_declared_broader_validation": eligible,
        "automatic_promotion_or_expansion": False,
        "interpretation": "post-diagnosis adaptive development; two historical actors, reused first20 tracks; not untouched/final evidence or a SOTA result",
        "uncertainty": "5000 paired track resamples, seed20261030; not independent training uncertainty or tuning-adjusted intervals",
        "official_reward_changed": False, "private_or_heldout_access": False}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", result)
    write_json(output / "provenance.json", {"evaluations": provenance,
        "config_sha256": sha256(campaign / "config.json"), "reporter_sha256": sha256(Path(__file__))})
    lines = ["# Objective-alignment development pilot", "",
        "Adaptive first20-development-track study; same policies, candidates, model and work. No final-test claim.", "",
        "| Actor seed | Condition | Return | Completion | Frames below 2 m/s |", "|---|---|---:|---:|---:|"]
    lines += [f"| {r['seed']} | {r['condition']} | {r['mean_return']:.2f} | {r['completion']:.1%} | {r['frame_weighted_decision_speed_below2mps']:.1%} |" for r in summaries]
    lines += ["", "All conditions, negative effects and costs retained. No automatic method promotion."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.campaign, args.output)


if __name__ == "__main__": main()
