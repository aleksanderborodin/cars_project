"""Report every frozen-policy condition and descriptive paired multi-seed contrasts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads((args.campaign / "results.json").read_text())
    rows = records["conditions"]
    if len(rows) != 30 or len({(r["seed"], r["condition"]) for r in rows}) != 30:
        raise ValueError("All three seeds and ten methods are required")
    tables, provenance = {}, {}
    for row in rows:
        root = Path(row["evaluation"])
        if sha256(root / "metrics.csv") != row["metrics_sha256"]:
            raise ValueError("Recorded evaluation changed")
        table = pd.read_csv(root / "metrics.csv")
        if len(table) != 200 or sha256(root / "config.json") != row["config_sha256"]:
            raise ValueError("Wrong panel size or changed configuration")
        tables[row["seed"], row["condition"]] = table
        provenance[str(root)] = {name: sha256(root / name) for name in ("config.json", "metrics.csv", "cost.json", "summary.json")}
    tracks = tables[0, "direct"].track_seed.to_numpy()
    for table in tables.values():
        np.testing.assert_array_equal(table.track_seed, tracks)
    rng = np.random.default_rng(20261030)
    draws = rng.integers(0, 200, size=(5000, 200))
    seed_draws = rng.integers(0, 3, size=(5000, 3))

    def values(seed: int, method: str, metric: str) -> np.ndarray:
        if method.startswith("random") and method.endswith("_mean"):
            base = method[:-5]
            return np.mean([tables[seed, base + f"_s{i}"][metric].to_numpy() for i in range(3)], axis=0)
        return tables[seed, method][metric].to_numpy().astype(float)

    comparisons = []
    for first, second in (("search21", "direct"), ("search7", "direct"), ("search21", "random21_mean"),
                           ("search7", "random7_mean"), ("search21", "fixed"), ("search21", "search7")):
        data = {"contrast": first + "_minus_" + second, "by_seed": []}
        differences = []
        for seed in (0, 1, 2):
            delta = values(seed, first, "return") - values(seed, second, "return")
            differences.append(delta)
            completion = values(seed, first, "success") - values(seed, second, "success")
            data["by_seed"].append({"seed": seed, "mean_return_change": float(delta.mean()),
                "paired_track_bootstrap95": np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
                "completion_change_percentage_points": float(completion.mean() * 100)})
        delta = np.stack(differences)
        hierarchical = delta[seed_draws[:, :, None], draws[:, None, :]].mean(axis=(1, 2))
        data["mean_across_seeds"] = float(delta.mean())
        data["descriptive_seed_and_track_bootstrap95"] = np.quantile(hierarchical, [.025, .975]).tolist()
        comparisons.append(data)
    result = {"conditions": rows, "contrasts": comparisons, "new_costs": records["new_costs"],
        "uncertainty": "5000 paired track and hierarchical training-seed/track resamples, seed20261030; exploratory adaptive development, historical seed0, three actors; no multiplicity or selection correction",
        "all_seeds_and_conditions_included": True, "held_out_evaluation": False,
        "interpretation": "Report harmful and beneficial effects equally; competence and SOTA require separate evidence"}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "summary.json", result)
    write_json(args.output / "provenance.json", {"runs": provenance,
        "campaign": {p.name: sha256(p) for p in args.campaign.iterdir() if p.is_file() and p.name != "status.json"},
        "campaign_status_at_analysis": json.loads((args.campaign / "status.json").read_text()),
        "analysis_script_sha256": sha256(Path(__file__))})
    lines = ["# Frozen-policy programmed-search development comparison", "",
        "All 200 development tracks; three selected actors; historical seed0 reused. Exploratory, not held-out evidence.", "",
        "| Training seed | Condition | Mean return | Completion | Reused historical panel |",
        "|---|---|---:|---:|---|"]
    lines += [f"| {r['seed']} | {r['condition']} | {r['mean_return']:.2f} | {r['completion']:.1%} | {r['reused']} |" for r in rows]
    lines += ["", "All paired contrasts, stream-specific results and cost ledgers are in summary.json.", "",
              "Checkpoint selection used standalone driving before search. No seed or condition was dropped based on score."]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
