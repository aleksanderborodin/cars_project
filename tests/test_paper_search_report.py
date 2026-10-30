"""Synthetic end-to-end report preserves negative seeds and paired contrasts."""
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import pytest

from cars.training import sha256, write_json


def test_all_policy_seeds_and_paired_contrasts_are_retained(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    rows = []
    methods = ("direct", "fixed", "random7_s0", "random7_s1", "random7_s2",
               "random21_s0", "random21_s1", "random21_s2", "search7", "search21")
    for seed in (0, 1, 2):
        for method in methods:
            root = tmp_path / f"seed{seed}_{method}"
            root.mkdir()
            delta = (-10, 5, 20)[seed] if method == "search21" else (0 if method == "direct" else 2)
            table = pd.DataFrame({"track_seed": np.arange(200),
                "return": np.arange(200) + 100 + delta, "success": np.zeros(200, dtype=bool)})
            table.to_csv(root / "metrics.csv", index=False)
            for name in ("config.json", "cost.json", "summary.json"):
                write_json(root / name, {})
            rows.append({"seed": seed, "condition": method, "evaluation": str(root),
                "metrics_sha256": sha256(root / "metrics.csv"), "config_sha256": sha256(root / "config.json"),
                "mean_return": float(table["return"].mean()), "completion": 0., "reused": seed == 0})
    write_json(campaign / "results.json", {"conditions": rows, "new_costs": []})
    write_json(campaign / "status.json", {"status": "evaluations_complete"})
    output = tmp_path / "report"
    script = Path(__file__).resolve().parents[1] / "scripts/summarize_paper_search.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--campaign", str(campaign), "--output", str(output)])
    runpy.run_path(str(script), run_name="__main__")
    summary = json.loads((output / "summary.json").read_text())
    primary = summary["contrasts"][0]
    assert [r["mean_return_change"] for r in primary["by_seed"]] == [-10, 5, 20]
    assert primary["mean_across_seeds"] == 5
    assert len(summary["conditions"]) == 30
    assert not summary["held_out_evaluation"]
    provenance = json.loads((output / "provenance.json").read_text())
    assert "status.json" not in provenance["campaign"]
    assert provenance["campaign_status_at_analysis"]["status"] == "evaluations_complete"
    # Missing a poor seed/condition must fail, rather than publishing a partial success table.
    write_json(campaign / "results.json", {"conditions": rows[1:], "new_costs": []})
    with pytest.raises(ValueError, match="All three seeds"):
        runpy.run_path(str(script), run_name="__main__")
