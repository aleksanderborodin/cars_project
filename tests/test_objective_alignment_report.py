"""Synthetic adverse pilot results must remain adverse and complete."""
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import pytest

from cars.training import sha256, write_json


def test_objective_report_retains_harm_and_refuses_missing_condition(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    rows = []
    for seed in (0, 1):
        for name, scale in (("direct", None), ("scale1", 1.), ("scale025", .25), ("scale0", 0.)):
            root = tmp_path / f"s{seed}_{name}"
            (root / "episodes").mkdir(parents=True)
            offset = (5 if seed == 0 else -5) if name == "scale025" else (-10 if name == "scale0" else 0)
            pd.DataFrame({"track_seed": np.arange(20), "return": np.arange(20) + 800 + offset,
                "success": np.zeros(20, bool), "progress": np.full(20, .9),
                "off_track_steps": np.zeros(20)}).to_csv(root / "metrics.csv", index=False)
            write_json(root / "config.json", {"settings": {"driver": {"penalty_scale": scale}}})
            for episode in range(20):
                np.savez(root / f"episodes/{episode:04d}.npz", frames=np.array([4, 4]),
                    observation_ego=np.full((3, 3), 5., np.float32))
            rows.append({"seed": seed, "condition": name, "penalty_scale": scale, "evaluation": str(root),
                "reused": name == "direct", "metrics_sha256": sha256(root / "metrics.csv"),
                "config_sha256": sha256(root / "config.json")})
    write_json(campaign / "results.json", {"conditions": rows, "new_costs": []})
    write_json(campaign / "config.json", {})
    script = Path(__file__).resolve().parents[1] / "scripts/summarize_objective_alignment.py"
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", [str(script), "--campaign", str(campaign), "--output", str(output)])
    runpy.run_path(str(script), run_name="__main__")
    result = json.loads((output / "summary.json").read_text())
    assert len(result["conditions"]) == 8
    assert result["screened_for_separately_declared_broader_validation"] == []
    effects = [c["mean_return_change"] for c in result["contrasts"]
               if c["condition"] == "scale025" and c["control"] == "scale1"]
    assert effects == [5., -5.]
    assert result["automatic_promotion_or_expansion"] is False
    write_json(campaign / "results.json", {"conditions": rows[:-1], "new_costs": []})
    with pytest.raises(ValueError, match="every declared condition"):
        runpy.run_path(str(script), run_name="__main__")
