"""Synthetic public records exercise budget and whole-track pairing rejection."""
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from cars.env.partitions import Partitions
from cars.training import write_json


def setup_config(root):
    (root / "episodes").mkdir()
    write_json(root / "config.json", {"kind": "development_research",
        "partition_sha256": Partitions().fingerprint, "settings": {"episodes": 2}})


def test_campaign_rejects_missing_shift_or_wrong_work(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    verify = importlib.import_module("sequence_campaign_common").verify_work
    setup_config(tmp_path)
    write_json(tmp_path / "cost.json", {"actor_forward_examples": 1600, "decisions": 4})
    for episode in range(2):
        write_json(tmp_path / f"episodes/{episode:04d}_diagnostics.json", [
            {"candidates": 21, "model_transitions": 420, "internal_physics_frames": 1680,
             "actor_forward_examples": 400, "actor_forward_calls": 39, "batch_sizes": [13, 8],
             "cached_rollouts_reused": 1, "shifted_elites_reevaluated": step, "selected_batch": 0} for step in range(2)])
    check = verify(tmp_path, {}, "icem")
    assert check["checked_decisions"] == 4 and check["shifted_elites_reevaluated"] == 2
    path = tmp_path / "episodes/0001_diagnostics.json"
    data = json.loads(path.read_text())
    for key, value in (("shifted_elites_reevaluated", 0), ("internal_physics_frames", 1660)):
        changed = [dict(d) for d in data]
        changed[1][key] = value
        write_json(path, changed)
        with pytest.raises(ValueError):
            verify(tmp_path, {}, "icem")


def test_campaign_rejects_duplicate_track_and_incomplete_panel(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    verify = importlib.import_module("sequence_campaign_common").complete_table
    setup_config(tmp_path)
    table = pd.DataFrame({"episode": [0, 1], "track_seed": Partitions().seeds("development", 2),
        "partition": "development", "information": "local", "partition_sha256": Partitions().fingerprint,
        "return": [500., 600.], "progress": [.4, .5], "steps": [8, 8], "decisions": [2, 2], "model_transitions": [840, 840]})
    write_json(tmp_path / "cost.json", {"native_frames": 16, "decisions": 4, "internal_physics_frames": 6720})
    table.to_csv(tmp_path / "metrics.csv", index=False)
    spec = {"partition_sha256": Partitions().fingerprint}
    assert len(verify(tmp_path, spec, 2)) == 2
    duplicate = table.copy()
    duplicate.loc[1, "track_seed"] = duplicate.loc[0, "track_seed"]
    for invalid in (duplicate, table.iloc[:1]):
        invalid.to_csv(tmp_path / "metrics.csv", index=False)
        with pytest.raises(AssertionError):
            verify(tmp_path, spec, 2)
