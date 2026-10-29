"""Scientific reporting gates: exact replay and common-track pairing, synthetic data only."""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_replay_rejects_changed_action_and_observation_but_ignores_timing(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    verify = importlib.import_module("reference_replication_common").verify_replay
    old, new = tmp_path / "old", tmp_path / "new"
    for path in (old, new):
        (path / "episodes").mkdir(parents=True)
    records = {"action": np.array([[0., 1., 0.]]), "reward": np.array([1.]),
        "observation_ego": np.zeros((2, 3)), "frames": np.array([4]),
        "latency_ms": np.array([10.]), "sensor_ms": np.array([1.]), "policy_ms": np.array([9.])}
    np.savez(old / "episodes/0000.npz", **records)
    np.savez(new / "episodes/0000.npz", **{**records, "latency_ms": np.array([100.])})
    verify(old, new, 0)
    for key in ("action", "observation_ego", "frames", "reward"):
        np.savez(new / "episodes/0000.npz", **{**records, key: records[key] + 1})
        with pytest.raises(AssertionError):
            verify(old, new, 0)


def test_panels_preserve_losses_and_reject_misaligned_tracks(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    contrasts = importlib.import_module("summarize_reference_replication").contrasts
    base = pd.DataFrame({"track_seed": np.arange(20), "return": np.zeros(20),
        "success": np.zeros(20), "progress": np.zeros(20), "off_track_steps": np.zeros(20)})
    candidate = base.copy()
    candidate["return"] = [10.] * 5 + [-2.] * 15
    tables = {"true_tail": candidate, "tangent": base.copy(), "local_context": base.copy()}
    result = contrasts(tables, {"bootstrap": {"reps": 100, "seed": 0}})
    selected = {r["panel"]: r for r in result if r["reference"] == "tangent" and r["metric"] == "return"}
    assert selected["all20"]["mean_difference"] == 1.
    assert selected["original5"]["mean_difference"] == 10.
    assert selected["additional15"]["mean_difference"] == -2.
    assert selected["all20"]["negative"] == 15
    assert "nominal_track_bootstrap_95_percent" not in selected["original5"]
    tables["tangent"].loc[19, "track_seed"] = 100
    with pytest.raises(AssertionError):
        contrasts(tables, {"bootstrap": {"reps": 100, "seed": 0}})
