"""Descriptive public executed-action audit; no simulator or counterfactual calls.

Run from repository root after the frozen comparison report is complete:
  uv run python <this-file> --report reports/sequence_comparison_2026-10-29
Writes a new trace_audit/ subdirectory; never replaces prior outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measures(actions: np.ndarray, frames: np.ndarray, ego: np.ndarray,
             diagnostics: list[dict], repeat: int = 4) -> dict:
    n = len(actions)
    if n < 2 or actions.shape != (n, 3) or ego.shape != (n + 1, 3) or len(diagnostics) != n:
        raise ValueError("Consistent complete episode arrays required")
    if frames.shape != (n,) or (frames < 1).any() or (frames > repeat).any():
        raise ValueError("Invalid executed action duration")
    if not np.isfinite(actions).all() or not np.isfinite(ego).all():
        raise ValueError("Finite executed observations/actions required")
    row = {"decisions": n, "native_frames": int(frames.sum())}
    changes = np.abs(np.diff(actions.astype(float), axis=0)).mean(axis=0)
    for k, value in zip(("steer", "gas", "brake"), changes, strict=True):
        row[f"mean_abs_adjacent_{k}_change"] = float(value)
    has_base = ["base_action" in d for d in diagnostics]
    if any(has_base) and not all(has_base):
        raise ValueError("Incomplete base-action diagnostics")
    if all(has_base):
        base = np.asarray([d["base_action"] for d in diagnostics], float)
        if base.shape != actions.shape or not np.isfinite(base).all():
            raise ValueError("Invalid base actions")
        correction = np.abs(actions - base)
        overrides = correction.max(axis=1) > 1e-6
        np.testing.assert_array_equal(overrides, [d["overrode_base"] for d in diagnostics])
        row["override_fraction"] = float(overrides.mean())
        for k, value in zip(("steer", "gas", "brake"), correction.mean(axis=0), strict=True):
            row[f"mean_abs_{k}_correction"] = float(value)
    has_prediction = ["predicted_next_ego" in d for d in diagnostics]
    if any(has_prediction) and not all(has_prediction):
        raise ValueError("Incomplete one-step predictions")
    if all(has_prediction):
        pred = np.asarray([d["predicted_next_ego"] for d in diagnostics], float)
        if pred.shape != (n, 3) or not np.isfinite(pred).all():
            raise ValueError("Finite body velocity/yaw predictions required")
        valid = frames == repeat
        if not valid.any():
            raise ValueError("No full-duration steps to compare")
        error = pred[valid] - ego[1:][valid]
        row.update(prediction_full_actions=int(valid.sum()),
                   prediction_partial_actions_excluded=int((~valid).sum()),
                   body_velocity_vector_rmse_mps=float(np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1)))),
                   yaw_rate_rmse_radps=float(np.sqrt(np.mean(error[:, 2] ** 2))))
    return row


def self_test() -> None:
    action = np.array([[0., .1, 0.], [.2, .1, .3], [.2, 0., .3]])
    diagnostics = [{"base_action": a.tolist(), "overrode_base": False,
                    "predicted_next_ego": e} for a, e in zip(action, [[3., 4., 2.], [0., 0., 0.], [999., 999., 999.]])]
    result = measures(action, np.array([4, 4, 1]), np.zeros((4, 3)), diagnostics)
    np.testing.assert_allclose(result["body_velocity_vector_rmse_mps"], np.sqrt(12.5))
    np.testing.assert_allclose(result["yaw_rate_rmse_radps"], np.sqrt(2.))
    assert result["prediction_partial_actions_excluded"] == 1
    assert result["override_fraction"] == 0
    direct = measures(action, np.array([4, 4, 4]), np.zeros((4, 3)), [{}, {}, {}])
    assert "body_velocity_vector_rmse_mps" not in direct
    np.testing.assert_allclose(direct["mean_abs_adjacent_steer_change"], .1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("Synthetic duration exclusion, units, direct-policy and change checks passed")
        return
    if args.report != Path("reports/sequence_comparison_2026-10-29"):
        raise ValueError("This analysis is specific to the declared comparison")
    campaign = Path("runs/campaigns/sequence_comparison_20261029")
    if json.loads((campaign / "status.json").read_text())["status"] != "complete":
        raise ValueError("Completed campaign required")
    report = json.loads((args.report / "analysis.json").read_text())
    expected = {"direct": "runs/recurrent_ppo/broad_driver_20261028_direct",
                "knots": "runs/policy_physics_search/broad_driver_20261028_search21",
                "colored": "runs/policy_colored_shooting/sequence_comparison_20261029_colored",
                "icem": "runs/policy_icem/sequence_comparison_20261029_icem"}
    assert {item["condition"]: item["run"] for item in report["inputs"]} == expected
    rows = []
    for item in report["inputs"]:
        if len(item["trajectories"]) != 20:
            raise ValueError("All twenty tracks required")
        for episode, record in enumerate(item["trajectories"]):
            path = Path(item["run"]) / f"episodes/{episode:04d}.npz"
            diag = path.with_name(path.stem + "_diagnostics.json")
            assert str(path) == record["path"]
            assert sha256(path) == record["sha256"] and sha256(diag) == record["diagnostics_sha256"]
            with np.load(path, allow_pickle=False) as trace:
                values = measures(trace["action"], trace["frames"], trace["observation_ego"], json.loads(diag.read_text()))
            rows.append({"condition": item["condition"], "episode": episode, **values})
    table = pd.DataFrame(rows)
    summary = table.groupby("condition", sort=False).mean(numeric_only=True).drop(columns="episode")
    target = args.report / "trace_audit"
    target.mkdir(exist_ok=False)
    table.to_csv(target / "episodes.csv", index=False)
    summary.to_csv(target / "summary.csv")
    plan = campaign / "trace_audit_plan.json"
    shutil.copy2(plan, target / "plan.json")
    shutil.copy2(__file__, target / "audit_executed_traces.py")
    metadata = {"primary_analysis_sha256": sha256(args.report / "analysis.json"),
                "plan_sha256": sha256(plan), "script_sha256": sha256(Path(__file__)),
                "checked_trajectories": len(rows), "checked_diagnostic_files": len(rows),
                "new_environment_frames": 0, "new_model_steps": 0,
                "aggregation": "equal-weight mean of per-track statistics; counts shown are mean counts per track",
                "scope": "post-hoc descriptive executed behavior; different on-policy state distributions; no causal attribution",
                "not_measured": "counterfactual candidate outcomes, ranks, regrets, long-horizon accuracy",
                "private_diagnostics_accessed": False}
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()
