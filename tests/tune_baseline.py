"""Manual seen-10-only tuning; not a pytest test. Each run saves full logs/config."""
from dataclasses import replace
from pathlib import Path

from cars.eval.runner import evaluate
from cars.policies.baseline import BaselineConfig, BaselinePolicy

if __name__ == "__main__":
    root = Path("runs/base")
    for accel in (60, 90, 120):
        cfg = replace(BaselineConfig(), lateral_accel=accel, braking_accel=40, max_speed=100)
        path = root / f"tune_accel_{accel}"
        if (path / "config.json").exists():
            continue
        df = evaluate(BaselinePolicy(cfg), "seen-10", 10, path)
        print(accel, df[["success", "return", "off_track_steps", "progress", "steps"]].mean().to_dict(), flush=True)
