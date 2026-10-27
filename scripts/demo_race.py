"""Drive every variant on one demo track (a seed outside all splits) and write episode logs for the video.

    uv run python scripts/demo_race.py --seed 30000 --out runs/demo/seed_30000

Checkpoints of runs still training are copied first, so the race uses one fixed snapshot.
"""
import argparse
import json
import shutil
from pathlib import Path

from cars.eval.runner import _episode
from cars.policies.baseline import BaselinePolicy
from cars.policies.sb3 import load_learned_policy

# key: (method, model path or None, video name, tag)
VARIANTS = {
    "base": ("base", None, "Baseline", "PID"),
    "bc": ("il_bc", "runs/il_bc/task2_smoke", "IL · BC (smoke)", "BC"),
    "dagger": ("il_dagger", "runs/il_dagger/task2_smoke", "IL · DAgger (smoke)", "DAG"),
    "ppo1": ("rl_ppo", "runs/rl_ppo/full_inf_s0/best/best_model.zip", "RL · PPO", "PPO"),
    "ppo2": ("rl_ppo", "runs/rl_ppo/ar4_inf_s0/best/best_model.zip", "RL · PPO ×4", "PPO4"),
    "sac": ("rl_sac", "runs/rl_sac/full_inf_s0/best/best_model.zip", "RL · SAC", "SAC"),
}


def snapshot(model: str, dest: Path) -> str:
    """Copy a best checkpoint (and its normalisation stats and run config) so training cannot overwrite it mid-race."""
    src = Path(model)
    if src.is_dir():
        return model
    dest.mkdir(parents=True, exist_ok=True)
    for f in (src, src.parent / "vecnormalize.pkl"):
        shutil.copy2(f, dest / f.name)
    shutil.copy2(src.parents[1] / "config.json", dest / "config.json")
    return str(dest / src.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=30000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if 10000 <= a.seed < 10100 or 20000 <= a.seed < 20100 or a.seed < 100:
        raise SystemExit("demo seed must be outside seen-*, unseen and hard")
    results = {}
    for key, (method, model, name, tag) in VARIANTS.items():
        directory = a.out / key
        if method == "base":
            policy = BaselinePolicy()
        else:
            policy = load_learned_policy(method, snapshot(model, a.out / "_checkpoints" / key))
        meta = {"method": method, "run_id": key, "split": "demo", "hard": False}
        row = _episode((policy, 0, a.seed, str(directory), meta))
        results[key] = {"name": name, "tag": tag, "model": model, "log": str(next(directory.glob("*.json.gz"))),
                        **{k: row[k] for k in ("success", "lap_time_s", "off_track_events", "return", "progress")}}
        print(f"{name:22s} return {row['return']:7.1f}  lap {row['lap_time_s'] or '-'}  off {row['off_track_events']}", flush=True)
    (a.out / "race.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
