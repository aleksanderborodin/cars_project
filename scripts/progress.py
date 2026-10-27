"""One line per run: env steps so far and the mean return of the last 50 training episodes.

    uv run python scripts/progress.py runs/rl_ppo/full_inf_s0 runs/rl_sac/full_inf_s0
"""
import sys
from pathlib import Path

import pandas as pd

for run in map(Path, sys.argv[1:]):
    frames = [pd.read_csv(f, skiprows=1) for f in sorted((run / "monitor").glob("*.monitor.csv"))]
    eps = pd.concat([f for f in frames if len(f)], ignore_index=True).sort_values("t") if any(len(f) for f in frames) else None
    if eps is None:
        print(f"{run.parent.name}/{run.name}: no finished episodes yet")
        continue
    last = eps.tail(50)
    print(f"{run.parent.name}/{run.name}: {int(eps['l'].sum()):>9,} steps  {len(eps):>5} episodes  "
          f"last-50 return {last['r'].mean():7.1f}  max {eps['r'].max():6.1f}", flush=True)
