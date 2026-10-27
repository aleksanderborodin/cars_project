"""Mean±sample-std and rliable IQM with reproducible 95% bootstrap CIs.

Independent training seeds (when supplied in metrics.csv) are the bootstrap
units; otherwise independent run_ids are used. A single baseline run only
supports an episode/track bootstrap, explicitly labelled in the output.
"""
import argparse
from contextlib import contextmanager
from threading import Lock
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from rliable import library, metrics

_BOOTSTRAP_LOCK = Lock()


@contextmanager
def _bootstrap_random_state(seed: int):
    # rliable 1.2's update_indices uses global np.random.choice despite its
    # random_state argument. Scope and restore that RNG for reproducible CIs.
    with _BOOTSTRAP_LOCK:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)


METRICS = ("return", "success", "lap_time_s", "off_track_events", "off_track_steps", "crash", "progress")


def aggregate(paths: list[str | Path], *, reps: int = 10000, seed: int = 0) -> pd.DataFrame:
    if not paths:
        raise ValueError("Provide at least one metrics.csv")
    if reps < 1:
        raise ValueError("reps must be positive")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "run_id" not in frame:
            frame["run_id"] = str(Path(path).resolve().parent)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    rows = []
    for (method, split), group in data.groupby(["method", "split"], sort=True):
        for metric in METRICS:
            values = group[metric].dropna().astype(float)
            independent = "training_seed" if "training_seed" in group and group["training_seed"].notna().all() else "run_id"
            run_means = group.groupby(independent)[metric].mean().dropna().astype(float)
            if len(run_means) > 1:
                scores = run_means.to_numpy().reshape(-1, 1)
                unit = "training_seeds" if independent == "training_seed" else "runs"
            else:
                # There is no training-seed CI to estimate from one baseline.
                scores = values.to_numpy().reshape(-1, 1)
                unit = "episodes"
            if len(scores):
                with _bootstrap_random_state(seed), warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="random_state is deprecated", category=FutureWarning)
                    point, interval = library.get_interval_estimates(
                        {"score": scores}, lambda x: np.array([metrics.aggregate_iqm(x)]),
                        reps=reps, confidence_interval_size=0.95, random_state=np.random.RandomState(seed))
                iqm = float(point["score"][0])
                lower, upper = map(float, interval["score"][:, 0])
            else:
                iqm = lower = upper = float("nan")
            mean, std = values.mean(), values.std(ddof=1)
            rows.append({"method": method, "split": split, "metric": metric, "n": len(values),
                         "n_runs": len(run_means), "mean": mean, "std": std,
                         "mean±std": f"{mean:.3f} ± {std:.3f}", "iqm": iqm,
                         "ci_low": lower, "ci_high": upper, "bootstrap_unit": unit})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+")
    parser.add_argument("--out")
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    table = aggregate(args.metrics, reps=args.reps, seed=args.seed)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(target, index=False)
    print(table[["method", "split", "metric", "mean±std", "iqm", "ci_low", "ci_high", "bootstrap_unit"]].to_string(index=False))


if __name__ == "__main__":
    main()
