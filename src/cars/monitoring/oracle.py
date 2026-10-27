"""Privileged replay worker. Real ranking values are written ONLY to private output.

No model, optimizer, selection callback or assistant should consume these files.
This module runs in a separate process and never returns actions to the driver.
"""
import argparse
import html
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr

from cars.env.make import make_env
from cars.env.partitions import Partitions
from cars.env.state import CarState
from cars.training import write_json


def ranking_statistics(utilities: np.ndarray, predicted_costs: np.ndarray, selected: int,
                       tolerance: float = 1e-6) -> dict:
    """Tie-aware metrics. Unit tests use synthetic values, never user reports."""
    utilities, predicted_costs = np.asarray(utilities), np.asarray(predicted_costs)
    if utilities.ndim != 1 or utilities.shape != predicted_costs.shape or not 0 <= selected < len(utilities):
        raise ValueError("Invalid candidate arrays")
    if not np.isfinite(utilities).all() or not np.isfinite(predicted_costs).all():
        raise ValueError("Nonfinite candidate scores")
    value = utilities[selected]
    better = int(np.sum(utilities > value + tolerance))
    tied = int(np.sum(abs(utilities - value) <= tolerance))
    spread = float(np.max(utilities) - np.min(utilities))
    correlation = (float(spearmanr(-predicted_costs, utilities).statistic)
                   if np.ptp(utilities) > tolerance and np.ptp(predicted_costs) > tolerance else None)
    return {"selected_candidate": selected, "rank_best": 1 + better,
        "rank_worst": better + tied, "rank_mid": 1 + better + (tied - 1) / 2,
        "tie_count": tied, "top1_including_ties": better == 0,
        "top3_by_best_rank": better < 3, "chosen_utility": float(value),
        "best_utility": float(np.max(utilities)), "uniform_candidate_utility": float(utilities.mean()),
        "hindsight_regret": float(np.max(utilities) - value),
        "normalized_regret": float((np.max(utilities) - value) / spread) if spread > tolerance else 0.,
        "uninformative_all_tied": spread <= tolerance, "spearman": correlation}


def replay_request(path: Path) -> tuple[dict, dict]:
    """Returns private outcome and public resource counts to the worker only."""
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    seed = int(data["track_seed"])
    if not Partitions().contains("development", seed):
        raise ValueError("Oracle worker only accepts reserved development tracks")
    candidates = data["candidates"]
    if candidates.ndim != 3 or candidates.shape[-1] != 3 or len(data["prefix"]) != len(data["prefix_frames"]):
        raise ValueError("Malformed frozen candidate request")
    selected = int(data["selected"])
    if selected != int(np.argmin(data["predicted_costs"])):
        raise ValueError("Selected candidate differs from the frozen planner choice")
    prefix_work = future_work = 0
    outcomes = []
    for candidate in candidates:
        env = make_env(seed, obs="raw")
        try:
            env.reset()
            for action, frames in zip(data["prefix"], data["prefix_frames"], strict=True):
                for _ in range(int(frames)):
                    _, _, terminated, truncated, _ = env.step(action)
                    prefix_work += 1
                    if terminated or truncated:
                        raise ValueError("Replay prefix ended before the audit state")
            state = CarState.from_env(env)
            observed_ego = np.asarray([*state.body_velocity, state.yaw_rate], dtype=np.float32)
            if not np.array_equal(observed_ego, data["expected_ego"]):
                raise ValueError("Replay prefix does not match recorded ego observation")
            reward = 0.
            offroad = 0
            executed = 0
            ended = False
            for action in candidate:
                if not env.action_space.contains(action):
                    raise ValueError("Frozen candidate contains invalid action")
                for _ in range(int(data["action_repeat"])):
                    _, value, terminated, truncated, _ = env.step(action)
                    reward += value
                    offroad += int(CarState.from_env(env).wheels_on_road == 0)
                    executed += 1
                    future_work += 1
                    if terminated or truncated:
                        ended = True
                        break
                if ended:
                    break
            outcomes.append({"official_return": reward, "offroad_frames": offroad,
                             "native_frames": executed, "terminated": terminated, "truncated": truncated})
        finally:
            env.close()
    utilities = np.array([item["official_return"] for item in outcomes])
    result = {"episode": int(data["episode"]), "decision": int(data["decision"]),
              "horizon_decisions": candidates.shape[1], "candidate_count": len(candidates),
              "metrics": ranking_statistics(utilities, data["predicted_costs"], selected),
              "predicted_costs": data["predicted_costs"].tolist(), "outcomes": outcomes}
    return result, {"prefix_native_frames": prefix_work, "candidate_native_frames": future_work,
                    "resets": len(candidates)}


def private_report(directory: Path, rows: list[dict], complete: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"access": "user_only_do_not_read_into_any_agent_or_training_pipeline",
        "complete": complete, "scope": "Short-horizon open-loop candidate utility on development tracks; not full-race performance",
        "caveats": ["Privileged hindsight includes information the driver did not have.",
                    "Planner cost and official return differ; this is overall choice quality, not isolated prediction error.",
                    "States within tracks and candidates within states are not independent trials."], "states": rows}
    temporary = directory / "results.pending"
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "results.json")
    table_rows = []
    details = []
    for row in rows:
        m = row["metrics"]
        corr = "—" if m["spearman"] is None else f"{m['spearman']:.3f}"
        table_rows.append(f"<tr><td>{row['episode']}</td><td>{row['decision']}</td>"
            f"<td>{m['rank_mid']:.1f} / {row['candidate_count']}</td><td>{m['rank_best']}–{m['rank_worst']}</td>"
            f"<td>{m['hindsight_regret']:.3f}</td><td>{m['chosen_utility']:.3f}</td><td>{m['best_utility']:.3f}</td><td>{corr}</td></tr>")
        candidate_rows = []
        for index in np.argsort([-outcome["official_return"] for outcome in row["outcomes"]], kind="stable"):
            item = row["outcomes"][index]
            chosen = " ← chosen" if int(index) == m["selected_candidate"] else ""
            candidate_rows.append(f"<tr><td>{index}{chosen}</td><td>{row['predicted_costs'][index]:.3f}</td>"
                f"<td>{item['official_return']:.3f}</td><td>{item['offroad_frames']}</td></tr>")
        details.append(f"<details><summary>Episode {row['episode']}, decision {row['decision']}: all candidates</summary>"
            "<table><tr><th>Candidate</th><th>Predicted cost ↓</th><th>Actual return ↑</th><th>Off-road frames</th></tr>"
            + "".join(candidate_rows) + "</table></details>")
    refresh = '<meta http-equiv="refresh" content="5">' if not complete else ""
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">{refresh}
<title>User-only candidate-ranking monitor</title><style>
body{{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#17212b;background:#f7f8fa}}
table{{border-collapse:collapse;background:white;width:100%;margin:20px 0}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}
.notice{{padding:18px;background:#e5edf7;border-radius:8px}}details{{margin:18px 0}}small{{color:#465563}}
</style><h1>Candidate-ranking monitor</h1><p class="notice">User-only diagnostic. No result is fed to the driver, training, checkpoint selection, or assistant summaries.</p>
<p>Status: {'complete' if complete else 'running — refreshes every five seconds'}. Audited states: {len(rows)}.</p>
<p>Lower rank/regret is better. Tied candidates receive a rank interval and midpoint. Actual utility is official reward over the frozen action sequence, with the original episode limit.</p>
<table><tr><th>Episode</th><th>Decision</th><th>Midrank ↓</th><th>Tie range</th><th>Hindsight regret ↓</th><th>Chosen return</th><th>Best return</th><th>Spearman ↑</th></tr>{''.join(table_rows)}</table>
{''.join(details)}<p><small>{html.escape(' '.join(payload['caveats']))}</small></p></html>"""
    temporary = directory / "index.pending"
    temporary.write_text(page)
    temporary.replace(directory / "index.html")


def worker(queue: Path, output: Path, status: Path) -> None:
    seen: set[str] = set()
    rows = []
    costs = {"prefix_native_frames": 0, "candidate_native_frames": 0, "resets": 0}
    started = last_job = time.monotonic()
    private_report(output, rows, False)
    try:
        while True:
            pending = [path for path in sorted(queue.glob("*.npz")) if path.name not in seen]
            for path in pending:
                result, work = replay_request(path)
                rows.append(result)
                seen.add(path.name)
                last_job = time.monotonic()
                for key, value in work.items():
                    costs[key] += value
                private_report(output, rows, False)
                write_json(status, {"state": "running", "completed_requests": len(seen), **costs})
            if (queue / "CLOSED").exists() and not pending:
                break
            if time.monotonic() - last_job > 180:
                raise TimeoutError("No new request or close signal within the monitor idle limit")
            time.sleep(.25)
        private_report(output, rows, True)
        write_json(status, {"state": "complete", "completed_requests": len(seen), **costs,
                            "wall_time_s": time.monotonic() - started,
                            "ranking_values": "private user report only"})
    except Exception as error:
        # Technical status contains no outcomes, ranks, returns or regrets.
        write_json(status, {"state": "failed", "error_type": type(error).__name__,
                            "technical_message": str(error), "completed_requests": len(seen), **costs})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    worker(args.queue, args.output, args.status)
