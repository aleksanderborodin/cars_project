"""Audit completed native training, partition/trace integrity and descriptive curves."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO

from cars.env.partitions import Partitions
from cars.training import sha256, write_json


def parameter_hash(path: Path) -> str:
    model = RecurrentPPO.load(path / "model.zip", device="cpu")
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.campaign / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Only a completed campaign may be summarized")
    spec = json.loads((args.campaign / "config.json").read_text())
    selected = json.loads((args.campaign / "selection.json").read_text())
    rows, tables, learning, artifacts = [], [], [], {}
    sources = None
    verified = traces = 0
    previous_hash = None
    train_seeds: set[int] = set()
    previous_streams = [set() for _ in range(spec["training"]["workers"])]
    prior_frames = 0
    reward_error = 0.
    rng = np.random.default_rng(20261029)
    draws = rng.integers(0, 50, size=(10000, 50))
    for stage in status["completed"]:
        train = Path("runs/native_training") / stage["training_run_id"]
        evaluation = Path("runs/native_evaluation") / stage["evaluation_run_id"]
        for root in (train, evaluation):
            config = json.loads((root / "config.json").read_text())
            assert config["partition_sha256"] == spec["partition_sha256"]
            assert config["protocol"]["view"] == "native_recipe"
            assert sources is None or sources == config["source_sha256"]
            sources = config["source_sha256"]
            for name, expected in sources.items():
                assert sha256(root / "source" / name) == expected
                verified += 1
            artifacts[str(root)] = {str(p.relative_to(root)): sha256(p)
                for p in root.rglob("*") if p.is_file() and "source" not in p.relative_to(root).parts
                and "checkpoints" not in p.relative_to(root).parts}
        tc = json.loads((train / "config.json").read_text())
        ec = json.loads((evaluation / "config.json").read_text())
        assert tc["settings"]["training"] == spec["training"]
        for name, expected in ec["settings"]["artifact_sha256"].items():
            assert sha256(train / name) == expected
        initial = json.loads((train / "initial_policy.json").read_text())
        if previous_hash:
            assert initial["parameters_sha256"] == previous_hash
        previous_hash = parameter_hash(train)
        state = json.loads((train / "training_state.json").read_text())
        step = state["decisions"]
        assert state["native_frames"] == step
        assert state["saved_after_optimizer_update"]
        assert np.isclose(state["reward_normalization_count"], step + .0001)
        assert state["learning_rate"] == 1e-4 * max(0, 1 - step / 4_000_000)
        assert stage["training_cost"]["native_frames"] == step - prior_frames
        current = pd.DataFrame([json.loads(x) for x in (train / "learning.jsonl").read_text().splitlines()])
        assert np.isfinite(current.to_numpy()).all()
        assert current.decisions.tolist() == list(range(prior_frames + 4096, step + 1, 4096))
        assert current.optimizer_steps_cumulative.iloc[-1] == step // 4096 * 320
        assert state["updates"] == step // 4096 * 10  # entered epochs, not rollouts
        prior_frames = step
        learning.append(current)
        tracks = json.loads((train / "training_tracks.json").read_text())
        streams = json.loads((train / "streams.json").read_text())
        for worker, (track_list, stream) in enumerate(zip(tracks, streams, strict=True)):
            assert len(set(track_list)) == len(track_list)
            assert all(Partitions().contains("train", value) for value in track_list)
            assert train_seeds.isdisjoint(track_list)
            train_seeds.update(track_list)
            assert set(stream["used"]) == previous_streams[worker] | set(track_list)
            previous_streams[worker] = set(stream["used"])
        table = pd.read_csv(evaluation / "episodes.csv")
        assert table.track_seed.tolist() == Partitions().seeds("development", 50)
        assert table.episode.tolist() == list(range(50))
        summary = json.loads((evaluation / "summary.json").read_text())
        assert summary["normalization_frozen"]
        assert np.isclose(table["return"].mean(), summary["mean_return"])
        assert table.success.mean() == summary["success_rate"]
        assert table.steps.sum() == stage["evaluation_cost"]["native_frames"]
        for index, row in table.iterrows():
            with np.load(evaluation / f"episodes/{index:04d}.npz") as trace:
                n = int(row["steps"])
                assert trace["observation"].shape == (n, 2, 64, 64)
                assert trace["observation"].dtype == np.uint8
                assert not trace["observation"][0, 0].any()
                np.testing.assert_array_equal(trace["observation"][1:, 0], trace["observation"][:-1, 1])
                np.testing.assert_array_equal(trace["terminal_observation"][0], trace["observation"][-1, 1])
                action = trace["action"]
                assert action.shape == (n, 3) and np.isfinite(action).all()
                assert np.all(action >= [-1, 0, 0]) and np.all(action <= 1)
                error = abs(float(trace["reward"].astype(np.float64).sum()) - row["return"])
                reward_error = max(reward_error, error)
                assert error < .001  # float32 trace versus float64 official cumulative metric
                traces += 1
        returns = table["return"].to_numpy()
        ci = np.quantile(returns[draws].mean(axis=1), [.025, .975]).tolist()
        keys = ["train/approx_kl", "train/clip_fraction", "train/std", "train/explained_variance"]
        rows.append({"requested_decisions": stage["stop_at"], "actual_decisions": step,
            "training_run_id": stage["training_run_id"], "evaluation_run_id": stage["evaluation_run_id"],
            "mean_return": float(returns.mean()), "return_track_bootstrap95": ci,
            "completions": int(table.success.sum()), "crashes": int(table.crash.sum()),
            "mean_progress": float(table.progress.mean()), "mean_offroad_steps": float(table.off_track_steps.mean()),
            "learning_segment": current[keys].agg(["min", "median", "max"]).to_dict(),
            "learning_last": current.iloc[-1].to_dict(), "training_state": state})
        tables.append(table)
    best = max(rows, key=lambda r: (r["mean_return"], r["completions"], -r["actual_decisions"]))
    assert best["training_run_id"] == selected["training_run_id"]
    contrasts = []
    for i in range(1, len(rows)):
        delta = tables[i]["return"].to_numpy() - tables[i - 1]["return"].to_numpy()
        contrasts.append({"contrast": f"{rows[i]['requested_decisions']}-{rows[i-1]['requested_decisions']}",
            "paired_mean_return_change": float(delta.mean()),
            "track_bootstrap95": np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()})
    result = {"campaign": str(args.campaign), "selection": selected, "checkpoints": rows,
        "sequential_contrasts": contrasts,
        "uncertainty": "10,000 paired track bootstrap draws, seed 20261029; descriptive development-only intervals, not training-seed uncertainty; selection bias not corrected",
        "audit": {"source_files_verified": verified, "all_source_snapshots_identical": True,
            "trace_episodes_checked": traces, "max_float32_trace_return_error": reward_error,
            "resume_parameter_hashes_match": True, "optimizer_clock_normalizer_stream_checks_pass": True,
            "unique_training_tracks": len(train_seeds), "train_dev_disjoint": True},
        "cost": {"training_native_frames": sum(r["training_cost"]["native_frames"] for r in status["completed"]),
            "evaluation_native_frames": sum(r["evaluation_cost"]["native_frames"] for r in status["completed"]),
            "reload_native_frames": sum(r["training_cost"]["reload_check"]["native_frames"] for r in status["completed"]),
            "reset_physics_ticks": sum(r["training_cost"]["training_reset_physics_ticks"] +
                r["training_cost"]["reload_check"]["reset_physics_ticks"] + r["evaluation_cost"]["reset_physics_ticks"] for r in status["completed"]),
            "rollout_updates": sum(r["training_cost"]["rollout_updates"] for r in status["completed"]),
            "wall_seconds": status["wall_seconds"], "internal_search_physics_steps": 0,
            "note": "Profile, readiness, unit tests and later diagnostics are additional costs"}}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "summary.json", result)
    write_json(args.output / "provenance.json", {"runs": artifacts,
        "campaign": {str(p.relative_to(args.campaign)): sha256(p) for p in args.campaign.iterdir() if p.is_file()},
        "analysis_script_sha256": sha256(Path(__file__))})
    frame = pd.concat(learning, ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    x = np.array([row["actual_decisions"] for row in rows]) / 1e6
    mean = np.array([row["mean_return"] for row in rows])
    ci = np.array([row["return_track_bootstrap95"] for row in rows])
    axes[0, 0].errorbar(x, mean, yerr=np.stack((mean-ci[:, 0], ci[:, 1]-mean)), marker="o", capsize=4)
    axes[0, 0].set(title="Official return: 50 development tracks", ylabel="Mean ± descriptive track-bootstrap 95%")
    axes[0, 1].plot(x, np.array([row["completions"] for row in rows]) * 2, marker="o")
    axes[0, 1].set(title="Strict completion before frame 1000", ylabel="Completion (%)", ylim=(0, 100))
    axes[1, 0].plot(frame.decisions / 1e6, frame["train/approx_kl"], alpha=.65, linewidth=.8)
    axes[1, 0].set(title="Logged approximate KL during PPO updates", ylabel="Approximate KL")
    axes[1, 1].plot(frame.decisions / 1e6, frame["train/std"], linewidth=1)
    axes[1, 1].set(title="Logged learned SDE scale", ylabel="SB3 train/std (not marginal action SD)")
    for ax in axes.flat:
        ax.set_xlabel("Cumulative native decisions (millions)")
        ax.grid(alpha=.2)
    fig.suptitle("Native CNN-LSTM PPO: one training seed; development selection, not held-out evidence")
    fig.savefig(args.output / "learning_curve.png", dpi=180)
    fig.savefig(args.output / "learning_curve.svg")
    plt.close(fig)
    print(json.dumps({"selection": selected["training_run_id"], "audit": result["audit"], "cost": result["cost"]}))


if __name__ == "__main__":
    main()
