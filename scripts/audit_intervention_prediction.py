"""Compare predictions of recorded executed actions, never candidate ranks."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cars.env.protocol import ProtocolConfig
from cars.models.physics import PhysicsDynamics
from cars.planning.intervention import InterventionConfig, LOCAL_FIELDS, intervention_worlds
from cars.research.artifacts import begin_run, read_config
from cars.research.frozen_code import source_hashes
from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    if spec["partition"] != "development" or spec["private_oracle_access"] or spec["new_real_frames"] != 0:
        raise ValueError("Recorded executed development trajectories only")
    if source_hashes(Path.cwd()) != spec["source_sha256"]:
        raise ValueError("Declared source changed")
    for item in spec["required_artifacts"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Recorded source artifact changed")
    source = Path(spec["source_run"])
    config = read_config(source)
    if config["settings"]["partition"] != "development" or config["protocol"] != spec["protocol"]:
        raise ValueError("Wrong source protocol")
    protocol = ProtocolConfig(**spec["protocol"])
    root = begin_run("intervention_prediction_audit", spec["id"], protocol, spec)
    shutil.copy2(__file__, root / "audit.py")
    shutil.copy2(args.manifest, root / "declaration.json")
    observer = PhysicsDynamics(protocol.action_repeat)
    rows, observer_work, forecast_work = [], 0, 0
    try:
        for episode in range(spec["episodes"]):
            observer.reset()
            with np.load(source / f"episodes/{episode:04d}.npz") as data:
                actions, frames = data["action"], data["frames"]
                observations = {key.removeprefix("observation_"): data[key]
                                for key in data if key.startswith("observation_")}
            if max(spec["roots"]) + spec["horizon"] > len(actions):
                raise ValueError("Predeclared window does not fit; do not silently select another")
            if not (frames[:max(spec["roots"]) + spec["horizon"]] == protocol.action_repeat).all():
                raise ValueError("Partial action interval in a declared window")
            for decision in range(max(spec["roots"]) + 1):
                obs = {key: value[decision] for key, value in observations.items()}
                local, world, _, _ = intervention_worlds(obs, protocol, InterventionConfig(), 6.67)
                if decision in spec["roots"]:
                    predictions = {}
                    for continuation in ("prior", "oracle"):
                        model = PhysicsDynamics(protocol.action_repeat)
                        try:
                            model.estimate = observer.estimate.copy() if observer.estimate is not None else None
                            _, road, _, _ = intervention_worlds(obs, protocol,
                                InterventionConfig(continuation=continuation), 6.67)
                            model.begin(local, road, 1)
                            pose, ego, previous = np.zeros((1, 3)), local["ego"][None].copy(), local["previous_action"][None].copy()
                            outputs = []
                            for offset in range(spec["horizon"]):
                                action = actions[decision + offset][None]
                                pose, ego = model.advance(pose, ego, action, previous, road.query(pose[:, :2]).grip)
                                forecast_work += protocol.action_repeat
                                previous = action
                                outputs.append(ego[0].copy())
                            predictions[continuation] = np.asarray(outputs)
                        finally:
                            model.reset()
                    np.testing.assert_array_equal(predictions["prior"][0], predictions["oracle"][0])
                    for horizon in spec["reported_horizons"]:
                        actual = observations["ego"][decision + horizon]
                        for continuation, prediction in predictions.items():
                            error = prediction[horizon - 1] - actual
                            rows.append({"episode": episode, "root_decision": decision, "horizon_decisions": horizon,
                                "continuation": continuation, "forward_error_mps": float(error[0]),
                                "lateral_error_mps": float(error[1]), "yaw_error_radps": float(error[2]),
                                "velocity_squared_error": float(error[:2] @ error[:2]),
                                "yaw_squared_error": float(error[2]**2)})
                if decision == max(spec["roots"]):
                    break
                # Observer replay uses ONLY the actual prefix, one first transition
                # per observation; future forecast states are discarded.
                observer.begin(local, world, 1)
                observer.advance(np.zeros((1, 3)), local["ego"][None], actions[decision][None],
                                 local["previous_action"][None], world.query(np.zeros((1, 2))).grip)
                observer.commit(0)
                observer_work += protocol.action_repeat
        total = observer_work + forecast_work
        if total > spec["internal_physics_frame_cap"]:
            raise RuntimeError("Declared audit cap exceeded")
        table = pd.DataFrame(rows)
        summaries = []
        for (continuation, horizon), group in table.groupby(["continuation", "horizon_decisions"], sort=False):
            summaries.append({"continuation": continuation, "horizon_decisions": int(horizon),
                "windows": len(group), "velocity_vector_rmse_mps": float(np.sqrt(group.velocity_squared_error.mean())),
                "yaw_rmse_radps": float(np.sqrt(group.yaw_squared_error.mean()))})
        table.to_csv(root / "windows.csv", index=False)
        pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
        write_json(root / "result.json", {"status": "complete", "declaration_sha256": sha256(args.manifest),
            "new_environment_frames": 0, "observer_internal_frames": observer_work,
            "forecast_internal_frames": forecast_work, "total_internal_frames": total,
            "identical_one_decision_prediction_for_every_root": True,
            "forecast_action_source": "recorded actions actually executed by baseline; no candidate selection",
            "interpretation": "post-hoc fixed windows on baseline trajectories; not interactive candidate-ranking accuracy or causal utility diagnosis",
            "private_oracle_outputs_read": False})
        report = Path(spec["report_directory"])
        report.mkdir(parents=True, exist_ok=False)
        for name in ("windows.csv", "summary.csv", "result.json", "declaration.json"):
            shutil.copy2(root / name, report / name)
        print(pd.DataFrame(summaries).to_string(index=False))
        print(f"Internal physics frames: {total}; no new real frames")
    except BaseException as error:
        write_json(root / "failure.json", {"error": str(error), "observer_work": observer_work, "forecast_work": forecast_work})
        raise
    finally:
        observer.reset()


if __name__ == "__main__":
    main()
