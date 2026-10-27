"""One-way export of frozen candidate plans to a separate diagnostic process."""
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

import numpy as np

from cars.env.protocol import ProtocolConfig
from cars.training import write_json


class UserMonitor:
    """Writes requests; never reads rankings, oracle returns or chosen alternatives."""

    def __init__(self, run_id: str, source: Path, protocol: ProtocolConfig):
        self.root = Path("runs/user_monitor") / run_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.queue = self.root / "queue"
        self.queue.mkdir()
        self.private = Path("private_diagnostics") / run_id
        if self.private.exists():
            raise FileExistsError("User-only report already exists")
        self.protocol = protocol
        write_json(self.root / "config.json", {"kind": "user_only_privileged_monitor",
            "source_evaluation": str(source.resolve()), "protocol": asdict(protocol),
            "partition": "development", "episodes": [0, 6, 12, 18], "decisions": [25, 100, 175],
            "selection_rule": "fixed sparse schedule; frozen candidates and selected action",
            "oracle": "official simulator reset and exact observed-prefix replay",
            "utility": "official cumulative reward over the frozen candidate horizon",
            "feedback": "none; user-only values; excluded from all agent/training/selection inputs",
            "private_output": str(self.private.resolve())})
        self.log = (self.root / "runtime.log").open("w")
        self.process = subprocess.Popen([sys.executable, "-m", "cars.monitoring.oracle",
            "--queue", str(self.queue.resolve()), "--output", str(self.private.resolve()),
            "--status", str((self.root / "status.json").resolve())], stdout=self.log, stderr=self.log)
        self.requests = 0

    def capture(self, episode: int, decision: int, *, track_seed: int, prefix: list,
                prefix_frames: list, observation: dict, candidates: np.ndarray,
                predicted_costs: np.ndarray, selected: int) -> None:
        if episode not in (0, 6, 12, 18) or decision not in (25, 100, 175):
            return
        filename = self.queue / f"episode_{episode:03d}_decision_{decision:04d}.npz"
        temporary = filename.with_suffix(".pending")
        with temporary.open("wb") as file:
            np.savez_compressed(file, episode=episode, decision=decision, track_seed=track_seed,
                prefix=np.asarray(prefix, dtype=np.float32).reshape(-1, 3),
                prefix_frames=np.asarray(prefix_frames, dtype=np.int64),
                expected_ego=observation["ego"], candidates=candidates,
                predicted_costs=predicted_costs, selected=selected,
                action_repeat=self.protocol.action_repeat)
        temporary.replace(filename)
        self.requests += 1

    def close(self) -> None:
        (self.queue / "CLOSED").touch()
        return_code = self.process.wait()
        self.log.close()
        if return_code:
            raise RuntimeError("User monitor failed; see its public technical status, not private results")
