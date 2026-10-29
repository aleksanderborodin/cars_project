"""Immutable run manifests and source snapshots for development experiments."""
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess

from cars.env.partitions import Partitions
from cars.env.protocol import ProtocolConfig
from cars.training import sha256, write_json


def begin_run(method: str, run_id: str, protocol: ProtocolConfig, settings: dict,
              runs_dir: Path = Path("runs")) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a directory name")
    root = (runs_dir / method / run_id).resolve()
    root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[3]
    snapshot = root / "source"
    shutil.copytree(repository / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(repository / name, snapshot / name)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    config = {"kind": "development_research", "method": method, "run_id": run_id,
              "protocol": asdict(protocol), "protocol_id": protocol.identifier,
              "partition_sha256": Partitions().fingerprint,
              "settings": settings, "git_commit": commit,
              "environment": {"id": "CarRacing-v3", "max_native_frames": 1000,
                              "fps": 50, "lap_complete_percent": .95,
                              "reward_physics_termination": "official_unmodified"},
              "supervision": {"local": "observed_local_geometry_and_ego", "native": "native_pixels_only",
                              "native_recipe": "native_pixels_only; resize64_gray_stack2; repeat1",
                              "global": "privileged_full_geometry; same_ego_sensors; no_true_wheel_state"}[protocol.view],
              "versions": {name: version(name) for name in ("gymnasium", "Box2D", "numpy", "scipy", "torch", "stable-baselines3", "sb3-contrib")},
              "runtime": {"platform": platform.platform(), "python": platform.python_version()},
              "source_sha256": {str(path.relative_to(snapshot)): sha256(path) for path in sorted(snapshot.rglob("*")) if path.is_file()}}
    write_json(root / "config.json", config)
    return root


def read_config(root: Path) -> dict:
    config = json.loads((root / "config.json").read_text())
    if config.get("kind") != "development_research":
        raise ValueError("Legacy or unverified artifact cannot be used as new paper data")
    if config["partition_sha256"] != Partitions().fingerprint:
        raise ValueError("Partition fingerprint mismatch")
    return config
