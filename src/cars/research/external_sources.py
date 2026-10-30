"""Hash-verified, separately staged official sources; no third-party imports here."""
from pathlib import Path
import hashlib
import json
import shutil


REVISIONS = {"MRQ": "073adcd6b30f84694c241e00bd684db50242a215",
             "MRSQ": "b12abaf309833f28619dfd87f4251074abed3ae3"}
FILES = {"MRQ": ("LICENSE", "MRQ/MRQ.py", "MRQ/models.py", "MRQ/buffer.py", "MRQ/utils.py"),
         "MRSQ": ("LICENSE", "MRSQ/MRSQ.py", "MRSQ/buffer.py", "MRSQ/utils.py")}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replay_source_patch(algorithm: str, source: bytes) -> tuple[bytes, dict]:
    """Explicit allocation and defined unused priorities; preserve valid priority rules."""
    if algorithm == "MRQ":
        changes = [("cpu_observation_storage", "if self.device == torch.device('cuda'):",
            "if False:  # cars adapter: explicit CPU replay storage"),
            ("initialize_unused_priorities", "self.priority = torch.empty(self.max_size, device=self.device) if self.prioritized else []",
             "self.priority = torch.zeros(self.max_size, device=self.device) if self.prioritized else []")]
    elif algorithm == "MRSQ":
        changes = [("cpu_observation_storage",
            "self.storage_device = self.device if check_memory(max_size, obs_shape, 1 if pixel_obs else 4, action_dim) else torch.device('cpu')",
            "self.storage_device = torch.device('cpu')  # cars adapter: explicit CPU replay storage")]
    else:
        raise ValueError("Unknown pinned algorithm")
    text = source.decode("utf-8")
    for _, before, after in changes:
        if text.count(before) != 1:
            raise ValueError("Each exact pinned replay patch must match once")
        text = text.replace(before, after)
    patched = text.encode("utf-8")
    return patched, {"kind": "explicit_replay_adaptation", "changes": [
        {"kind": kind, "before": before, "after": after} for kind, before, after in changes],
        "original_sha256": sha256(source), "adapted_sha256": sha256(patched),
        "learner_hyperparameters_or_valid_priority_rule_changed": False,
        "uninitialized_source_sampling_parity": "not claimed; unused slots explicitly defined"}


def stage_sources(algorithm: str, cache: Path, audit: Path, destination: Path) -> dict:
    """Validate every input before creating an immutable runnable source directory.

    The destination contains no environment runner or benchmark result files.
    Run agents in isolated processes to avoid upstream bare-import collisions.
    """
    if algorithm not in REVISIONS:
        raise ValueError("Only MRQ and MRSQ are supported")
    record = json.loads(audit.read_text())
    revision = REVISIONS[algorithm]
    if record["repositories"][algorithm]["commit"] != revision:
        raise ValueError("Wrong official revision in source audit")
    entries = [e for e in record["files"] if e["repo"] == algorithm]
    if len({e["path"] for e in entries}) != len(entries):
        raise ValueError("Duplicate source records")
    indexed = {e["path"]: e for e in entries}
    prepared = []
    for name in FILES[algorithm]:
        entry = indexed[name]
        path = cache / algorithm / name
        original = path.read_bytes()
        if entry["commit"] != revision or sha256(original) != entry["sha256"]:
            raise ValueError(f"Pinned source mismatch: {name}")
        adapted, patch = replay_source_patch(algorithm, original) if name.endswith("/buffer.py") else (original, None)
        prepared.append((Path(name).name, adapted, {"upstream_path": name,
            "url": entry["url"], "upstream_sha256": sha256(original),
            "adapted_sha256": sha256(adapted), "patch": patch}))
    manifest = {"algorithm": algorithm, "revision": revision, "license": "CC-BY-NC-4.0",
        "audit_sha256": sha256(audit.read_bytes()), "source_files": [v[2] for v in prepared],
        "observation_storage_device": "cpu", "algorithm_defaults_changed": False,
        "third_party_code_executed": False,
        "runtime_status": "staged_only; external harness/runtime checks still required"}
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name, content, _ in prepared:
            (destination / name).write_bytes(content)
        (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(destination)
        raise
    return manifest
