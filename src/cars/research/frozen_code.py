"""Run long experiments from a copied, hash-checked source tree."""
import os
from pathlib import Path
import shutil

from cars.training import sha256


def source_hashes(repository: Path) -> dict[str, str]:
    files = [p for p in (repository / "src").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    files += [repository / "pyproject.toml", repository / "uv.lock"]
    return {str(path.relative_to(repository)): sha256(path) for path in sorted(files)}


def freeze_sources(repository: Path, destination: Path, expected: dict[str, str]) -> dict[str, str]:
    if source_hashes(repository) != expected:
        raise ValueError("Declared production source changed before campaign launch")
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(repository / "src", destination / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(repository / name, destination / name)
    if source_hashes(destination) != expected:
        raise ValueError("Frozen source copy differs from declaration")
    return {**os.environ, "PYTHONPATH": str((destination / "src").resolve()),
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
