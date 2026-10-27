"""ctypes ABI v1 adapter. Only raw CarState is passed to Rust candidates."""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from cars.env.state import Array, CarState, TrackInfo

ABI_VERSION = 1
FloatPointer = ct.POINTER(ct.c_float)


def build_rust(crate_dir: str | Path, target_dir: str | Path | None = None) -> Path:
    crate = Path(crate_dir).resolve()
    target = Path(target_dir).resolve() if target_dir else Path(tempfile.mkdtemp(prefix="carctl-build-"))
    cargo = Path.home() / ".cargo/bin/cargo"
    subprocess.run([str(cargo), "build", "--release", "--manifest-path", str(crate / "Cargo.toml"),
                    "--target-dir", str(target)], check=True, env={**os.environ})
    library = target / "release/libcarctl.so"
    if not library.is_file():
        raise FileNotFoundError(f"Crate must build libcarctl.so: {library}")
    return library


class RustPolicy:
    def __init__(self, path: str | Path):
        self.path = str(Path(path).resolve())
        self._lib = None
        self._handle = None

    def configuration(self) -> dict:
        return {"path": self.path, "sha256": hashlib.sha256(Path(self.path).read_bytes()).hexdigest(),
                "abi_version": ABI_VERSION}

    def _load(self) -> None:
        lib = ct.CDLL(self.path)
        lib.carctl_abi_version.argtypes = []
        lib.carctl_abi_version.restype = ct.c_uint32
        if lib.carctl_abi_version() != ABI_VERSION:
            raise ValueError("Rust controller ABI version mismatch")
        lib.carctl_new.argtypes = []
        lib.carctl_new.restype = ct.c_void_p
        lib.carctl_free.argtypes = [ct.c_void_p]
        lib.carctl_free.restype = None
        lib.carctl_reset.argtypes = [ct.c_void_p, FloatPointer, ct.c_uint32, ct.c_uint32]
        lib.carctl_reset.restype = None
        lib.carctl_act.argtypes = [ct.c_void_p, FloatPointer, ct.c_uint32, FloatPointer]
        lib.carctl_act.restype = None
        self._lib = lib
        self._handle = lib.carctl_new()
        if not self._handle:
            raise MemoryError("carctl_new returned null")

    def reset(self, track: TrackInfo) -> None:
        if self._lib is None:
            self._load()
        data = track.as_array()
        self._lib.carctl_reset(self._handle, data.ctypes.data_as(FloatPointer), len(data), data.shape[1])

    def act(self, state: CarState, obs: Array) -> Array:
        if self._lib is None or self._handle is None:
            raise RuntimeError("Call reset before act")
        data = state.as_array()
        out = np.zeros(3, dtype=np.float32)
        self._lib.carctl_act(self._handle, data.ctypes.data_as(FloatPointer), len(data), out.ctypes.data_as(FloatPointer))
        if not np.isfinite(out).all():
            raise ValueError("Rust candidate returned a non-finite action")
        return np.clip(out, [-1, 0, 0], [1, 1, 1]).astype(np.float32)

    def close(self) -> None:
        if self._handle is not None:
            self._lib.carctl_free(self._handle)
            self._handle = None
            self._lib = None

    def __getstate__(self) -> dict:
        # Spawn processes load their own handles and shared libraries.
        return {"path": self.path, "_lib": None, "_handle": None}

    def __del__(self):
        self.close()
