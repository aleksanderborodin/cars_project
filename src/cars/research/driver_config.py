"""Resolved recurrent-driver training choices and a native-frame learning schedule."""
from dataclasses import asdict, dataclass
import json
import math


@dataclass(frozen=True)
class DriverTrainingConfig:
    seed: int = 0
    workers: int = 4
    vector_backend: str = "subproc"
    device: str = "cpu"
    torch_threads: int = 1
    total_native_frames: int = 8_000_000
    checkpoints: tuple[int, ...] = (200_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000)
    n_steps: int = 512
    batch_size: int = 128
    n_epochs: int = 10
    learning_rate: float = 1e-4
    gamma: float = .99
    gae_lambda: float = .95
    ent_coef: float = 0.
    vf_coef: float = .5
    max_grad_norm: float = .5
    clip_range: float = .2
    use_sde: bool = True
    sde_sample_freq: int = 4
    lstm_hidden_size: int = 128
    features_dim: int = 128
    log_std_init: float = -2.
    reward_clip: float = 10.
    target_kl: float | None = None

    def __post_init__(self) -> None:
        if min(self.workers, self.torch_threads, self.total_native_frames, self.n_steps, self.batch_size, self.n_epochs) < 1:
            raise ValueError("Positive training dimensions required")
        if self.batch_size > self.workers * self.n_steps or self.batch_size < 2:
            raise ValueError("Batch size must fit a complete rollout and exceed one")
        if self.vector_backend not in ("dummy", "subproc") or self.device not in ("cpu", "cuda"):
            raise ValueError("Explicit supported device/vector backend required")
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1 or self.learning_rate <= 0:
            raise ValueError("Invalid discount/schedule")
        if any(not 0 < frame <= self.total_native_frames for frame in self.checkpoints):
            raise ValueError("Checkpoint outside declared budget")
        if self.target_kl is not None and (not math.isfinite(self.target_kl) or self.target_kl <= 0):
            raise ValueError("target_kl must be finite and positive, or None")

    def resolved(self) -> dict:
        return json.loads(json.dumps(asdict(self)))


@dataclass
class NativeSchedule:
    initial: float
    total: int
    frames: int = 0

    def __call__(self, progress_remaining: float) -> float:
        # SB3's chunk-local progress is deliberately ignored. This clock is
        # updated with actual native frames and restored across fresh episodes.
        return self.initial * max(0., 1. - self.frames / self.total)
