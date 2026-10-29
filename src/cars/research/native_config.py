"""Resolved native recipe; source defaults remain distinct from local adaptations."""
from dataclasses import asdict, dataclass
import json

from cars.research.driver_config import DriverTrainingConfig


@dataclass(frozen=True)
class NativeTrainingConfig:
    seed: int = 0
    workers: int = 8
    vector_backend: str = "dummy"
    device: str = "cpu"
    torch_threads: int = 1
    total_decisions: int = 4_000_000
    checkpoints: tuple[int, ...] = (500_000, 1_000_000, 2_000_000, 4_000_000)
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
    log_std_init: float = -2.
    reward_clip: float = 10.

    def __post_init__(self) -> None:
        # Share validation, not the local encoder or its frame-budget defaults.
        values = asdict(self)
        values["total_native_frames"] = values.pop("total_decisions")
        DriverTrainingConfig(**values)

    def resolved(self) -> dict:
        return json.loads(json.dumps(asdict(self)))
