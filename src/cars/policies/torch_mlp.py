"""Deterministic state MLP with bounded actions and checkpointed normalization."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cars.env.state import Array, CarState, OBS_SCHEMA, OBS_VERSION, TrackInfo


@dataclass(frozen=True)
class MLPConfig:
    hidden_sizes: tuple[int, ...] = (256, 256)
    normalization_floor: float = 0.01
    clip_obs: float = 10.0


class StateMLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        sizes = (len(OBS_SCHEMA), *config.hidden_sizes, 3)
        layers: list[nn.Module] = []
        for i, (left, right) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(left, right))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.register_buffer("obs_mean", torch.zeros(len(OBS_SCHEMA)))
        self.register_buffer("obs_std", torch.ones(len(OBS_SCHEMA)))
        self.clip_obs = config.clip_obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.clamp((obs - self.obs_mean) / self.obs_std, -self.clip_obs, self.clip_obs)
        x = torch.tanh(self.layers(x))
        return torch.cat((x[..., :1], (x[..., 1:] + 1) / 2), dim=-1)


class MLPPolicy:
    def __init__(self, config: MLPConfig | None = None, *, method: str = "il_bc",
                 training_seed: int = 0):
        if method not in ("il_bc", "il_dagger"):
            raise ValueError(method)
        self.config = config or MLPConfig()
        self.method, self.training_seed = method, training_seed
        self.network = StateMLP(self.config).cpu().eval()
        self.path: Path | None = None

    def reset(self, track: TrackInfo) -> None:
        pass

    def act(self, state: CarState, obs: Array) -> Array:
        with torch.inference_mode():
            return self.network(torch.as_tensor(obs, dtype=torch.float32)).cpu().numpy().copy()

    def save(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        torch.save({"config": asdict(self.config), "method": self.method,
                    "training_seed": self.training_seed, "obs_version": OBS_VERSION,
                    "obs_schema": OBS_SCHEMA, "state_dict": self.network.state_dict()}, self.path)

    @classmethod
    def load(cls, path: str | Path) -> "MLPPolicy":
        path = Path(path).resolve()
        if path.is_dir():
            path = path / "model.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["obs_version"] != OBS_VERSION or saved["obs_schema"] != OBS_SCHEMA:
            raise ValueError("Incompatible StateObs schema")
        saved["config"]["hidden_sizes"] = tuple(saved["config"]["hidden_sizes"])
        policy = cls(MLPConfig(**saved["config"]), method=saved["method"], training_seed=saved["training_seed"])
        policy.network.load_state_dict(saved["state_dict"])
        policy.path = path
        return policy

    def configuration(self) -> dict:
        from cars.training import sha256

        return {"method": self.method, "training_seed": self.training_seed, **asdict(self.config),
                "artifact": str(self.path) if self.path else None,
                "sha256": sha256(self.path) if self.path else None,
                "training_run": str(self.path.parent) if self.path else None}
