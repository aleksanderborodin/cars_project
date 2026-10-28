"""Same sensor features with deterministic adaptive-average pooling on CUDA."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from cars.research.recurrent import SensorEncoder


class DeterministicAveragePool(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        # The primary 61x61 local input produces 9x9 conv features. Its adaptive
        # four-bin intervals are exactly 3x3 windows with stride two.
        if (height, width) == (9, 9):
            return F.avg_pool2d(value, kernel_size=3, stride=2)
        return torch.stack([torch.stack([
            value[..., math.floor(i * height / 4):math.ceil((i + 1) * height / 4),
                  math.floor(j * width / 4):math.ceil((j + 1) * width / 4)].mean(dim=(-2, -1))
            for j in range(4)], dim=-1) for i in range(4)], dim=-2)


class DriverSensorEncoder(SensorEncoder):
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.image[4] = DeterministicAveragePool()
