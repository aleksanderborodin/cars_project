"""Validation/planning RNG scopes; initialize any CUDA use before entering."""
from contextlib import contextmanager
from collections.abc import Iterator
import random

import numpy as np
import torch


@contextmanager
def isolated_rng(seed: int) -> Iterator[None]:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("An explicit uint32 random-stream seed is required")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cuda_initialized = torch.cuda.is_initialized()
    devices = list(range(torch.cuda.device_count())) if cuda_initialized else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if not cuda_initialized and torch.cuda.is_initialized():
            raise RuntimeError("Initialize CUDA before entering an isolated RNG scope")
