from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cars.env.state import CarState, TrackInfo


class Policy(Protocol):
    def reset(self, track: TrackInfo) -> None: ...
    def act(self, state: CarState, obs: NDArray[np.float32]) -> NDArray[np.float32]: ...
