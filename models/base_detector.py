from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class Detection:
    cls_name: str
    conf: float
    box: tuple  # x1, y1, x2, y2


class BaseDetector(ABC):
    name = "base"

    @abstractmethod
    def load(self, weights: str | None = None): ...

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Detection]: ...

    @abstractmethod
    def train(self, data, epochs: int = 50, **kw): ...

    def benchmark_latency(self, frame: np.ndarray, runs: int = 20) -> float:
        """Mean inference latency in ms (after 3 warm-up runs)."""
        for _ in range(3):
            self.predict(frame)
        t = time.perf_counter()
        for _ in range(runs):
            self.predict(frame)
        return (time.perf_counter() - t) / runs * 1000
