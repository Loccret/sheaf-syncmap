"""Original SyncMap model."""
from __future__ import annotations
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from src.models.training_observer import StepObserver, TrainingStepObservation

class StandardSyncMap:
    """Original SyncMap update rule with optional step observations."""

    def __init__(
        self,
        input_size: int,
        dimensions: int,
        adaptation_rate: float,
        use_tqdm: bool = False,
        fix_seed: bool = True,
        space_size: float = 10.0,
    ) -> None:
        self.input_size = input_size
        self.dimensions = dimensions
        self.adaptation_rate = adaptation_rate
        self.use_tqdm = use_tqdm
        self.space_size = space_size
        if fix_seed:
            np.random.seed(42)
        self.syncmap = np.random.rand(input_size, dimensions)
        self.total_activation = np.zeros(input_size)

    def fit(self, input_sequence: np.ndarray, observer: StepObserver | None = None) -> None:
        """Process a sequence and emit requested post-state coordinates."""

        plus = input_sequence > 0.1
        minus = ~plus
        iterator = enumerate(plus)
        if self.use_tqdm:
            iterator = tqdm(iterator, total=len(input_sequence))

        for idx, vplus in iterator:
            vminus = minus[idx]
            plus_mass = vplus.sum()
            minus_mass = vminus.sum()
            self.total_activation += vplus.astype(int)
            if plus_mass > 1 and minus_mass > 1:
                center_plus = np.dot(vplus, self.syncmap) / plus_mass
                center_minus = np.dot(vminus, self.syncmap) / minus_mass
                dist_plus = distance.cdist(center_plus[None, :], self.syncmap, "euclidean").T
                dist_minus = distance.cdist(center_minus[None, :], self.syncmap, "euclidean").T

                update_plus = vplus[:, np.newaxis] * ((center_plus - self.syncmap) / dist_plus)
                update_minus = vminus[:, np.newaxis] * ((center_minus - self.syncmap) / dist_minus)

                self.syncmap += self.adaptation_rate * (update_plus - update_minus)
                maximum = self.syncmap.max()
                self.syncmap = self.space_size * self.syncmap / maximum
            step = idx + 1
            if observer is not None and observer.wants_step(step):
                observer.record(TrainingStepObservation(step=step, coordinates=self.syncmap))
