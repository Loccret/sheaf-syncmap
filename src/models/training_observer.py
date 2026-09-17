"""Neutral observations emitted by SyncMap training loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from src.regularizers.radial_correction import RadialCorrectionResult


Coordinates = np.ndarray | torch.Tensor


@dataclass(frozen=True)
class RadialStepObservation:
    """Radial correction inputs and result for one processed state."""

    coordinates_before: torch.Tensor
    proposal: torch.Tensor
    result: RadialCorrectionResult


@dataclass(frozen=True)
class TrainingStepObservation:
    """Coordinates and optional radial evidence after one processed state."""

    step: int
    coordinates: Coordinates
    radial: RadialStepObservation | None = None


class StepObserver(Protocol):
    """Consumer of selected model-step observations."""

    def wants_step(self, step: int) -> bool:
        """Return whether the one-based processed-state step is requested."""

    def record(self, observation: TrainingStepObservation) -> None:
        """Consume one requested training-step observation."""
