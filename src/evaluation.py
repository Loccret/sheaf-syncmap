"""DBSCAN evaluation and ground-truth scale traces."""
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score

def _history_window_indices(
    steps: np.ndarray,
    total_steps: int,
    start_step: int,
    end_step: int,
) -> np.ndarray:
    """Return explicit frame indices for a relative evaluation window.

    Args:
        steps: One-based frame steps.
        total_steps: Number of processed training states.
        start_step: Inclusive relative start offset.
        end_step: Exclusive relative end offset.

    Returns:
        Selected frame indices, or all indices when the interval is empty.
    """

    start = total_steps + start_step
    end = total_steps + end_step
    indices = np.flatnonzero((steps >= start) & (steps < end))
    if indices.size == 0:
        return np.arange(steps.shape[0], dtype=np.int64)
    return indices

def evaluate_history(
    history: np.ndarray,
    steps: np.ndarray,
    total_steps: int,
    labels: np.ndarray,
    eps_values: np.ndarray,
    min_samples: int,
    history_window_start_step: int,
    history_window_end_step: int,
) -> dict[str, Any]:
    """Select DBSCAN epsilon by mean NMI in the final evaluation window."""

    if history.shape[0] != steps.shape[0]:
        raise ValueError("SyncMap history and explicit steps must have the same length")
    window_indices = _history_window_indices(
        steps,
        total_steps=total_steps,
        start_step=history_window_start_step,
        end_step=history_window_end_step,
    )
    window = history[window_indices]
    labels = labels - labels.min()
    result: dict[str, Any] = {"history_frames": int(window.shape[0])}
    mean_nmi_by_eps: dict[float, float] = {}
    for eps in eps_values:
        nmis = []
        for frame in window:
            preds = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(frame)
            nmis.append(normalized_mutual_info_score(labels, preds))
        mean_nmi_by_eps[float(eps)] = float(np.mean(nmis))
    best_eps, best_nmi = max(mean_nmi_by_eps.items(), key=lambda item: item[1])
    result.update(
        {
            "best_eps": float(best_eps),
            "best_nmi": float(best_nmi),
            "mean_nmi_by_eps": {
                str(key): value for key, value in mean_nmi_by_eps.items()
            },
        }
    )
    return result

def nmi_by_frame(
    history: np.ndarray,
    labels: np.ndarray,
    best_eps: float,
    min_samples: int,
) -> np.ndarray:
    """Compute DBSCAN NMI for each exported SyncMap frame."""

    normalized_labels = labels - labels.min()
    scores = [
        normalized_mutual_info_score(
            normalized_labels,
            DBSCAN(eps=float(best_eps), min_samples=int(min_samples)).fit_predict(frame),
        )
        for frame in history
    ]
    return np.asarray(scores, dtype=np.float32)

def ground_truth_chunk_mean_pairwise_distance_by_frame(
    history: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean pairwise distances for nonsingleton ground-truth labels.

    Args:
        history: Coordinate history with shape ``frames x nodes x dimensions``.
        labels: Integer ground-truth label for every node.

    Returns:
        Sorted nonsingleton labels as ``int64`` and aligned ordinary mean
        pairwise distances as ``float32``.

    Raises:
        ValueError: When history or labels do not form a finite aligned history.
    """

    coordinates = np.asarray(history, dtype=np.float64)
    label_array = np.asarray(labels)
    if coordinates.ndim != 3 or not np.isfinite(coordinates).all():
        raise ValueError(
            "Ground-truth mean pairwise distance requires a finite "
            "frames x nodes x dimensions history"
        )
    if label_array.ndim != 1 or label_array.shape[0] != coordinates.shape[1]:
        raise ValueError("Ground-truth labels must align with history nodes")
    if not np.issubdtype(label_array.dtype, np.integer):
        raise ValueError("Ground-truth labels must be integers")

    unique_labels, counts = np.unique(label_array, return_counts=True)
    group_labels = unique_labels[counts > 1].astype(np.int64, copy=False)
    distances = np.empty(
        (coordinates.shape[0], group_labels.shape[0]),
        dtype=np.float32,
    )
    for group_index, group_label in enumerate(group_labels):
        group_coordinates = coordinates[:, label_array == group_label, :]
        first, second = np.triu_indices(group_coordinates.shape[1], k=1)
        differences = (
            group_coordinates[:, first, :] - group_coordinates[:, second, :]
        )
        pairwise_distances = np.sqrt(np.sum(differences * differences, axis=2))
        distances[:, group_index] = pairwise_distances.mean(axis=1).astype(np.float32)
    return group_labels, distances
