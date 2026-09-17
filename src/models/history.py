"""CPU and CUDA coactivation histories."""
from __future__ import annotations
from collections import deque
from typing import Iterable
import numpy as np
import torch

class VariableTracker:
    """Track recent co-activations for the NodeSyncMap history term."""

    def __init__(self, vars: int = 8, max_length: int = 3) -> None:
        self.vars = vars
        self.max_length = max_length
        self.vars_tracker = [deque(maxlen=max_length + 1) for _ in range(vars)]
        self.last_vector = np.array([False for _ in range(vars)])



    def write(self, vec: np.ndarray, changed: bool | None = None) -> None:
        if changed is None:
            changed = not np.all(self.last_vector == vec)
        if changed:
            activated_idx = np.where(vec)[0]
            for idx in activated_idx:
                self.vars_tracker[idx].append(activated_idx)
        self.last_vector = vec

    def coactivation_weight(
        self,
        ele_idx: Iterable[int],
        max_past_activate: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return weighted historical co-activations without dense history expansion.

        Sum historical activation outer products with recency weights, writing
        only nonzero active pairs into the final ``vars x vars`` matrix.
        """

        history_lists = [list(self.vars_tracker[int(idx)])[-2::-1] for idx in ele_idx]
        max_len = max((len(history) for history in history_lists), default=0)
        total_weight = torch.zeros((self.vars, self.vars), device=device, dtype=dtype)
        if max_len == 0:
            return total_weight

        weights = np.linspace(
            max_past_activate,
            0,
            max_len + 2,
            dtype=np.float32,
        )[1:-1]
        grouped_histories: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
        for history in history_lists:
            for age, active_idxs in enumerate(history):
                key = (age, id(active_idxs))
                if key in grouped_histories:
                    stored_idxs, count = grouped_histories[key]
                    grouped_histories[key] = (stored_idxs, count + 1)
                else:
                    grouped_histories[key] = (active_idxs, 1)

        if not grouped_histories:
            return total_weight

        row_chunks = []
        col_chunks = []
        value_chunks = []
        for (age, _), (active_idxs, count) in grouped_histories.items():
            active_count = len(active_idxs)
            if active_count == 0:
                continue
            row_chunks.append(np.repeat(active_idxs, active_count))
            col_chunks.append(np.tile(active_idxs, active_count))
            value_chunks.append(
                np.full(
                    active_count * active_count,
                    float(weights[age]) * int(count),
                    dtype=np.float32,
                )
            )
        if not row_chunks:
            return total_weight

        rows = torch.as_tensor(np.concatenate(row_chunks), dtype=torch.long, device=device)
        cols = torch.as_tensor(np.concatenate(col_chunks), dtype=torch.long, device=device)
        values = torch.as_tensor(np.concatenate(value_chunks), dtype=dtype, device=device)
        total_weight.index_put_((rows, cols), values, accumulate=True)
        return total_weight


class TensorVariableTracker:
    """GPU-friendly active-index history for NodeSyncMap co-activation weights."""

    def __init__(self, vars: int, max_length: int, initial_capacity: int = 32) -> None:
        self.vars = vars
        self.max_length = max_length
        self.capacity = min(max(initial_capacity, 1), vars)
        self.history_indices: torch.Tensor | None = None
        self.history_counts: torch.Tensor | None = None
        self.history_lengths: torch.Tensor | None = None
        self.last_vector: torch.Tensor | None = None
        self.age_indices: torch.Tensor | None = None
        self.slot_indices: torch.Tensor | None = None

    def _ensure_state(self, device: torch.device) -> None:
        if self.history_indices is not None and self.history_indices.device == device:
            return
        self.history_indices = torch.full(
            (self.vars, self.max_length + 1, self.capacity),
            -1,
            device=device,
            dtype=torch.long,
        )
        self.history_counts = torch.zeros(
            (self.vars, self.max_length + 1),
            device=device,
            dtype=torch.long,
        )
        self.history_lengths = torch.zeros(self.vars, device=device, dtype=torch.long)
        self.last_vector = torch.zeros(self.vars, device=device, dtype=torch.bool)
        self.age_indices = torch.arange(self.max_length, device=device)
        self.slot_indices = torch.arange(self.capacity, device=device)

    def _grow_capacity(self, active_count: int) -> None:
        if active_count <= self.capacity:
            return
        if self.history_indices is None:
            self.capacity = min(active_count, self.vars)
            return

        new_capacity = min(max(active_count, self.capacity * 2), self.vars)
        grown = torch.full(
            (self.vars, self.max_length + 1, new_capacity),
            -1,
            device=self.history_indices.device,
            dtype=torch.long,
        )
        grown[:, :, : self.capacity] = self.history_indices
        self.history_indices = grown
        self.capacity = new_capacity
        self.slot_indices = torch.arange(self.capacity, device=self.history_indices.device)

    def write(self, vec: torch.Tensor, changed: bool | None = None) -> torch.Tensor:
        """Append one activation vector and return its active indices."""

        self._ensure_state(vec.device)
        if (
            self.history_indices is None
            or self.history_counts is None
            or self.history_lengths is None
            or self.last_vector is None
        ):
            raise RuntimeError("TensorVariableTracker state was not initialized")

        active_idx = torch.nonzero(vec, as_tuple=True)[0]
        if changed is None:
            changed = not torch.equal(self.last_vector, vec)
        return self.write_indices(active_idx, vec, changed=changed)

    def write_indices(
        self,
        active_idx: torch.Tensor,
        vec: torch.Tensor,
        changed: bool | None = None,
    ) -> torch.Tensor:
        """Append one activation vector using precomputed active indices."""

        vec = vec.bool()
        self._ensure_state(vec.device)
        if (
            self.history_indices is None
            or self.history_counts is None
            or self.history_lengths is None
            or self.last_vector is None
        ):
            raise RuntimeError("TensorVariableTracker state was not initialized")

        if changed is None:
            changed = not torch.equal(self.last_vector, vec)
        if not changed:
            return active_idx

        active_count = active_idx.numel()
        if active_count == 0:
            self.last_vector.copy_(vec)
            return active_idx

        self._grow_capacity(active_count)
        selected_history = self.history_indices[active_idx]
        selected_counts = self.history_counts[active_idx]
        self.history_indices[active_idx, 1:] = selected_history[:, :-1].clone()
        self.history_counts[active_idx, 1:] = selected_counts[:, :-1].clone()
        self.history_indices[active_idx, 0] = -1
        self.history_indices[active_idx, 0, :active_count] = active_idx.unsqueeze(0).expand(
            active_count,
            -1,
        )
        self.history_counts[active_idx, 0] = active_count
        self.history_lengths[active_idx] = torch.clamp(
            self.history_lengths[active_idx] + 1,
            max=self.max_length + 1,
        )
        self.last_vector.copy_(vec)
        return active_idx

    def coactivation_weight(
        self,
        ele_idx: torch.Tensor,
        max_past_activate: float,
        dtype: torch.dtype,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return weighted historical co-activations from compact active indices."""

        if (
            self.history_indices is None
            or self.history_counts is None
            or self.history_lengths is None
            or self.age_indices is None
            or self.slot_indices is None
        ):
            raise RuntimeError("TensorVariableTracker state was not initialized")

        if out is None:
            total_weight = torch.zeros((self.vars, self.vars), device=ele_idx.device, dtype=dtype)
        else:
            total_weight = out
            total_weight.zero_()
        if ele_idx.numel() == 0 or self.max_length == 0:
            return total_weight

        histories = self.history_indices[ele_idx, 1:]
        counts = self.history_counts[ele_idx, 1:]
        lengths = torch.clamp(self.history_lengths[ele_idx] - 1, min=0)
        max_len = torch.clamp(lengths.max(), min=1)

        ages = self.age_indices
        max_len_float = max_len.to(dtype=dtype)
        weights = max_past_activate * (max_len_float - ages.to(dtype=dtype)) / (max_len_float + 1)
        weights = torch.where(ages < max_len, weights, torch.zeros_like(weights))

        valid = self.slot_indices.view(1, 1, -1) < counts.unsqueeze(2)
        if ele_idx.device.type == "cuda":
            dense_history = torch.zeros(
                (ele_idx.numel(), self.max_length, self.vars),
                device=ele_idx.device,
                dtype=dtype,
            )
            dense_history.scatter_(2, histories.clamp_min(0), valid.to(dtype=dtype))
            coactivations = torch.bmm(
                dense_history.permute(1, 2, 0),
                dense_history.permute(1, 0, 2),
            )
            torch.sum(coactivations * weights.view(-1, 1, 1), dim=0, out=total_weight)
            return total_weight

        pair_valid = valid.unsqueeze(3) & valid.unsqueeze(2) & (weights.view(1, -1, 1, 1) > 0)

        rows = histories.unsqueeze(3).expand(-1, -1, -1, self.capacity)[pair_valid]
        cols = histories.unsqueeze(2).expand(-1, -1, self.capacity, -1)[pair_valid]
        values = weights.view(1, -1, 1, 1).expand(
            ele_idx.numel(),
            -1,
            self.capacity,
            self.capacity,
        )[pair_valid]
        total_weight.index_put_((rows, cols), values.to(dtype=dtype), accumulate=True)
        return total_weight
