"""Pairwise decentralized updates and CUDA graph integration."""
from __future__ import annotations
import torch

def pairwise_update_reduction(
    direction: torch.Tensor,
    distance: torch.Tensor,
    plus_weight: torch.Tensor,
    minus_mask: torch.Tensor,
    history_repel: torch.Tensor,
    plus_factor: float,
    minus_factor: float,
    plus_exp_factor: float,
    minus_exp_factor: float,
    attract_range: float,
    repel_range: float,
    history_repel_multiplier: float,
) -> torch.Tensor:
    """Combine attraction and repulsion into one pairwise reduction."""

    weighted_plus = plus_weight * (
        1 + plus_exp_factor * torch.exp(-distance / attract_range)
    )
    weighted_minus = (
        minus_mask
        * minus_exp_factor
        * torch.exp(-distance / repel_range)
        + history_repel * history_repel_multiplier
    )
    combined_weight = minus_factor * weighted_minus - plus_factor * weighted_plus
    return (direction * combined_weight.unsqueeze(2)).sum(dim=0)

def pairwise_update_from_state(
    syncmap: torch.Tensor,
    plus_weight: torch.Tensor,
    vminus: torch.Tensor,
    history_repel: torch.Tensor,
    plus_factor: float,
    minus_factor: float,
    plus_exp_factor: float,
    minus_exp_factor: float,
    attract_range: float,
    repel_range: float,
    history_repel_multiplier: float,
) -> torch.Tensor:
    """Build pairwise geometry and reduce the NodeSyncMap update."""

    coordinate_diff = syncmap.unsqueeze(0) - syncmap.unsqueeze(1)
    distance = torch.sqrt(torch.sum(coordinate_diff * coordinate_diff, dim=-1))
    inverse_distance = torch.nan_to_num(torch.reciprocal(distance), posinf=0)
    direction = coordinate_diff * inverse_distance.unsqueeze(2)
    minus_mask = (vminus.unsqueeze(1) & vminus.unsqueeze(0)).to(dtype=syncmap.dtype)
    return pairwise_update_reduction(
        direction,
        distance,
        plus_weight,
        minus_mask,
        history_repel,
        plus_factor,
        minus_factor,
        plus_exp_factor,
        minus_exp_factor,
        attract_range,
        repel_range,
        history_repel_multiplier,
    )

class CudaGraphPairwiseUpdate:
    """Replay the fixed-shape pairwise update and SyncMap integration."""

    def __init__(
        self,
        num_nodes: int,
        dimensions: int,
        dtype: torch.dtype,
        device: torch.device,
        plus_factor: float,
        minus_factor: float,
        plus_exp_factor: float,
        minus_exp_factor: float,
        attract_range: float,
        repel_range: float,
        history_repel_multiplier: float,
        adaptation_rate: float,
        normalization: bool,
        std_factor: float,
        initial_syncmap: torch.Tensor,
    ) -> None:
        self.syncmap = initial_syncmap.detach().clone()
        self.plus_weight = torch.empty((num_nodes, num_nodes), dtype=dtype, device=device)
        self.vminus = torch.empty(num_nodes, dtype=torch.bool, device=device)
        self.history_repel = torch.empty((num_nodes, num_nodes), dtype=dtype, device=device)
        self.output: torch.Tensor | None = self.syncmap
        self.graph = torch.cuda.CUDAGraph()
        self.adaptation_rate = adaptation_rate
        self.normalization = normalization
        self.std_factor = std_factor
        self.factors = (
            plus_factor,
            minus_factor,
            plus_exp_factor,
            minus_exp_factor,
            attract_range,
            repel_range,
            history_repel_multiplier,
        )
        self._capture()

    def _run_step(self) -> torch.Tensor:
        update = pairwise_update_from_state(
            self.syncmap,
            self.plus_weight,
            self.vminus,
            self.history_repel,
            *self.factors,
        )
        self.syncmap.add_(update, alpha=self.adaptation_rate)
        if self.normalization:
            self.syncmap.sub_(self.syncmap.mean())
            self.syncmap.div_(self.std_factor * self.syncmap.std(correction=0) + 1e-10)
        return self.syncmap

    def _capture(self) -> None:
        initial_syncmap = self.syncmap.clone()
        self.plus_weight.zero_()
        self.vminus.zero_()
        self.history_repel.zero_()
        for _ in range(3):
            self.output = self._run_step()
        torch.cuda.synchronize()
        self.syncmap.copy_(initial_syncmap)
        with torch.cuda.graph(self.graph):
            self.output = self._run_step()
        self.syncmap.copy_(initial_syncmap)

    def __call__(
        self,
        vminus: torch.Tensor,
        history_repel: torch.Tensor,
    ) -> torch.Tensor:
        self.vminus.copy_(vminus)
        self.history_repel.copy_(history_repel)
        self.graph.replay()
        if self.output is None:
            raise RuntimeError("CUDA graph pairwise update was not captured")
        return self.output
