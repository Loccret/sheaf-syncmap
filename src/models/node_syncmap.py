"""Decentralized SyncMap with optional directional history and sheaf correction."""
from __future__ import annotations
from typing import Any, Iterable, Mapping
import numpy as np
import torch
from tqdm import tqdm
from src.models.history import VariableTracker, TensorVariableTracker
from src.models.pairwise import CudaGraphPairwiseUpdate, pairwise_update_from_state
from src.models.training_observer import RadialStepObservation, StepObserver, TrainingStepObservation
from src.regularizers.radial_correction import RadialCorrectionResult, RadialVelocityRegularizer

class NodeSyncMap:
    """Shared decentralized and Sheaf SyncMap update rule."""

    def __init__(
        self,
        input_size: int,
        dimensions: int,
        adaptation_rate: float,
        plus_factor: float = 1,
        minus_factor: float = 0.1,
        attract_range: float = 0.0001,
        repel_range: float = 1,
        plus_exp_factor: float = 0.2,
        minus_exp_factor: float = 1,
        history_repel_factor: float = 1000,
        history_repel_multiplier: float = 1,
        max_track_length: int = 2,
        max_past_activate: float = 0.001,
        use_tqdm: bool = True,
        normalization: bool = False,
        std_factor: float = 2,
        fix_seed: bool = True,
        device: str = "cpu",
        radial_regularization: Mapping[str, Any] | None = None,
        single_direction_repel_history: bool = False,
    ) -> None:
        self.device = torch.device(device)
        if fix_seed:
            np.random.seed(42)
        self.syncmap = torch.tensor(
            np.random.rand(input_size, dimensions).astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        # Resolve aliases such as ``cuda`` to the concrete device selected by
        # PyTorch so all model-owned state uses the same canonical device.
        self.device = self.syncmap.device

        self.input_size = input_size
        self.dimensions = dimensions
        self.adaptation_rate = adaptation_rate
        self.use_tqdm = use_tqdm
        self.plus_factor = plus_factor
        self.minus_factor = minus_factor
        self.attract_range = attract_range
        self.repel_range = repel_range
        self.plus_exp_factor = plus_exp_factor
        self.minus_exp_factor = minus_exp_factor
        self.history_repel_factor = history_repel_factor
        self.history_repel_multiplier = history_repel_multiplier
        self.history_repel = torch.zeros((input_size, input_size), device=self.device)
        self.variable_tracker = VariableTracker(vars=input_size, max_length=max_track_length)
        self.tensor_variable_tracker = TensorVariableTracker(vars=input_size, max_length=max_track_length)
        self.radial_regularizer = RadialVelocityRegularizer(radial_regularization)
        if not isinstance(single_direction_repel_history, bool):
            raise ValueError("single_direction_repel_history must be a Boolean")
        self.single_direction_repel_history = single_direction_repel_history
        self.cuda_graph_pairwise: CudaGraphPairwiseUpdate | None = None
        if self.device.type == "cuda":
            self.cuda_graph_pairwise = CudaGraphPairwiseUpdate(
                input_size,
                dimensions,
                self.syncmap.dtype,
                self.device,
                plus_factor,
                minus_factor,
                plus_exp_factor,
                minus_exp_factor,
                attract_range,
                repel_range,
                history_repel_multiplier,
                adaptation_rate,
                normalization,
                std_factor,
                self.syncmap,
            )
            self.syncmap = self.cuda_graph_pairwise.syncmap
        self.max_past_activate = max_past_activate
        self.normalization = normalization
        self.std_factor = std_factor
        self.last_radial_result: RadialCorrectionResult | None = None
        self.last_positive_weights: torch.Tensor | None = None
        self.last_current_positive_weights: torch.Tensor | None = None
        self._cached_plus_mask: torch.Tensor | None = None
        self._cached_minus_mask: torch.Tensor | None = None
        self._cached_plus_weight_valid = False


    def maybe_tqdm(self, iterable: Iterable, total: int | None = None, use_tqdm: bool = True) -> Iterable:
        if use_tqdm:
            return tqdm(iterable, total=total)
        return iterable

    def fit(
        self,
        input_sequence: torch.Tensor,
        observer: StepObserver | None = None,
        transition_edges: torch.Tensor | None = None,
    ) -> None:
        """Process a sequence and optional aligned transition evidence."""
        transition_mode = self.single_direction_repel_history
        if transition_mode:
            if transition_edges is None:
                raise ValueError("single_direction_repel_history requires transition_edges metadata")
            if transition_edges.ndim != 2 or int(transition_edges.shape[1]) != 2:
                raise ValueError("transition_edges must have shape steps x 2")
            if int(transition_edges.shape[0]) != len(input_sequence):
                raise ValueError("transition_edges must align one-to-one with input_sequence")
            if transition_edges.device != input_sequence.device:
                raise ValueError("transition_edges must share the input sequence device")
            if transition_edges.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise ValueError("transition_edges must have an integer dtype")
            invalid_source = transition_edges[:, 0] == -1
            invalid_target = transition_edges[:, 1] == -1
            if bool((invalid_source != invalid_target).any()):
                raise ValueError("invalid transition_edges must be exactly (-1, -1)")
            valid_edges = transition_edges[~invalid_source]
            if valid_edges.numel() > 0:
                if bool(((valid_edges < 0) | (valid_edges >= self.input_size)).any()):
                    raise ValueError("transition_edges contain out-of-bounds endpoints")
                if bool((valid_edges[:, 0] == valid_edges[:, 1]).any()):
                    raise ValueError("transition_edges endpoints must differ")
            transition_valid = (~invalid_source).detach().cpu()
        else:
            transition_valid = None
        input_negative = input_sequence.logical_not()
        state_changed = torch.ones(len(input_sequence), dtype=torch.bool)
        if len(input_sequence) > 1:
            state_changed[1:] = (input_sequence[1:] != input_sequence[:-1]).any(dim=1).detach().cpu()
        active_indices = None
        active_counts = None
        if self.device.type == "cuda":
            active_counts_cuda = input_sequence.sum(dim=1).to(dtype=torch.long)
            active_counts = active_counts_cuda.detach().cpu()
            max_active = int(active_counts.max().item())
            active_indices = torch.zeros(
                (len(input_sequence), max_active),
                dtype=torch.long,
                device=self.device,
            )
            row_idx, col_idx = torch.nonzero(input_sequence, as_tuple=True)
            starts = torch.cumsum(active_counts_cuda, dim=0) - active_counts_cuda
            active_positions = torch.arange(col_idx.numel(), device=self.device) - starts[row_idx]
            active_indices[row_idx, active_positions] = col_idx

        with torch.inference_mode():
            for item in self.maybe_tqdm(
                range(len(input_sequence)),
                total=len(input_sequence),
                use_tqdm=self.use_tqdm,
            ):
                idx = item
                vplus = input_sequence[idx]
                vminus = input_negative[idx]
                changed = bool(state_changed[idx])
                if active_indices is None:
                    active_idx = None
                    active_count = None
                else:
                    active_idx = active_indices[idx]
                    active_count = active_counts[idx] if active_counts is not None else None
                if active_idx is not None and active_count is not None:
                    new_syncmap = self.one_step_organize(
                        vplus, vminus, idx, changed,
                        active_idx[: int(active_count)],
                        transition_edge=(transition_edges[idx] if transition_edges is not None else None),
                        transition_valid=(bool(transition_valid[idx]) if transition_valid is not None else None),
                        observer=observer,
                    )
                else:
                    new_syncmap = self.one_step_organize(
                        vplus, vminus, idx, changed,
                        transition_edge=(transition_edges[idx] if transition_edges is not None else None),
                        transition_valid=(bool(transition_valid[idx]) if transition_valid is not None else None),
                        observer=observer,
                    )
                if new_syncmap is not None:
                    self.syncmap = new_syncmap




    def repel_constant_update(
        self,
        plus_mask: torch.Tensor,
        minus_mask: torch.Tensor,
        transition_edge: torch.Tensor | None = None,
        transition_valid: bool | None = None,
    ) -> torch.Tensor:
        """Update persistent repulsion and clear the configured attractor memory."""

        self.history_repel.add_(
            minus_mask.to(dtype=self.history_repel.dtype),
            alpha=1.0 / self.history_repel_factor,
        )
        if self.single_direction_repel_history:
            if transition_valid:
                if transition_edge is None:
                    raise ValueError("a valid transition requires transition_edge metadata")
                self.history_repel[transition_edge[0], transition_edge[1]] = 0
        else:
            self.history_repel.masked_fill_(plus_mask, 0)
        self.history_repel.clamp_(max=1.0)
        return self.history_repel

    def compute_update(
        self,
        syncmap_previous: torch.Tensor,
        vplus: torch.Tensor,
        vminus: torch.Tensor,
        state_changed: bool | None = None,
        active_idx: torch.Tensor | None = None,
        transition_edge: torch.Tensor | None = None,
        transition_valid: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        reuse_cached_state = (
            self.cuda_graph_pairwise is not None
            and state_changed is False
            and self._cached_plus_weight_valid
            and self._cached_plus_mask is not None
            and self._cached_minus_mask is not None
        )
        if self.device.type == "cuda":
            if reuse_cached_state:
                total_plus_weight = self.cuda_graph_pairwise.plus_weight
            else:
                if active_idx is None:
                    plus_idx = self.tensor_variable_tracker.write(vplus, changed=state_changed)
                else:
                    plus_idx = self.tensor_variable_tracker.write_indices(
                        active_idx,
                        vplus,
                        changed=state_changed,
                    )
                total_plus_weight = self.tensor_variable_tracker.coactivation_weight(
                    plus_idx,
                    max_past_activate=self.max_past_activate,
                    dtype=syncmap_previous.dtype,
                    out=self.cuda_graph_pairwise.plus_weight if self.cuda_graph_pairwise is not None else None,
                )
        else:
            vplus_np = vplus.cpu().numpy()
            self.variable_tracker.write(vplus_np, changed=state_changed)
            plus_idx = np.where(vplus_np)[0]
            total_plus_weight = self.variable_tracker.coactivation_weight(
                plus_idx,
                max_past_activate=self.max_past_activate,
                device=self.device,
                dtype=syncmap_previous.dtype,
            )

        if reuse_cached_state:
            plus_mask = self._cached_plus_mask
            minus_mask = self._cached_minus_mask
        else:
            plus_mask = vplus.unsqueeze(1) & vplus.unsqueeze(0)
            plus_weight_current = plus_mask.to(dtype=syncmap_previous.dtype)
            minus_mask = vminus.unsqueeze(1) & vminus.unsqueeze(0)
            total_plus_weight += plus_weight_current
            if self.cuda_graph_pairwise is not None:
                self._cached_plus_mask = plus_mask
                self._cached_minus_mask = minus_mask
                self._cached_plus_weight_valid = True
        self.last_positive_weights = total_plus_weight
        self.last_current_positive_weights = plus_mask
        history_repel = self.repel_constant_update(
            plus_mask,
            minus_mask,
            transition_edge=transition_edge,
            transition_valid=transition_valid,
        )
        effective_history_repel = history_repel
        if self.single_direction_repel_history:
            effective_history_repel = (history_repel + history_repel.T) / 2
        if self.cuda_graph_pairwise is not None:
            return self.cuda_graph_pairwise(
                vminus,
                effective_history_repel,
            )
        update_value = pairwise_update_from_state(
            syncmap_previous,
            total_plus_weight,
            vminus,
            effective_history_repel,
            self.plus_factor,
            self.minus_factor,
            self.plus_exp_factor,
            self.minus_exp_factor,
            self.attract_range,
            self.repel_range,
            self.history_repel_multiplier,
        )
        return update_value

    def one_step_organize(
        self,
        vplus: torch.Tensor,
        vminus: torch.Tensor,
        current_state_idx: int,
        state_changed: bool | None = None,
        active_idx: torch.Tensor | None = None,
        transition_edge: torch.Tensor | None = None,
        transition_valid: bool | None = None,
        observer: StepObserver | None = None,
    ) -> torch.Tensor:
        """Apply one model update and emit an optional post-state observation."""
        step = current_state_idx + 1
        record_step = observer is not None and observer.wants_step(step)
        regularization_enabled = (
            self.radial_regularizer.config.mode != "baseline" or record_step
        )
        syncmap_previous = self.syncmap.clone() if regularization_enabled else self.syncmap
        update_result = self.compute_update(
            syncmap_previous,
            vplus,
            vminus,
            state_changed=state_changed,
            active_idx=active_idx,
            transition_edge=transition_edge,
            transition_valid=transition_valid,
        )
        if self.cuda_graph_pairwise is not None and type(self).compute_update is NodeSyncMap.compute_update:
            new_syncmap = update_result
        else:
            if isinstance(update_result, tuple):
                update_value = update_result[0] + update_result[1]
            else:
                update_value = update_result
            new_syncmap = self.syncmap.clone()
            new_syncmap += self.adaptation_rate * update_value
            if self.normalization:
                new_syncmap = (
                    (new_syncmap - new_syncmap.mean())
                    / (self.std_factor * new_syncmap.std(correction=0) + 1e-10)
                )

        if regularization_enabled:
            if self.last_positive_weights is None:
                raise RuntimeError("NodeSyncMap positive weights were not constructed")
            proposal = new_syncmap - syncmap_previous
            self.last_radial_result = self.radial_regularizer.correct(
                syncmap_previous,
                proposal,
                self.last_positive_weights,
                step=step,
                current_positive_weights=self.last_current_positive_weights,
            )
            if self.radial_regularizer.config.mode != "baseline":
                new_syncmap = syncmap_previous + self.last_radial_result.velocity
                if self.cuda_graph_pairwise is not None:
                    self.cuda_graph_pairwise.syncmap.copy_(new_syncmap)
                    new_syncmap = self.cuda_graph_pairwise.syncmap


        if observer is not None and record_step:
            radial_observation = None
            if regularization_enabled and self.last_radial_result is not None:
                radial_observation = RadialStepObservation(
                    coordinates_before=syncmap_previous,
                    proposal=proposal,
                    result=self.last_radial_result,
                )
            observer.record(
                TrainingStepObservation(
                    step=step,
                    coordinates=new_syncmap,
                    radial=radial_observation,
                )
            )

        return new_syncmap


