"""Implicit radial-velocity sheaf correction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from numbers import Integral
from typing import Any, Mapping

import torch

from src.regularizers.radial_sheaf import (
    RadialEdgeSet,
    RadialGeometry,
    apply_pure_repeller_strength_gate,
    apply_radial_laplacian,
    build_complete_edge_set,
    build_radial_geometry,
    find_pure_repellers,
    radial_energy,
    radial_laplacian_diagonal,
)


@dataclass(frozen=True)
class RadialRegularizationConfig:
    """Validated radial regularization settings."""

    mode: str = "baseline"
    strain_mode: str = "relative"
    distance_epsilon: float = 1e-6
    min_distance: float = 1e-6
    implicit_strength: float = 1.0
    cg_relative_tolerance: float = 1e-5
    cg_absolute_tolerance: float = 1e-8
    cg_max_iterations: int = 50
    cg_preconditioner: str = "jacobi"
    solver_failure: str = "error"
    decay_base: float = 2.0
    normalize_implicit_edge_weights: str = "global"
    pure_repeller_sheaf_strength: float = 1.0
    sheaf_repeller_with_history: bool = True
    sheaf_launch_delay: int = 0
    edge_gate_selector: str = "min"

    def __post_init__(self) -> None:
        """Validate configuration values after dataclass construction.

        Raises:
            ValueError: When a mode, scalar range, or solver setting is invalid.
        """

        if self.mode not in {"baseline", "implicit"}:
            raise ValueError("mode must be baseline or implicit")
        if self.strain_mode not in {"relative", "absolute"}:
            raise ValueError("strain_mode must be relative or absolute")
        if isinstance(self.pure_repeller_sheaf_strength, bool):
            raise ValueError("pure_repeller_sheaf_strength must be a finite number")
        finite_fields = (
            "distance_epsilon",
            "min_distance",
            "implicit_strength",
            "cg_relative_tolerance",
            "cg_absolute_tolerance",
            "decay_base",
            "pure_repeller_sheaf_strength",
        )
        for field in finite_fields:
            if not isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.distance_epsilon <= 0:
            raise ValueError("distance_epsilon must be positive")
        if self.min_distance < 0:
            raise ValueError("min_distance must be non-negative")
        if self.implicit_strength < 0:
            raise ValueError("implicit_strength must be non-negative")
        if self.decay_base < 1:
            raise ValueError("decay_base must be at least one")
        if self.pure_repeller_sheaf_strength < 1:
            raise ValueError("pure_repeller_sheaf_strength must be at least one")
        if self.edge_gate_selector not in {"min", "max", "mean"}:
            raise ValueError("edge_gate_selector must be min, max, or mean")
        if not isinstance(self.sheaf_repeller_with_history, bool):
            raise ValueError("sheaf_repeller_with_history must be a Boolean")
        if isinstance(self.sheaf_launch_delay, bool) or not isinstance(
            self.sheaf_launch_delay,
            Integral,
        ):
            raise ValueError("sheaf_launch_delay must be a non-negative integer")
        if self.sheaf_launch_delay < 0:
            raise ValueError("sheaf_launch_delay must be non-negative")
        if self.normalize_implicit_edge_weights not in {"global", "identical"}:
            raise ValueError(
                "normalize_implicit_edge_weights must be global or identical"
            )
        if self.cg_relative_tolerance < 0:
            raise ValueError("cg_relative_tolerance must be non-negative")
        if self.cg_absolute_tolerance < 0:
            raise ValueError("cg_absolute_tolerance must be non-negative")
        if self.cg_max_iterations <= 0:
            raise ValueError("cg_max_iterations must be positive")
        if self.cg_preconditioner not in {"none", "jacobi"}:
            raise ValueError("cg_preconditioner must be none or jacobi")
        if self.solver_failure not in {"error", "use_last_iterate"}:
            raise ValueError("solver_failure must be error or use_last_iterate")

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
    ) -> "RadialRegularizationConfig":
        """Construct configuration from a plain or OmegaConf-like mapping.

        Args:
            mapping: Optional nested model configuration.

        Returns:
            Validated immutable radial configuration.

        Raises:
            ValueError: When the mapping contains unknown configuration keys.
        """

        try:
            return cls(**dict(mapping or {}))
        except TypeError as exc:
            raise ValueError(f"Invalid radial regularization configuration: {exc}") from exc


@dataclass(frozen=True)
class RadialSolverDiagnostics:
    """Convergence evidence from an implicit correction solve."""

    iterations: int
    converged: bool
    initial_residual_norm: float
    final_residual_norm: float
    relative_residual: float


@dataclass(frozen=True)
class RadialCorrectionResult:
    """Corrected proposal and the evidence needed for later diagnostics."""

    velocity: torch.Tensor
    geometry: RadialGeometry
    pre_energy: torch.Tensor
    post_energy: torch.Tensor
    correction_norm: torch.Tensor
    step: int
    solver: RadialSolverDiagnostics | None = None
    pure_repeller_mask: torch.Tensor | None = None






def _result(
    proposal: torch.Tensor,
    velocity: torch.Tensor,
    geometry: RadialGeometry,
    step: int,
    solver: RadialSolverDiagnostics | None = None,
) -> RadialCorrectionResult:
    """Build a correction result with common energy and norm fields."""

    return RadialCorrectionResult(
        velocity=velocity,
        geometry=geometry,
        pre_energy=radial_energy(proposal, geometry),
        post_energy=radial_energy(velocity, geometry),
        correction_norm=torch.linalg.vector_norm(velocity - proposal),
        step=step,
        solver=solver,
    )




def _implicit_operator(
    velocity: torch.Tensor,
    geometry: RadialGeometry,
    strength: float,
) -> torch.Tensor:
    """Apply the symmetric positive-definite implicit system matrix."""

    return velocity + strength * apply_radial_laplacian(velocity, geometry)


def _relative_residual(residual_norm: torch.Tensor, right_hand_norm: torch.Tensor) -> float:
    """Convert a tensor residual to a stable scalar relative residual."""

    denominator = max(float(right_hand_norm.item()), torch.finfo(residual_norm.dtype).tiny)
    return float(residual_norm.item()) / denominator


def implicit_radial_correction(
    proposal: torch.Tensor,
    geometry: RadialGeometry,
    strength: float,
    relative_tolerance: float = 1e-5,
    absolute_tolerance: float = 1e-8,
    max_iterations: int = 50,
    preconditioner: str = "jacobi",
    failure_policy: str = "error",
    step: int = 0,
) -> RadialCorrectionResult:
    """Solve ``(I + strength * L) v = proposal`` with matrix-free CG.

    Args:
        proposal: Baseline node displacement and right-hand side.
        geometry: Current radial restrictions.
        strength: Non-negative implicit objective weight.
        relative_tolerance: Residual tolerance relative to the right-hand norm.
        absolute_tolerance: Absolute residual floor.
        max_iterations: Positive conjugate-gradient iteration limit.
        preconditioner: ``none`` or coordinate-wise ``jacobi``.
        failure_policy: ``error`` or ``use_last_iterate``.
        step: Source NodeSyncMap step for diagnostics.

    Returns:
        Globally corrected proposal and solver diagnostics.

    Raises:
        ValueError: When solver settings are invalid.
        RuntimeError: When strict convergence is requested but not achieved.
    """

    if strength < 0:
        raise ValueError("strength must be non-negative")
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("solver tolerances must be non-negative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if preconditioner not in {"none", "jacobi"}:
        raise ValueError("preconditioner must be none or jacobi")
    if failure_policy not in {"error", "use_last_iterate"}:
        raise ValueError("failure_policy must be error or use_last_iterate")

    if strength == 0:
        diagnostics = RadialSolverDiagnostics(0, True, 0.0, 0.0, 0.0)
        return _result(proposal, proposal, geometry, step, solver=diagnostics)

    right_hand = proposal
    right_hand_norm = torch.linalg.vector_norm(right_hand)
    velocity = proposal.clone()
    # residual = proposal - delta, where delta = (I + strength * L) @ velocity
    residual = right_hand - _implicit_operator(velocity, geometry, strength)
    initial_residual = torch.linalg.vector_norm(residual)
    # threshold measures the acceptable linear-system residual norm, higher it is, higher the edge strains
    threshold = absolute_tolerance + relative_tolerance * float(right_hand_norm.item())
    converged = float(initial_residual.item()) <= threshold
    iterations = 0

    if preconditioner == "jacobi":
        # This rescales each node-coordinate according to how strongly it is constrained, usually improving convergence.
        inverse_diagonal = torch.reciprocal(
            1 + strength * radial_laplacian_diagonal(geometry)
        )
    else:
        inverse_diagonal = torch.ones_like(proposal)

    if not converged:
        preconditioned_residual = inverse_diagonal * residual
        direction = preconditioned_residual.clone()
        residual_product = torch.sum(residual * preconditioned_residual)
        tiny = torch.finfo(proposal.dtype).tiny
        for iterations in range(1, max_iterations + 1):
            # update operator_direction (Ap_k) every iteration
            operator_direction = _implicit_operator(direction, geometry, strength)
            denominator = torch.sum(direction * operator_direction)
            if abs(float(denominator.item())) <= tiny:
                break
            alpha = residual_product / denominator
            velocity = velocity + alpha * direction
            residual = residual - alpha * operator_direction
            residual_norm = torch.linalg.vector_norm(residual)
            if float(residual_norm.item()) <= threshold:
                residual = right_hand - _implicit_operator(velocity, geometry, strength)
                residual_norm = torch.linalg.vector_norm(residual)
                if float(residual_norm.item()) <= threshold:
                    converged = True
                    break
                preconditioned_residual = inverse_diagonal * residual
                direction = preconditioned_residual.clone()
                residual_product = torch.sum(residual * preconditioned_residual)
                continue
            preconditioned_residual = inverse_diagonal * residual
            next_product = torch.sum(residual * preconditioned_residual)
            beta = next_product / residual_product
            direction = preconditioned_residual + beta * direction
            residual_product = next_product

    residual = right_hand - _implicit_operator(velocity, geometry, strength)
    final_residual = torch.linalg.vector_norm(residual)
    converged = float(final_residual.item()) <= threshold
    diagnostics = RadialSolverDiagnostics(
        iterations=iterations,
        converged=converged,
        initial_residual_norm=float(initial_residual.item()),
        final_residual_norm=float(final_residual.item()),
        relative_residual=_relative_residual(final_residual, right_hand_norm),
    )
    if not converged and failure_policy == "error":
        raise RuntimeError(
            "Implicit radial correction did not converge: "
            f"relative residual {diagnostics.relative_residual:.3e} after {iterations} iterations"
        )
    return _result(proposal, velocity, geometry, step, solver=diagnostics)


class RadialVelocityRegularizer:
    """Build current geometry and dispatch one configured correction mode."""

    def __init__(
        self,
        config: RadialRegularizationConfig | Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the regularizer from validated settings.

        Args:
            config: Dataclass or nested model configuration mapping.
        """

        if isinstance(config, RadialRegularizationConfig):
            self.config = config
        else:
            self.config = RadialRegularizationConfig.from_mapping(config)
        self._complete_edge_cache: dict[
            tuple[int, torch.dtype, torch.device],
            RadialEdgeSet,
        ] = {}

    def _complete_edges(self, coordinates: torch.Tensor) -> RadialEdgeSet:
        """Return the cached complete topology matching current tensor metadata.

        Args:
            coordinates: Node coordinates that determine node count, dtype, and device.

        Returns:
            A cached unit-weight complete undirected edge set.
        """

        key = (int(coordinates.shape[0]), coordinates.dtype, coordinates.device)
        edges = self._complete_edge_cache.get(key)
        if edges is None:
            edges = build_complete_edge_set(
                key[0],
                dtype=key[1],
                device=key[2],
            )
            self._complete_edge_cache[key] = edges
        return edges

    def correct(
        self,
        coordinates: torch.Tensor,
        proposal: torch.Tensor,
        positive_weights: torch.Tensor,
        step: int,
        current_positive_weights: torch.Tensor | None = None,
    ) -> RadialCorrectionResult:
        """Apply the configured correction to one complete NodeSyncMap proposal.

        Args:
            coordinates: Pre-step node coordinates.
            proposal: Complete baseline coordinate displacement.
            positive_weights: Same-step positive/coactivation relation matrix.
            step: NodeSyncMap step index.
            current_positive_weights: Optional current-state coactivation matrix.
                Required for a non-default pure-repeller gate when history is
                excluded from pure-repeller detection.

        Returns:
            Corrected proposal with current geometry and numerical evidence.
        """

        config = self.config
        edges = self._complete_edges(coordinates)
        geometry = build_radial_geometry(
            coordinates,
            edges,
            strain_mode=config.strain_mode,
            distance_epsilon=config.distance_epsilon,
            min_distance=config.min_distance,
            decay_base=config.decay_base if config.mode == "implicit" else 1.0,
            normalize_implicit_edge_weights=(
                config.normalize_implicit_edge_weights
                if config.mode == "implicit"
                else "global"
            ),
        )
        pure_repeller_mask = None
        if config.mode == "implicit" and config.pure_repeller_sheaf_strength > 1:
            attraction_weights = (
                positive_weights
                if config.sheaf_repeller_with_history
                else current_positive_weights
            )
            if attraction_weights is None:
                raise ValueError(
                    "current_positive_weights are required when "
                    "sheaf_repeller_with_history is false"
                )
            pure_repeller_mask = find_pure_repellers(attraction_weights)
            geometry = apply_pure_repeller_strength_gate(
                geometry,
                pure_repeller_mask,
                config.pure_repeller_sheaf_strength,
                config.edge_gate_selector,
            )
        if config.mode == "baseline" or step < config.sheaf_launch_delay:
            result = _result(proposal, proposal, geometry, step)
        else:
            result = implicit_radial_correction(
                proposal,
                geometry,
                strength=config.implicit_strength,
                relative_tolerance=config.cg_relative_tolerance,
                absolute_tolerance=config.cg_absolute_tolerance,
                max_iterations=config.cg_max_iterations,
                preconditioner=config.cg_preconditioner,
                failure_policy=config.solver_failure,
                step=step,
            )
        if pure_repeller_mask is not None:
            result = replace(result, pure_repeller_mask=pure_repeller_mask)
        return result
