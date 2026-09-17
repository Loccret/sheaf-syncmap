"""Matrix-free radial sheaf geometry and Laplacian operations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite, log

import torch


@dataclass(frozen=True)
class RadialEdgeSet:
    """Unique undirected positive-relation edges.

    Args:
        edge_index: Endpoint indices with shape ``2 x E`` and ``source < target``.
        weight: Raw non-negative edge weights with shape ``E``.
        num_nodes: Number of nodes represented by the source weight matrix.
        raw_weight_sum: Sum of raw retained edge weights.
    """

    edge_index: torch.Tensor
    weight: torch.Tensor
    num_nodes: int
    raw_weight_sum: torch.Tensor

    @property
    def num_edges(self) -> int:
        """Return the number of retained undirected edges."""

        return int(self.edge_index.shape[1])


@dataclass(frozen=True)
class RadialGeometry:
    """Edge-local radial restrictions derived from current coordinates.

    Args:
        edge_index: Endpoint indices with shape ``2 x E``.
        weight: Effective valid-edge weights. Global weights sum to one;
            identical weights equal the distance-decay factors.
        distance: Current Euclidean edge distances.
        direction: Unit vectors pointing from target to source.
        covector: Absolute or distance-normalized radial restrictions.
        valid: Mask selecting edges with a defined radial direction.
        num_nodes: Number of node velocity stalks.
        raw_weight_sum: Sum of valid base weights before decay and normalization;
            identical mode replaces raw relation weights by one first.
        strain_mode: ``relative`` or ``absolute`` restriction mode.
        decay_base: Exponential distance-decay base used for these weights.
        normalize_implicit_edge_weights: ``global`` or ``identical`` weighting.
    """

    edge_index: torch.Tensor
    weight: torch.Tensor
    distance: torch.Tensor
    direction: torch.Tensor
    covector: torch.Tensor
    valid: torch.Tensor
    num_nodes: int
    raw_weight_sum: torch.Tensor
    strain_mode: str
    decay_base: float = 1.0
    normalize_implicit_edge_weights: str = "global"

    @property
    def source(self) -> torch.Tensor:
        """Return source endpoint indices."""

        return self.edge_index[0]

    @property
    def target(self) -> torch.Tensor:
        """Return target endpoint indices."""

        return self.edge_index[1]

    @property
    def weight_sum(self) -> torch.Tensor:
        """Return the sum of effective edge weights."""

        return self.weight.sum()




def find_pure_repellers(attraction_weights: torch.Tensor) -> torch.Tensor:
    """Return nodes with no positive off-diagonal attraction relation.

    Args:
        attraction_weights: Square Boolean or floating attraction matrix.

    Returns:
        Boolean mask with one entry per node; ``True`` denotes a repeller.

    Raises:
        ValueError: When the relation matrix is malformed or nonfinite.
    """

    if (
        attraction_weights.ndim != 2
        or attraction_weights.shape[0] != attraction_weights.shape[1]
    ):
        raise ValueError("attraction_weights must be a square matrix")
    if attraction_weights.dtype != torch.bool and not attraction_weights.is_floating_point():
        raise ValueError("attraction_weights must use a Boolean or floating dtype")
    if attraction_weights.is_floating_point() and not torch.isfinite(attraction_weights).all():
        raise ValueError("attraction_weights must be finite")

    positive = attraction_weights > 0
    positive.fill_diagonal_(False)
    has_attraction = positive.any(dim=0) | positive.any(dim=1)
    return ~has_attraction


def apply_pure_repeller_strength_gate(
    geometry: RadialGeometry,
    pure_repeller: torch.Tensor,
    strength: float,
    edge_gate_selector: str = "min",
) -> RadialGeometry:
    """Scale edges by the selected endpoint repeller-strength reduction.

    Non-repeller nodes receive factor one and repellers receive ``strength``.
    The selector combines the two endpoint factors using their minimum,
    maximum, or arithmetic mean.

    Args:
        geometry: Existing radial geometry with normalized effective weights.
        pure_repeller: Boolean repeller mask with shape ``N``.
        strength: Finite edge-strength multiplier of at least one.
        edge_gate_selector: ``min``, ``max``, or ``mean`` endpoint reduction.

    Returns:
        A geometry sharing all tensors except the gated edge-weight tensor.

    Raises:
        ValueError: When the mask, multiplier, or selector is invalid.
    """

    if pure_repeller.ndim != 1 or int(pure_repeller.shape[0]) != geometry.num_nodes:
        raise ValueError("pure_repeller must have shape num_nodes")
    if pure_repeller.dtype != torch.bool:
        raise ValueError("pure_repeller must use a Boolean dtype")
    if pure_repeller.device != geometry.weight.device:
        raise ValueError("pure_repeller and geometry weights must share a device")
    if isinstance(strength, bool) or not isfinite(strength) or strength < 1:
        raise ValueError("pure-repeller strength must be finite and at least one")
    if edge_gate_selector not in {"min", "max", "mean"}:
        raise ValueError("edge_gate_selector must be min, max, or mean")
    if strength == 1:
        return geometry

    node_strength = torch.where(
        pure_repeller,
        geometry.weight.new_tensor(strength),
        geometry.weight.new_tensor(1.0),
    )
    source_strength = node_strength[geometry.source]
    target_strength = node_strength[geometry.target]
    if edge_gate_selector == "min":
        edge_gate = torch.minimum(source_strength, target_strength)
    elif edge_gate_selector == "max":
        edge_gate = torch.maximum(source_strength, target_strength)
    else:
        edge_gate = 0.5 * (source_strength + target_strength)
    return replace(geometry, weight=geometry.weight * edge_gate)




def build_complete_edge_set(
    num_nodes: int,
    dtype: torch.dtype,
    device: torch.device,
) -> RadialEdgeSet:
    """Build the static uniformly weighted complete undirected topology.

    Args:
        num_nodes: Number of nodes in the complete graph.
        dtype: Floating dtype for unit edge weights.
        device: Device holding edge indices and weights.

    Returns:
        Unique upper-triangular edges with unit raw weights.

    Raises:
        ValueError: When the node count or weight dtype is invalid.
    """

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("dtype must be floating")
    edge_index = torch.triu_indices(
        num_nodes,
        num_nodes,
        offset=1,
        device=device,
    )
    weight = torch.ones(edge_index.shape[1], dtype=dtype, device=device)
    return RadialEdgeSet(
        edge_index=edge_index,
        weight=weight,
        num_nodes=num_nodes,
        raw_weight_sum=weight.sum(),
    )


def _effective_radial_weights(
    raw_weight: torch.Tensor,
    distance: torch.Tensor,
    valid: torch.Tensor,
    decay_base: float,
    normalization: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build effective radial weights and return their valid base-weight sum.

    Args:
        raw_weight: Existing non-negative relation weights with shape ``E``.
        distance: Current edge distances with shape ``E``.
        valid: Mask selecting nondegenerate edges.
        decay_base: Exponential base ``P`` in ``rho_e = P ** (1 - d_e)``.
        normalization: ``global`` uses normalized raw relation weights;
            ``identical`` replaces every valid raw relation weight by one.

    Returns:
        Effective weights ``omega_e`` and the sum of valid base weights.
    """

    if normalization == "global":
        base_weight = raw_weight
    else:
        base_weight = torch.ones_like(raw_weight)
    valid_base_weight = base_weight * valid.to(dtype=base_weight.dtype)
    raw_weight_sum = valid_base_weight.sum()
    if raw_weight.numel() == 0:
        return valid_base_weight, raw_weight_sum

    if decay_base == 1.0:
        if normalization == "identical":
            return valid_base_weight, raw_weight_sum
        tiny = torch.finfo(raw_weight.dtype).tiny
        return valid_base_weight / raw_weight_sum.clamp_min(tiny), raw_weight_sum

    if normalization == "identical":
        decay = torch.exp((1 - distance) * log(decay_base))
        return torch.where(valid, decay, torch.zeros_like(decay)), raw_weight_sum

    active = valid & (base_weight > 0)
    reference_distance = torch.where(
        active,
        distance,
        torch.full_like(distance, torch.inf),
    ).min()
    relative_log_decay = -(distance - reference_distance) * log(decay_base)
    log_unnormalized = torch.log(base_weight) + relative_log_decay
    masked_log_weight = torch.where(
        active,
        log_unnormalized,
        torch.full_like(log_unnormalized, -torch.inf),
    )
    maximum_log_weight = masked_log_weight.max()
    scaled_weight = torch.exp(masked_log_weight - maximum_log_weight)
    scaled_weight = torch.nan_to_num(
        scaled_weight,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    tiny = torch.finfo(raw_weight.dtype).tiny
    normalized_weight = scaled_weight / scaled_weight.sum().clamp_min(tiny)
    return normalized_weight, raw_weight_sum


def build_radial_geometry(
    coordinates: torch.Tensor,
    edges: RadialEdgeSet,
    strain_mode: str = "relative",
    distance_epsilon: float = 1e-6,
    min_distance: float = 1e-6,
    decay_base: float = 1.0,
    normalize_implicit_edge_weights: str = "global",
) -> RadialGeometry:
    """Build radial restriction covectors from pre-step coordinates.

    Args:
        coordinates: Current node coordinates with shape ``N x d``.
        edges: Positive-relation edges over the same nodes.
        strain_mode: ``relative`` divides radial motion by current distance;
            ``absolute`` measures direct radial displacement.
        distance_epsilon: Positive stabilizer for relative normalization.
        min_distance: Edges at or below this distance are excluded.
        decay_base: Exponential distance-decay base ``P``. A value of one
            preserves the original operator exactly.
        normalize_implicit_edge_weights: ``global`` computes normalized
            ``a_e * P ** (1 - d_e)`` weights; ``identical`` computes
            ``P ** (1 - d_e)`` for every valid edge.

    Returns:
        Geometry containing effective valid-edge weights and covectors.

    Raises:
        ValueError: When shapes, devices, dtypes, or scalar settings are invalid.
    """

    if coordinates.ndim != 2 or coordinates.shape[0] != edges.num_nodes:
        raise ValueError("coordinates must have shape num_nodes x dimensions")
    if not coordinates.is_floating_point():
        raise ValueError("coordinates must use a floating dtype")
    if coordinates.device != edges.weight.device or coordinates.dtype != edges.weight.dtype:
        raise ValueError("coordinates and edge weights must share device and dtype")
    if strain_mode not in {"relative", "absolute"}:
        raise ValueError("strain_mode must be 'relative' or 'absolute'")
    if distance_epsilon <= 0:
        raise ValueError("distance_epsilon must be positive")
    if min_distance < 0:
        raise ValueError("min_distance must be non-negative")
    if not isfinite(decay_base) or decay_base < 1:
        raise ValueError("decay_base must be finite and at least one")
    if normalize_implicit_edge_weights not in {"global", "identical"}:
        raise ValueError(
            "normalize_implicit_edge_weights must be global or identical"
        )

    source, target = edges.edge_index
    displacement = coordinates[source] - coordinates[target]
    distance = torch.linalg.vector_norm(displacement, dim=1)
    valid = distance > min_distance
    safe_distance = distance.clamp_min(distance_epsilon)
    direction = displacement / safe_distance.unsqueeze(1)
    direction = direction * valid.unsqueeze(1).to(dtype=coordinates.dtype)
    if strain_mode == "relative":
        covector = direction / (distance + distance_epsilon).unsqueeze(1)
    else:
        covector = direction

    effective_weight, raw_weight_sum = _effective_radial_weights(
        edges.weight,
        distance,
        valid,
        decay_base,
        normalize_implicit_edge_weights,
    )
    return RadialGeometry(
        edge_index=edges.edge_index,
        weight=effective_weight,
        distance=distance,
        direction=direction,
        covector=covector,
        valid=valid,
        num_nodes=edges.num_nodes,
        raw_weight_sum=raw_weight_sum,
        strain_mode=strain_mode,
        decay_base=decay_base,
        normalize_implicit_edge_weights=normalize_implicit_edge_weights,
    )


def radial_strain(velocity: torch.Tensor, geometry: RadialGeometry) -> torch.Tensor:
    """Project relative endpoint velocities onto each radial covector.

    Args:
        velocity: Node proposal velocities with shape ``N x d``.
        geometry: Radial restrictions for the same node geometry.

    Returns:
        One signed radial strain per edge.
    """

    relative_velocity = velocity[geometry.source] - velocity[geometry.target]
    return torch.sum(relative_velocity * geometry.covector, dim=1)


def radial_energy(velocity: torch.Tensor, geometry: RadialGeometry) -> torch.Tensor:
    """Return one half of the effective weighted squared radial strain.

    Args:
        velocity: Node proposal velocities with shape ``N x d``.
        geometry: Radial restrictions for the same node geometry.

    Returns:
        Scalar positive-semidefinite radial energy.
    """

    strain = radial_strain(velocity, geometry)
    return 0.5 * torch.sum(geometry.weight * strain.square())


def apply_radial_laplacian(
    velocity: torch.Tensor,
    geometry: RadialGeometry,
) -> torch.Tensor:
    """Perform ``B.T @ W @ B @ velocity`` without materializing either matrix.

    Args:
        velocity: Node proposal velocities with shape ``N x d``.
        geometry: Radial restrictions for the same node geometry.

    Returns:
        Laplacian action with the same shape as ``velocity``.
    """

    strain = radial_strain(velocity, geometry)  # B @ velocity
    edge_force = geometry.weight.unsqueeze(1) * strain.unsqueeze(1) * geometry.covector  # W @ B @ velocity; shape: E, d
    result = torch.zeros_like(velocity)
    result.index_add_(0, geometry.source, edge_force)  # B.T @ W @ B @ velocity
    result.index_add_(0, geometry.target, -edge_force) # B.T @ W @ B @ velocity
    return result


def radial_laplacian_diagonal(geometry: RadialGeometry) -> torch.Tensor:
    """Return the coordinate-wise diagonal of the radial Laplacian.

    Args:
        geometry: Radial restrictions over ``N`` nodes in ``d`` dimensions.

    Returns:
        Tensor with shape ``N x d`` for Jacobi preconditioning.
    """

    contribution = geometry.weight.unsqueeze(1) * geometry.covector.square()
    diagonal = torch.zeros(
        (geometry.num_nodes, geometry.covector.shape[1]),
        dtype=geometry.covector.dtype,
        device=geometry.covector.device,
    )
    diagonal.index_add_(0, geometry.source, contribution)
    diagonal.index_add_(0, geometry.target, contribution)
    return diagonal
