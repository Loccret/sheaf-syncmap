"""Probabilistic graph loading and working-memory input preparation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable
import re
import warnings

import networkx as nx
import numpy as np
import torch


NODE_PATTERN = re.compile(r'^\s*"?([^"\s\[]+)"?\s+\[.*label\s*=\s*"?([^"\]\s]+)"?.*\]')
EDGE_PATTERN = re.compile(r'^\s*"?([^"\s;]+)"?\s*->\s*"?([^"\s;]+)"?')


@dataclass(frozen=True)
class DotGraphRecord:
    """A loaded DOT graph and the arrays needed by SyncMap training."""

    name: str
    path: Path
    graph: nx.DiGraph
    labels: np.ndarray
    adjacency: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.adjacency.shape[0])


@dataclass(frozen=True)
class PreparedGraphSequence:
    """Working-memory states with aligned raw-walk transition evidence."""

    sequence: np.ndarray | torch.Tensor
    transition_edges: np.ndarray | torch.Tensor
    retained_raw_indices: np.ndarray
    raw_trajectory: np.ndarray
    transition_reasons: tuple[str, ...]
    provenance: dict[str, object]


def resolve_graph_paths(data_root: str | Path, graph_names: Iterable[str] | None = None) -> list[Path]:
    """Resolve CGCP DOT graph paths in deterministic order."""

    root = Path(data_root)
    if graph_names is None:
        paths = sorted(root.glob("*.dot"))
    else:
        paths = [root / name for name in graph_names]
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing CGCP graph files: {missing_text}")
    if not paths:
        raise FileNotFoundError(f"No DOT graph files found under {root}")
    return paths


def read_cgcp_dot(path: str | Path) -> nx.DiGraph:
    """Read the simple CGCP DOT format without requiring pygraphviz."""

    graph_path = Path(path)
    graph = nx.DiGraph()
    for line in graph_path.read_text(encoding="utf-8").splitlines():
        node_match = NODE_PATTERN.match(line)
        if node_match:
            node, label = node_match.groups()
            graph.add_node(node, label=label)
            continue
        edge_match = EDGE_PATTERN.match(line)
        if edge_match:
            source, target = edge_match.groups()
            graph.add_edge(source, target)
    return graph


def load_dot_graph(path: str | Path) -> DotGraphRecord:
    """Load one DOT graph while preserving its declared node order."""

    graph_path = Path(path)
    graph = read_cgcp_dot(graph_path)
    nodes = list(graph.nodes)
    labels = np.array([int(graph.nodes[node]["label"]) for node in nodes])
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    node_index = {node: idx for idx, node in enumerate(nodes)}
    for source, target in graph.edges:
        adjacency[node_index[source], node_index[target]] = 1.0
    return DotGraphRecord(
        name=graph_path.stem,
        path=graph_path,
        graph=graph,
        labels=labels,
        adjacency=adjacency,
    )


def load_cgcp_graphs(data_root: str | Path, graph_names: Iterable[str] | None = None) -> list[DotGraphRecord]:
    """Load the configured CGCP DOT graphs."""

    return [load_dot_graph(path) for path in resolve_graph_paths(data_root, graph_names)]


def dynamic_state_memory(num_nodes: int) -> int:
    """Use ten percent of the variable count, clipped to the interval [2, 30]."""

    return int(np.clip(0.1 * num_nodes, 2, 30))




def _random_walk_with_causes(
    adjacency: np.ndarray,
    length: int,
    reset_time: int | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Generate a random walk and record how every visited state was reached."""

    num_nodes = int(adjacency.shape[0])
    no_outgoing = np.where(np.sum(adjacency, axis=1) == 0)[0]
    if len(no_outgoing) != 0:
        warnings.warn("Some nodes have no outgoing connections.", stacklevel=2)

    starting_node = np.random.choice(num_nodes)
    while starting_node in no_outgoing:
        warnings.warn("Starting node has no outgoing connections. Choosing another node.", stacklevel=2)
        starting_node = np.random.choice(num_nodes)

    trajectory: list[int] = []
    one_hot_vectors: list[np.ndarray] = []
    current_node = int(starting_node)
    steps_since_reset = 0
    current_cause = "initial"
    causes: list[str] = []

    for _ in range(length):
        trajectory.append(current_node)
        causes.append(current_cause)
        one_hot = np.zeros(num_nodes, dtype=np.bool_)
        one_hot[current_node] = True
        one_hot_vectors.append(one_hot)

        if np.sum(adjacency[current_node]) == 0 or (
            reset_time is not None and steps_since_reset == reset_time
        ):
            current_node = int(np.random.choice(num_nodes))
            warnings.warn("No outgoing connections from current node. Choosing another node.", stacklevel=2)
            steps_since_reset = 0
            current_cause = "reset"
        else:
            prob = adjacency[current_node] / np.sum(adjacency[current_node])
            current_node = int(np.random.choice(num_nodes, p=prob))
            steps_since_reset += 1
            current_cause = "graph_transition"

    return np.asarray(trajectory), np.asarray(one_hot_vectors), tuple(causes)


def working_memory_sequence(input_seq: np.ndarray, state_memory: int) -> np.ndarray:
    """Combine activations over the finite working-memory window."""

    memory: deque[np.ndarray] = deque(maxlen=state_memory)
    output_seq: list[np.ndarray] = []
    for state in input_seq:
        memory.append(state)
        current_working_mem = np.asarray(memory)
        output_seq.append(np.sum(current_working_mem, axis=0).astype(np.bool_))
    return np.asarray(output_seq)




def _array_sha256(array: np.ndarray) -> str:
    """Hash array shape, dtype, and contiguous values for provenance."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def prepare_graph_sequence_with_metadata(
    record: DotGraphRecord,
    max_seq_length: int,
    state_memory: int | str = "dynamic",
    device: str = "cpu",
    reset_time: int | None = None,
) -> PreparedGraphSequence:
    """Prepare states and one aligned transition edge per retained raw state."""

    memory_size = dynamic_state_memory(record.num_nodes) if state_memory == "dynamic" else int(state_memory)
    trajectory, one_hot, causes = _random_walk_with_causes(
        record.adjacency,
        length=max_seq_length,
        reset_time=reset_time,
    )
    if memory_size == 1:
        sequence = one_hot
        retained = np.arange(len(one_hot), dtype=np.int64)
    else:
        unfiltered = working_memory_sequence(one_hot, memory_size)
        retained = np.flatnonzero(unfiltered.sum(axis=1) > 1).astype(np.int64)
        sequence = unfiltered[retained]

    edges = np.full((len(retained), 2), -1, dtype=np.int64)
    reasons: list[str] = []
    for processed_idx, raw_idx_value in enumerate(retained):
        raw_idx = int(raw_idx_value)
        reason = "valid"
        if raw_idx == 0:
            reason = "initial"
        elif causes[raw_idx] == "reset":
            reason = "reset"
        else:
            source = int(trajectory[raw_idx - 1])
            target = int(trajectory[raw_idx])
            if source == target:
                reason = "self"
            elif record.adjacency[source, target] <= 0:
                reason = "absent_edge"
            else:
                edges[processed_idx] = (source, target)
        reasons.append(reason)

    reason_counts = {
        reason: reasons.count(reason)
        for reason in ("valid", "initial", "reset", "self", "absent_edge")
    }
    provenance: dict[str, object] = {
        "raw_length": int(len(trajectory)),
        "processed_length": int(len(sequence)),
        "sequence_sha256": _array_sha256(sequence),
        "transition_edges_sha256": _array_sha256(edges),
        "retained_raw_indices_sha256": _array_sha256(retained),
        "transition_reason_counts": reason_counts,
        "graph_sha256": hashlib.sha256(record.path.read_bytes()).hexdigest(),
    }
    if device == "cuda":
        sequence_output: np.ndarray | torch.Tensor = torch.tensor(
            sequence,
            dtype=torch.bool,
            device="cuda",
        )
        edge_output: np.ndarray | torch.Tensor = torch.tensor(
            edges,
            dtype=torch.long,
            device="cuda",
        )
    else:
        sequence_output = sequence
        edge_output = edges
    return PreparedGraphSequence(
        sequence=sequence_output,
        transition_edges=edge_output,
        retained_raw_indices=retained,
        raw_trajectory=trajectory,
        transition_reasons=tuple(reasons),
        provenance=provenance,
    )

