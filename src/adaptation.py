"""One continuous fit over independently prepared graph stages."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.artifacts import config_hash, write_json
from src.data.cgcp import (
    DotGraphRecord,
    _array_sha256,
    load_cgcp_graphs,
    prepare_graph_sequence_with_metadata,
)
from src.evaluation import (
    evaluate_history,
    ground_truth_chunk_mean_pairwise_distance_by_frame,
    nmi_by_frame,
)
from src.models import NodeSyncMap
from src.models.training_observer import TrainingStepObservation


@dataclass(frozen=True)
class Stage:
    """Prepared graph segment and its cumulative processed-step interval."""

    record: DotGraphRecord
    sequence: torch.Tensor
    transitions: torch.Tensor
    provenance: dict[str, Any]
    start: int
    end: int


def prepare_stages(config: dict[str, Any], project_root: Path) -> list[Stage]:
    """Rebuild working memory and reset the same input seed for each graph."""
    train = config["train"]
    records = load_cgcp_graphs(project_root / config["data"]["root"], train["graph_names"])
    if len(records) < 2:
        raise ValueError("Adaptation requires at least two ordered graphs")
    nodes = list(records[0].graph.nodes)
    if any(list(record.graph.nodes) != nodes for record in records[1:]):
        raise ValueError("Adaptation graphs must have identical node identities and order")
    stages: list[Stage] = []
    end = 0
    for record in records:
        np.random.seed(train["seed"])
        torch.manual_seed(train["seed"])
        prepared = prepare_graph_sequence_with_metadata(
            record,
            max_seq_length=train["max_seq_length"],
            state_memory=train["state_memory"],
            device=train["device"],
            reset_time=None,
        )
        if len(prepared.sequence) == 0:
            raise ValueError(f"Graph {record.name} produced no training states")
        start = end
        end += len(prepared.sequence)
        stages.append(Stage(
            record,
            torch.as_tensor(prepared.sequence, dtype=torch.bool, device=train["device"]),
            torch.as_tensor(prepared.transition_edges, dtype=torch.long, device=train["device"]),
            {**prepared.provenance, "seed": train["seed"]},
            start,
            end,
        ))
    return stages


def coordinates_numpy(value: torch.Tensor) -> np.ndarray:
    """Copy model state before the next in-place CUDA update."""
    return value.detach().cpu().numpy().copy()


class Recorder:
    """Collect periodic histories, exact boundaries, and durable diagnostics."""

    def __init__(
        self, cadence: int, boundaries: list[int], writer: SummaryWriter, journal: TextIO,
    ) -> None:
        """Initialize storage for a single continuous fit."""
        self.cadence = cadence
        self.boundaries = set(boundaries)
        self.writer = writer
        self.journal = journal
        self.steps: list[int] = []
        self.history: list[np.ndarray] = []
        self.boundary_coordinates: dict[int, np.ndarray] = {}
        self.last_step = 0

    def wants_step(self, step: int) -> bool:
        """Observe regular samples and exact stage endings independently."""
        return step % self.cadence == 0 or step in self.boundaries

    def record(self, observation: TrainingStepObservation) -> None:
        """Persist compact diagnostics and retain requested coordinate copies."""
        step = observation.step
        coordinates = coordinates_numpy(observation.coordinates)
        if step in self.boundaries:
            self.boundary_coordinates[step] = coordinates
        if step % self.cadence == 0:
            self.steps.append(step)
            self.history.append(coordinates)
        diagnostic: dict[str, Any] = {"step": step}
        if observation.radial is not None:
            result = observation.radial.result
            diagnostic.update(
                original_energy=float(result.pre_energy.item()),
                corrected_energy=float(result.post_energy.item()),
                correction_norm=float(result.correction_norm.item()),
            )
            if result.solver is not None:
                diagnostic.update(
                    solver_iterations=result.solver.iterations,
                    solver_converged=result.solver.converged,
                    relative_residual=result.solver.relative_residual,
                )
            for key, value in diagnostic.items():
                if key != "step":
                    self.writer.add_scalar(f"radial/{key}", value, step)
        self.journal.write(json.dumps(diagnostic, allow_nan=False) + "\n")
        self.journal.flush()
        self.writer.flush()
        self.last_step = step


def evaluate_stage(
    stage: Stage, recorder: Recorder, config: dict[str, Any], artifacts: Path,
    initial_coordinates: np.ndarray,
) -> dict[str, Any]:
    """Evaluate periodic stage frames with the original final-window epsilon sweep."""
    all_steps = np.asarray(recorder.steps, dtype=np.int64)
    mask = (all_steps > stage.start) & (all_steps <= stage.end)
    steps = all_steps[mask]
    if steps.size:
        history = np.stack([frame for frame, keep in zip(recorder.history, mask) if keep])
    else:
        history = recorder.boundary_coordinates[stage.end][None, :, :]
        steps = np.asarray([stage.end], dtype=np.int64)
    evaluation = config["eval"]
    metrics = evaluate_history(
        history, steps, stage.end, stage.record.labels,
        np.arange(evaluation["eps_start"], evaluation["eps_stop"], evaluation["eps_step"]),
        evaluation["min_samples"], evaluation["history_window_start_step"],
        evaluation["history_window_end_step"],
    )
    nmi = nmi_by_frame(history, stage.record.labels, metrics["best_eps"], evaluation["min_samples"])
    group_labels, distances = ground_truth_chunk_mean_pairwise_distance_by_frame(
        history, stage.record.labels,
    )
    breathing = np.empty(0, dtype=np.float64)
    if len(history) > 1 and len(group_labels):
        log_changes = np.log((distances[1:].astype(np.float64) + 1e-12)
                             / (distances[:-1].astype(np.float64) + 1e-12))
        breathing = np.sqrt(np.mean(np.square(log_changes), axis=1))
        metrics["b_rms"] = float(np.sqrt(np.mean(np.square(log_changes))))
    trace_dir = artifacts / "traces"
    trace_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        trace_dir / f"{stage.record.name}.npz",
        steps=steps, coordinates=history, labels=stage.record.labels, nmi=nmi,
        group_labels=group_labels, mean_pairwise_distance=distances,
        breathing_steps=steps[1:] if breathing.size else np.empty(0, dtype=np.int64),
        breathing=breathing,
    )
    for step, value in zip(steps, nmi):
        recorder.writer.add_scalar(f"graph/{stage.record.name}/nmi", value, int(step))
    for step, value in zip(steps[1:], breathing):
        recorder.writer.add_scalar(f"graph/{stage.record.name}/b_rms", value, int(step))
    recorder.writer.add_scalar(f"graph/{stage.record.name}/best_nmi", metrics["best_nmi"], stage.end)
    return {
        **metrics, "graph": stage.record.name, "start_step": stage.start,
        "end_step": stage.end, "processed_length": stage.end - stage.start,
        "initial_coordinates_sha256": _array_sha256(initial_coordinates),
        "final_coordinates_sha256": _array_sha256(recorder.boundary_coordinates[stage.end]),
        "input_provenance": stage.provenance,
    }


def run_adaptation(config: dict[str, Any], project_root: Path, leaf: Path) -> dict[str, Any]:
    """Train and evaluate one arm; write completed or failed leaf status."""
    leaf.mkdir(parents=True, exist_ok=False)
    artifacts = leaf / "artifacts"
    status_path = artifacts / "status.json"
    write_json(status_path, {"status": "running", "last_durable_step": 0})
    write_json(artifacts / "config/params.json", config)
    recorder = None
    try:
        stages = prepare_stages(config, project_root)
        write_json(artifacts / "config/provenance.json", {
            "resolved_config_sha256": config_hash(config),
            "state_transfer": "full_model", "fit_calls": 1,
            "input_seed_policy": "reset_same_seed_per_stage",
            "inputs": {stage.record.name: stage.provenance for stage in stages},
            "versions": {"numpy": np.__version__, "torch": torch.__version__},
            "device": config["train"]["device"],
        })
        np.random.seed(config["train"]["seed"])
        torch.manual_seed(config["train"]["seed"])
        model = NodeSyncMap(
            input_size=stages[0].record.num_nodes,
            dimensions=config["train"]["map_dimensions"],
            device=config["train"]["device"], **config["model"],
        )
        initial = coordinates_numpy(model.syncmap)
        sequence = torch.cat([stage.sequence for stage in stages])
        transitions = torch.cat([stage.transitions for stage in stages])
        diagnostics = artifacts / "diagnostics"
        diagnostics.mkdir()
        with SummaryWriter(log_dir=str(leaf)) as writer, (diagnostics / "partial-steps.jsonl").open("w") as journal:
            writer.add_text("configuration", json.dumps(config, sort_keys=True))
            recorder = Recorder(config["logging"]["valid_every"], [s.end for s in stages], writer, journal)
            started = perf_counter()
            model.fit(sequence, observer=recorder, transition_edges=transitions)
            runtime = perf_counter() - started
            boundaries = [initial, *[recorder.boundary_coordinates[s.end] for s in stages]]
            boundary_dir = artifacts / "adaptation"
            boundary_dir.mkdir()
            np.savez_compressed(
                boundary_dir / "boundary_coordinates.npz",
                coordinates=np.stack(boundaries),
                steps=np.asarray([0, *[s.end for s in stages]], dtype=np.int64),
                graph_names=np.asarray([s.record.name for s in stages]),
            )
            graphs = [evaluate_stage(stage, recorder, config, artifacts, boundaries[index])
                      for index, stage in enumerate(stages)]
            metrics = {
                "arm": config["arm"], "seed": config["train"]["seed"],
                "graph_order": [s.record.name for s in stages],
                "max_seq_length": config["train"]["max_seq_length"],
                "total_steps": stages[-1].end, "runtime_seconds": runtime,
                "state_transfer": "full_model", "graphs": graphs,
                "mean_best_nmi": float(np.mean([g["best_nmi"] for g in graphs])),
            }
            write_json(artifacts / "metrics.json", metrics)
            writer.add_scalar("mean_best_nmi", metrics["mean_best_nmi"], stages[-1].end)
        write_json(status_path, {"status": "completed", "last_durable_step": recorder.last_step})
        return metrics
    except BaseException as error:
        write_json(status_path, {
            "status": "failed", "error_type": type(error).__name__, "reason": str(error),
            "last_durable_step": recorder.last_step if recorder else 0,
        })
        raise
