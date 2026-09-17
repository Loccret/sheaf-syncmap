"""Run the four-arm ablation or the Sheaf-only adaptation experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
import logging
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from src.adaptation import run_adaptation
from src.artifacts import allocate_run, write_json


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the experiment choice and small set of runtime overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=("ablation", "adaption"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--steps", type=int, help="Raw random-walk length per graph")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--valid-every", type=int)
    parser.add_argument("--sheaf-launch-delay", type=int)
    parser.add_argument("--threads", type=int, help="PyTorch CPU threads")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quiet", action="store_true", help="Disable model progress bars")
    return parser.parse_args(argv)


def load_config(project_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Compose common and experiment settings, then apply explicit overrides."""
    with (project_root / "configs/defaults.yaml").open() as handle:
        config = yaml.safe_load(handle)
    with (project_root / f"configs/{args.experiment}.yaml").open() as handle:
        config.update(yaml.safe_load(handle))
    for argument, key in [("steps", "max_seq_length"), ("seed", "seed"),
                          ("device", "device"), ("threads", "num_threads")]:
        value = getattr(args, argument)
        if value is not None:
            config["train"][key] = value
    if args.valid_every is not None:
        config["logging"]["valid_every"] = args.valid_every
    if args.sheaf_launch_delay is not None:
        config["model"]["radial_regularization"]["sheaf_launch_delay"] = args.sheaf_launch_delay
    if args.quiet:
        config["model"]["use_tqdm"] = False
    train = config["train"]
    if train["max_seq_length"] < 2:
        raise ValueError("steps must be at least two")
    if train["num_threads"] < 1 or config["logging"]["valid_every"] < 1:
        raise ValueError("threads and valid-every must be positive")
    if train["device"] == "auto":
        train["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if train["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    if train["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if config["eval"]["eps_step"] <= 0 or not 0 < config["eval"]["eps_start"] < config["eval"]["eps_stop"]:
        raise ValueError("evaluation epsilon range must be positive and increasing")
    config["experiment"] = args.experiment
    return config


def arm_config(config: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    """Resolve one condition with both independent mechanism switches."""
    resolved = deepcopy(config)
    resolved.pop("arms")
    resolved["arm"] = arm["name"]
    resolved["model"]["radial_regularization"]["mode"] = arm["mode"]
    resolved["model"]["single_direction_repel_history"] = arm["directional_history"]
    return resolved


def run_experiment(config: dict[str, Any], project_root: Path, output_root: Path) -> Path:
    """Run fresh arm/seed leaves sequentially and track their parent manifest."""
    torch.set_num_threads(config["train"]["num_threads"])
    run, run_id = allocate_run(output_root, config["experiment"])
    seed = config["train"]["seed"]
    manifest = {
        "experiment": config["experiment"], "run_id": run_id, "status": "running",
        "graph_order": config["train"]["graph_names"],
        "max_seq_length_per_graph": config["train"]["max_seq_length"],
        "seed": seed, "state_transfer": "full_model",
        "arms": [{**arm, "path": f"{arm['name']}/seed-{seed}", "status": "planned"}
                 for arm in config["arms"]],
    }
    path = run / "artifacts/run_manifest.json"
    write_json(path, manifest)
    logger = logging.getLogger(__name__)
    logger.info("Run directory: %s", run)
    try:
        for arm in manifest["arms"]:
            arm["status"] = "running"
            write_json(path, manifest)
            logger.info("Starting %s (seed %s)", arm["name"], seed)
            run_adaptation(arm_config(config, arm), project_root, run / arm["path"])
            arm["status"] = "completed"
            write_json(path, manifest)
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        for arm in manifest["arms"]:
            if arm["status"] == "running":
                arm["status"] = "failed"
        raise
    finally:
        write_json(path, manifest)
    logger.info("Completed: %s", run)
    return run


def main(argv: Sequence[str] | None = None) -> None:
    """Run from release-local source and configuration without parent lookups."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root, args)
    output_root = args.output_dir or project_root / config["logging"]["tensorboard_dir"]
    run_experiment(config, project_root, output_root)


if __name__ == "__main__":
    main()
