"""Release-local run artifacts without repository discovery."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def json_text(value: Any) -> str:
    """Serialize deterministic, finite JSON for artifacts and config hashes."""
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json(path: Path, value: Any) -> None:
    """Atomically replace one artifact, creating its directory if necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json_text(value), encoding="utf-8")
    temporary.replace(path)


def allocate_run(root: Path, experiment: str) -> tuple[Path, str]:
    """Claim a fresh timestamped run directory with a unique run ID."""
    now = datetime.now().astimezone()
    run_id = uuid4().hex[:12]
    parent = root / now.strftime("%Y-%m%d")
    parent.mkdir(parents=True, exist_ok=True)
    run = parent / f"{now:%Y-%m%d-%H%M%S}-{experiment}-{run_id}"
    run.mkdir(exist_ok=False)
    return run, run_id


def config_hash(config: dict[str, Any]) -> str:
    """Hash the exact resolved configuration written with the run."""
    return hashlib.sha256(json_text(config).encode("utf-8")).hexdigest()
