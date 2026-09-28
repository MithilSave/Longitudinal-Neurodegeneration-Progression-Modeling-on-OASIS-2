"""Checkpoint, JSON, and array helpers for the oasis2 package."""
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

from oasis2.config import get_outputs_dir


def save_checkpoint(data: Any, name: str, directory: Optional[Path] = None) -> Path:
    """Pickle-save `data` to `directory/<name>.pkl`. Default directory is outputs/checkpoints/."""
    if directory is None:
        directory = get_outputs_dir("checkpoints")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def load_checkpoint(name: str, directory: Optional[Path] = None) -> Any:
    """Load and return a previously saved checkpoint. Raises FileNotFoundError if missing."""
    if directory is None:
        directory = get_outputs_dir("checkpoints")
    directory = Path(directory)
    path = directory / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def checkpoint_exists(name: str, directory: Optional[Path] = None) -> bool:
    if directory is None:
        directory = get_outputs_dir("checkpoints")
    return (Path(directory) / f"{name}.pkl").exists()


def save_json(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def z_score(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standardise array along given axis (zero mean, unit variance)."""
    mu = arr.mean(axis=axis, keepdims=True)
    sigma = arr.std(axis=axis, keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (arr - mu) / sigma
