"""Project paths and runtime-environment detection for the oasis2 package.

Locally, the project root is this repo's checkout. In Colab, it's the fast
local workspace mounted by colab_setup.bootstrap() (/content/oasis2). An
explicit OASIS2_PROJECT_ROOT env var overrides both, mainly for tests.
"""
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict


def is_colab() -> bool:
    """Return True if running in Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def get_project_root() -> Path:
    env_override = os.environ.get("OASIS2_PROJECT_ROOT")
    if env_override:
        return Path(env_override)
    if is_colab():
        return Path("/content/oasis2")
    # src/oasis2/config.py -> src/oasis2 -> src -> repo root
    return Path(__file__).resolve().parents[2]


def get_data_dir(sub: str = "") -> Path:
    d = get_project_root() / "data" / sub if sub else get_project_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_outputs_dir(sub: str = "") -> Path:
    d = get_project_root() / "outputs" / sub if sub else get_project_root() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_models_dir() -> Path:
    d = get_project_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_gpu_available() -> bool:
    """Return True if a CUDA GPU is accessible."""
    import torch
    return torch.cuda.is_available()


def get_device():
    """Return a torch.device: cuda if available, else cpu."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment_summary() -> Dict[str, Any]:
    """Return a dict summarising the current runtime environment."""
    info = {
        "platform": platform.system(),
        "python": sys.version,
        "colab": is_colab(),
        "gpu_available": is_gpu_available(),
        "project_root": str(get_project_root()),
    }
    if info["gpu_available"]:
        try:
            import torch
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
        except Exception:
            pass
    return info
