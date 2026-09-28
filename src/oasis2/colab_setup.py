"""Colab-only environment bootstrap: GPU check, Drive mount, dependency install.

Import this module only when running in Colab (guarded by config.is_colab()).
Importing it outside Colab is safe (it has no top-level `google.colab` import);
`mount_drive()` and `colab_drive_path()` raise RuntimeError if called outside
Colab, and `bootstrap()` will fail at the `mount_drive()` step. This keeps the
rest of the oasis2 package importable and testable in any environment while
still letting this module itself be imported (e.g. for `check_gpu()`, which
works everywhere) without a Colab runtime.
"""
import subprocess
import sys
from pathlib import Path

from oasis2.config import is_colab
from oasis2.logging_utils import setup_logging

logger = setup_logging("colab_setup")


def check_gpu() -> None:
    try:
        gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if gpu.returncode != 0:
            logger.warning('No GPU detected! Runtime > Change runtime type > T4 GPU. '
                            'Phase 1 (SynthSeg) will be ~12x slower on CPU.')
        else:
            lines = [l for l in gpu.stdout.split('\n') if any(k in l for k in ('GeForce', 'Tesla', 'T4', 'A100'))]
            logger.info(f"GPU: {lines[0].strip() if lines else 'detected'}")
    except FileNotFoundError:
        logger.warning('nvidia-smi not found. No GPU detected!')


def mount_drive() -> Path:
    if not is_colab():
        raise RuntimeError("mount_drive() only works inside Google Colab")
    from google.colab import drive
    drive.mount('/content/drive')
    drive_base = Path('/content/drive/MyDrive/oasis2_pipeline')
    drive_base.mkdir(parents=True, exist_ok=True)
    local_base = Path('/content/oasis2')
    for sub in ['data/raw', 'data/processed', 'outputs', 'models']:
        (local_base / sub).mkdir(parents=True, exist_ok=True)
    logger.info(f"Drive mounted at: {drive_base}")
    logger.info(f"Local workspace: {local_base}")
    return drive_base


def install_dependencies() -> None:
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q',
         'nibabel', 'nilearn', 'SimpleITK', 'networkx', 'statsmodels', 'kaggle', 'tqdm', 'openpyxl', 'captum'],
        check=True,
    )
    logger.info("Dependencies installed.")


def colab_drive_path(subpath: str = "") -> Path:
    if not is_colab():
        raise RuntimeError("colab_drive_path() only works inside Google Colab")
    base = Path("/content/drive/MyDrive/oasis2_pipeline")
    base.mkdir(parents=True, exist_ok=True)
    return base / subpath if subpath else base


def bootstrap() -> None:
    """Full Colab bootstrap: GPU check, Drive mount, dependency install."""
    check_gpu()
    mount_drive()
    install_dependencies()
