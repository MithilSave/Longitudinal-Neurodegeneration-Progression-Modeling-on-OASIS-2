"""Template connectome, graph Laplacian, and network-diffusion-model (NDM)
propagation.

Ported from OASIS2_Colab.py (lines 2173-2342).

NOTE (bugfix): the original script called `scipy.linalg.eigh` in
`eigen_decompose_laplacian` without ever importing `scipy` or `scipy.linalg`
before that point in the file (the only `scipy` import in the whole 3549-line
script is `import scipy.stats`, which appears textually AFTER this function is
defined and called). Since the script runs top-to-bottom, this meant
`run_phase3()` — whose first step is `eigen_decompose_laplacian` — crashed
with `NameError: name 'scipy' is not defined` on every run. This module adds
the missing `import scipy.linalg` to fix that.
"""
import scipy.linalg
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from oasis2.config import get_data_dir
from oasis2.logging_utils import setup_logging

logger = setup_logging("connectome")


# ---------------------------------------------------------------------------
# Template connectome
# ---------------------------------------------------------------------------

CONNECTOME_REGIONS = [
    # Desikan-Killiany parcellation regions (subset used in Raj et al.)
    # Left hemisphere
    "ctx-lh-bankssts", "ctx-lh-caudalanteriorcingulate", "ctx-lh-caudalmiddlefrontal",
    "ctx-lh-cuneus", "ctx-lh-entorhinal", "ctx-lh-fusiform",
    "ctx-lh-inferiorparietal", "ctx-lh-inferiortemporal", "ctx-lh-isthmuscingulate",
    "ctx-lh-lateraloccipital", "ctx-lh-lateralorbitofrontal", "ctx-lh-lingual",
    "ctx-lh-medialorbitofrontal", "ctx-lh-middletemporal", "ctx-lh-parahippocampal",
    "ctx-lh-paracentral", "ctx-lh-parsopercularis", "ctx-lh-parsorbitalis",
    "ctx-lh-parstriangularis", "ctx-lh-pericalcarine", "ctx-lh-postcentral",
    "ctx-lh-posteriorcingulate", "ctx-lh-precentral", "ctx-lh-precuneus",
    "ctx-lh-rostralanteriorcingulate", "ctx-lh-rostralmiddlefrontal",
    "ctx-lh-superiorfrontal", "ctx-lh-superiorparietal", "ctx-lh-superiortemporal",
    "ctx-lh-supramarginal", "ctx-lh-frontalpole", "ctx-lh-temporalpole",
    "ctx-lh-transversetemporal", "ctx-lh-insula",
    # Subcortical (bilateral)
    "Left-Hippocampus", "Right-Hippocampus",
    "Left-Amygdala", "Right-Amygdala",
    "Left-Thalamus-Proper", "Right-Thalamus-Proper",
    "Left-Caudate", "Right-Caudate",
    "Left-Putamen", "Right-Putamen",
    # Right hemisphere cortical
    "ctx-rh-bankssts", "ctx-rh-caudalanteriorcingulate", "ctx-rh-caudalmiddlefrontal",
    "ctx-rh-entorhinal", "ctx-rh-inferiorparietal", "ctx-rh-inferiortemporal",
    "ctx-rh-middletemporal", "ctx-rh-parahippocampal", "ctx-rh-precuneus",
    "ctx-rh-superiorfrontal", "ctx-rh-superiortemporal", "ctx-rh-insula",
]

N_REGIONS = len(CONNECTOME_REGIONS)


def build_synthetic_connectome(n_regions: int = N_REGIONS, seed: int = 42) -> np.ndarray:
    """
    Build a synthetic structural connectivity matrix for testing/fallback.

    In practice, this should be replaced with a real template connectome
    (e.g., Budapest Reference Connectome, DSI Studio HCP atlas).

    The structure here approximates known brain connectivity properties:
    - Scale-free-like degree distribution
    - Higher within-hemisphere than between-hemisphere connections
    - Stronger medial temporal lobe connections (critical for AD)
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((n_regions, n_regions))

    # Random sparse base connectivity
    base = rng.exponential(scale=0.3, size=(n_regions, n_regions))
    base = (base + base.T) / 2
    np.fill_diagonal(base, 0)

    # Threshold to get sparsity (~20% density, typical for structural connectomes)
    threshold = np.percentile(base, 80)
    A = np.where(base > threshold, base, 0)

    # Boost medial temporal connections (hippocampus/entorhinal — AD signature)
    mtl_indices = [i for i, r in enumerate(CONNECTOME_REGIONS[:n_regions])
                   if any(k in r.lower() for k in ["hippocampus", "entorhinal", "parahippocampal"])]
    for i in mtl_indices:
        for j in mtl_indices:
            if i != j:
                A[i, j] += 0.5

    np.fill_diagonal(A, 0)
    return A


def load_template_connectome(connectome_path: Optional[Path] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Load a structural connectome adjacency matrix.

    Priority:
    1. Load from connectome_path if provided (must be .npy or .csv).
    2. Try to find a saved template in data/processed/.
    3. Fall back to the synthetic approximate connectome.

    Returns:
        (A, region_names) where A is the (n, n) adjacency matrix.
    """
    if connectome_path is not None and connectome_path.exists():
        try:
            if connectome_path.suffix == ".npy":
                A = np.load(str(connectome_path))
                logger.info(f"Loaded connectome from {connectome_path}: shape {A.shape}")
                return A, CONNECTOME_REGIONS[:A.shape[0]]
            elif connectome_path.suffix == ".csv":
                df = pd.read_csv(connectome_path, index_col=0)
                A = df.values
                regions = list(df.columns)
                logger.info(f"Loaded connectome CSV: {A.shape}, regions={regions[:3]}...")
                return A, regions
        except Exception as e:
            logger.warning(f"Failed to load connectome from {connectome_path}: {e}")

    # Check for a cached template
    cached_path = get_data_dir("processed") / "template_connectome.npy"
    if cached_path.exists():
        A = np.load(str(cached_path))
        logger.info(f"Loaded cached template connectome: {A.shape}")
        return A, CONNECTOME_REGIONS[:A.shape[0]]

    # Fallback: synthetic
    logger.warning(
        "No template connectome found. Using synthetic approximate connectome.\n"
        "For production use, replace with a real template such as:\n"
        "  - Budapest Reference Connectome (http://braingraph.org/cms/download-datasets/)\n"
        "  - DSI Studio HCP-1065 atlas\n"
        "This is consistent with Raj et al. (2015) precedent (healthy-subject reference)."
    )
    A = build_synthetic_connectome(N_REGIONS)
    np.save(str(cached_path), A)
    return A, CONNECTOME_REGIONS


# ---------------------------------------------------------------------------
# Graph Laplacian
# ---------------------------------------------------------------------------

def compute_laplacian(A: np.ndarray) -> np.ndarray:
    """
    Compute the normalized graph Laplacian L = D - A.
    D is the degree matrix (diagonal).
    """
    # Ensure symmetry and non-negative weights
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    A = np.maximum(A, 0)

    degree = A.sum(axis=1)
    D = np.diag(degree)
    L = D - A
    return L


def eigen_decompose_laplacian(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigendecompose the Laplacian: L = V @ diag(eigenvalues) @ V^T.
    Uses scipy.linalg.eigh (symmetric matrix — guaranteed real eigenvalues).

    Returns (eigenvalues, eigenvectors) — compute once, reuse for all t/β.
    """
    eigenvalues, eigenvectors = scipy.linalg.eigh(L)
    # Clip tiny negative eigenvalues (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# NDM propagation
# ---------------------------------------------------------------------------

def ndm_propagate(
    x0: np.ndarray,
    t: float,
    beta: float,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """
    Compute x(t) = exp(-β·L·t)·x(0) using pre-computed eigen-decomposition.

    x(t) = V @ diag(exp(-β·λ_i·t)) @ V^T @ x(0)

    This is O(n^2) after the initial O(n^3) decomposition, vs. O(n^3) per call
    for scipy.linalg.expm.
    """
    decay = np.exp(-beta * eigenvalues * t)
    # x(t) = V · (decay · (V^T · x0))
    x_t = eigenvectors @ (decay * (eigenvectors.T @ x0))
    return x_t
