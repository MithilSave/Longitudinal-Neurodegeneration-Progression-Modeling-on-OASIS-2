import numpy as np
import pytest

from oasis2 import connectome


def test_build_synthetic_connectome_shape_and_symmetry():
    A = connectome.build_synthetic_connectome(n_regions=10, seed=1)
    assert A.shape == (10, 10)
    assert np.all(np.diag(A) == 0)
    assert np.all(A >= 0)


def test_load_template_connectome_falls_back_to_synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv("OASIS2_PROJECT_ROOT", str(tmp_path))
    A, region_names = connectome.load_template_connectome(connectome_path=None)
    assert A.shape[0] == A.shape[1] == len(region_names)
    # falls back to N_REGIONS by default
    assert A.shape[0] == connectome.N_REGIONS


def test_compute_laplacian_is_symmetric_and_psd(synthetic_connectome):
    L = connectome.compute_laplacian(synthetic_connectome)
    assert np.allclose(L, L.T)
    eigvals = np.linalg.eigvalsh(L)
    assert eigvals.min() > -1e-8  # positive semi-definite


def test_eigen_decompose_laplacian_does_not_raise_nameerror(synthetic_connectome):
    """Regression test for the missing `import scipy.linalg` bug."""
    L = connectome.compute_laplacian(synthetic_connectome)
    eigenvalues, eigenvectors = connectome.eigen_decompose_laplacian(L)
    assert eigenvalues.shape == (8,)
    assert eigenvectors.shape == (8, 8)
    assert np.all(eigenvalues >= 0)


def test_ndm_propagate_identity_at_t_zero(synthetic_connectome):
    L = connectome.compute_laplacian(synthetic_connectome)
    eigenvalues, eigenvectors = connectome.eigen_decompose_laplacian(L)
    x0 = np.random.default_rng(0).random(8)
    x_t0 = connectome.ndm_propagate(x0, t=0.0, beta=0.5, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
    assert np.allclose(x_t0, x0, atol=1e-6)


def test_ndm_propagate_decays_toward_uniform(synthetic_connectome):
    """As t -> large, diffusion should shrink the spread of x(t) (high-freq eigenmodes decay)."""
    L = connectome.compute_laplacian(synthetic_connectome)
    eigenvalues, eigenvectors = connectome.eigen_decompose_laplacian(L)
    x0 = np.array([5.0, -5.0, 3.0, -3.0, 1.0, -1.0, 0.5, -0.5])
    x_t_small = connectome.ndm_propagate(x0, t=0.1, beta=0.5, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
    x_t_large = connectome.ndm_propagate(x0, t=50.0, beta=0.5, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
    assert x_t_large.std() < x_t_small.std()
