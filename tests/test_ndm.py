import numpy as np
import pandas as pd
import pytest

from oasis2 import connectome, ndm


def test_compute_w_scores_adds_wscore_columns(synthetic_regional_volumes, synthetic_demo_df):
    w = ndm.compute_w_scores(synthetic_regional_volumes, synthetic_demo_df)
    wscore_cols = [c for c in w.columns if c.endswith("_wscore")]
    assert len(wscore_cols) > 0
    assert not w[wscore_cols].isna().all().all()


def test_run_phase3_end_to_end_on_synthetic_data(tmp_path, monkeypatch, synthetic_regional_volumes, synthetic_demo_df):
    monkeypatch.setenv("OASIS2_PROJECT_ROOT", str(tmp_path))
    result = ndm.run_phase3(
        regional_volumes=synthetic_regional_volumes,
        demo_df=synthetic_demo_df,
        out_dir=tmp_path / "phase3",
    )
    assert "eigenvalues" in result and "eigenvectors" in result
    assert "w_scores" in result
    assert "beta_df" in result
    assert "validation" in result
    # regression check: must not raise NameError from connectome.eigen_decompose_laplacian
    assert result["eigenvalues"].shape[0] == result["eigenvectors"].shape[0]


def test_run_phase3_handles_empty_regional_volumes_gracefully(tmp_path, monkeypatch, synthetic_demo_df):
    monkeypatch.setenv("OASIS2_PROJECT_ROOT", str(tmp_path))
    empty = pd.DataFrame({"subject_id": synthetic_demo_df["subject_id"].unique()})
    result = ndm.run_phase3(regional_volumes=empty, demo_df=synthetic_demo_df, out_dir=tmp_path / "phase3")
    assert result["beta_df"].empty
    assert result["validation"]["n_subjects_validated"] == 0


def test_fit_beta_optimizer_recovers_known_beta():
    """Exercises the actual scipy.optimize.minimize_scalar path inside fit_beta.

    Uses a connectome region count (8) that matches the W-score vector length,
    so the dimension-mismatch NaN-guard does NOT fire and the optimizer's core
    math is actually reached. We forward-simulate x_t from a known true beta
    via ndm_propagate, then check fit_beta recovers it — a real correctness
    check on the optimizer, not just "returned something finite".
    """
    n_regions = 8
    A = connectome.build_synthetic_connectome(n_regions=n_regions, seed=7)
    L = connectome.compute_laplacian(A)
    eigenvalues, eigenvectors = connectome.eigen_decompose_laplacian(L)

    rng = np.random.default_rng(3)
    x0 = rng.uniform(0.2, 1.0, size=n_regions)
    true_beta = 1.5
    t = 2.0
    x_t = ndm.ndm_propagate(x0, t, true_beta, eigenvalues, eigenvectors)

    fitted_beta = ndm.fit_beta(x0, x_t, t, eigenvalues, eigenvectors)

    assert isinstance(fitted_beta, float)
    assert not np.isnan(fitted_beta)
    lo, hi = (0.001, 10.0)
    assert lo <= fitted_beta <= hi
    # noiseless recovery problem: optimizer should land close to the true beta
    assert abs(fitted_beta - true_beta) < 0.05


def test_fit_beta_region_count_mismatch_returns_nan():
    """Directly exercises the dimension-mismatch NaN-fallback guard in fit_beta.

    Uses the *default* (56-region) synthetic connectome's eigendecomposition
    against an 8-region W-score vector, i.e. the exact fixture mismatch the
    reviewer identified — but tested deliberately/directly rather than as an
    accidental side effect of run_phase3's fixture defaults.
    """
    n_regions_wscores = 8
    rng = np.random.default_rng(4)
    x0 = rng.uniform(0.2, 1.0, size=n_regions_wscores)
    x_t = rng.uniform(0.2, 1.0, size=n_regions_wscores)

    A_mismatched = connectome.build_synthetic_connectome()  # default N_REGIONS=56
    L = connectome.compute_laplacian(A_mismatched)
    eigenvalues, eigenvectors = connectome.eigen_decompose_laplacian(L)
    assert eigenvectors.shape[0] != n_regions_wscores  # sanity check on the mismatch

    result = ndm.fit_beta(x0, x_t, 1.0, eigenvalues, eigenvectors)
    assert np.isnan(result)
