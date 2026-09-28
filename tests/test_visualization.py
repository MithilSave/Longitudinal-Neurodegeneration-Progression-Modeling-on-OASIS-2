import matplotlib
matplotlib.use("Agg")  # no display in CI/test environments

import numpy as np
import pandas as pd

from oasis2 import ndm, visualization


def test_plot_phase3_ndm_results_on_synthetic_phase3_output(
    tmp_path, monkeypatch, synthetic_regional_volumes, synthetic_demo_df
):
    """Fixture-chaining test (same pattern as tests/test_ndm.py and
    tests/test_deficits.py): runs the real Phase 3 pipeline on the synthetic
    fixtures to get real beta_df/w_scores/eigenvalues/eigenvectors, then feeds
    them into plot_phase3_ndm_results.

    The synthetic fixtures use an 8-region connectome (see conftest.py
    N_REGIONS) while run_phase3's default template connectome has a different
    region count, so fit_all_betas may produce a beta_df that is non-empty but
    has no non-NaN beta values in this environment. We check which case we
    landed in and assert the behavior that matches it, rather than assuming
    one outcome.
    """
    monkeypatch.setenv("OASIS2_PROJECT_ROOT", str(tmp_path))
    result = ndm.run_phase3(
        regional_volumes=synthetic_regional_volumes,
        demo_df=synthetic_demo_df,
        out_dir=tmp_path / "phase3",
    )
    beta_df = result["beta_df"]
    out = visualization.plot_phase3_ndm_results(
        beta_df, result["w_scores"], result["eigenvalues"], result["eigenvectors"],
        tmp_path / "viz",
    )
    has_valid_beta = (
        not beta_df.empty
        and "beta" in beta_df.columns
        and beta_df["beta"].notna().any()
    )
    if has_valid_beta:
        assert out is not None
        assert out.exists()
    else:
        assert out is None


def test_plot_phase3_ndm_results_writes_png_with_valid_beta(tmp_path):
    """Hand-built success-path test: a small, internally consistent beta_df /
    w_scores / eigendecomposition (all keyed to the same 3-region space) so
    the plotting function actually reaches its ndm_propagate / savefig path,
    independent of whether the chained-fixture test above lands in the
    all-NaN branch in this environment.
    """
    eigenvalues = np.array([0.1, 0.5, 1.0])
    eigenvectors = np.eye(3)
    beta_df = pd.DataFrame({
        "subject_id": ["OAS2_0001", "OAS2_0002"],
        "beta": [0.5, 1.2],
    })
    w_scores = pd.DataFrame({
        "subject_id": ["OAS2_0001", "OAS2_0002"],
        "region_0_wscore": [0.1, 0.2],
        "region_1_wscore": [0.3, 0.4],
        "region_2_wscore": [0.5, 0.6],
    })
    out = visualization.plot_phase3_ndm_results(beta_df, w_scores, eigenvalues, eigenvectors, tmp_path)
    assert out is not None
    assert out.exists()


def test_plot_phase0_overview_writes_png(tmp_path, synthetic_demo_df):
    out = visualization.plot_phase0_overview(synthetic_demo_df, tmp_path)
    assert out.exists()
    assert out.suffix == ".png"


def test_plot_phase4_deficit_profiles_writes_png(tmp_path):
    import pandas as pd
    import numpy as np
    deficit_df = pd.DataFrame(
        np.random.default_rng(0).random((10, 4)),
        columns=["memory", "language", "executive", "visuospatial"],
        index=[f"OAS2_{i:04d}" for i in range(10)],
    )
    out = visualization.plot_phase4_deficit_profiles(deficit_df, tmp_path)
    assert out.exists()


def test_plot_phase4_deficit_profiles_handles_empty_df(tmp_path):
    import pandas as pd
    out = visualization.plot_phase4_deficit_profiles(pd.DataFrame(), tmp_path)
    assert out is None
