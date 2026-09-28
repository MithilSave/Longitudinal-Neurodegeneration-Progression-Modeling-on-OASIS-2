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
