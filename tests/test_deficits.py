import pandas as pd

from oasis2 import ndm, deficits


def test_run_phase4_on_synthetic_ndm_results(tmp_path, monkeypatch, synthetic_regional_volumes, synthetic_demo_df):
    monkeypatch.setenv("OASIS2_PROJECT_ROOT", str(tmp_path))
    p3 = ndm.run_phase3(regional_volumes=synthetic_regional_volumes, demo_df=synthetic_demo_df, out_dir=tmp_path / "phase3")
    p3["region_names"] = [f"region_{i}" for i in range(p3["eigenvalues"].shape[0])]
    result = deficits.run_phase4(ndm_results=p3, demo_df=synthetic_demo_df, out_dir=tmp_path / "phase4")
    assert "deficit_scores" in result
    assert isinstance(result["deficit_scores"], pd.DataFrame)


def test_get_domain_weights_returns_nonempty_dict():
    weights = deficits.get_domain_weights("Left-Hippocampus")
    assert isinstance(weights, dict)
    assert len(weights) > 0


def test_region_domain_map_is_nonempty_dict():
    assert isinstance(deficits.REGION_DOMAIN_MAP, dict)
    assert len(deficits.REGION_DOMAIN_MAP) > 0
