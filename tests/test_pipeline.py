from oasis2 import pipeline


def test_run_pipeline_end_to_end_on_synthetic_data(tmp_path, monkeypatch, synthetic_demo_df, synthetic_regional_volumes, synthetic_connectome):
    monkeypatch.setenv("OASIS2_PROJECT_ROOT", str(tmp_path))
    import numpy as np
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    np.save(str(tmp_path / "data" / "processed" / "template_connectome.npy"), synthetic_connectome)

    result = pipeline.run_pipeline(
        demo_df=synthetic_demo_df,
        regional_volumes=synthetic_regional_volumes,
        outputs_dir=tmp_path / "outputs",
        dl_epochs=1,  # keep the integration test fast
    )

    assert "phase3" in result and "phase4" in result
    assert "benchmark" in result
    assert "digital_twin" in result
    rollout = result["digital_twin"]["rollout"]
    assert rollout is not None
    bands = result["digital_twin"]["confidence_bands"]
    assert bands is not None
    mean_pred, lower, upper = bands
    assert (upper >= lower).all()
