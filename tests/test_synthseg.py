import pandas as pd

from oasis2.data.synthseg import (
    synthseg_available,
    extract_fsl_seg_volumes,
    aggregate_volumes,
)


def test_synthseg_available_returns_bool():
    assert isinstance(synthseg_available(), bool)


def test_extract_fsl_seg_volumes_on_missing_dir_returns_empty(tmp_path):
    result = extract_fsl_seg_volumes(tmp_path / "does_not_exist")
    assert result == {}


def test_aggregate_volumes_does_not_explode_multivisit_rows(tmp_path):
    """Regression test (Task 13 follow-up): demo_df is one row per visit
    while `volumes` is one row per session. A subject_id-only merge produces
    a row explosion and mis-pairs eTIV across visits for multi-visit
    subjects. With 2 sessions and 2 demographic visit rows for the same
    subject, the merged frame must have exactly 2 rows (not 4), and each
    session's eTIV normalization must use its OWN visit's eTIV.
    """
    vol_csv_mr1 = tmp_path / "mr1_volumes.csv"
    vol_csv_mr2 = tmp_path / "mr2_volumes.csv"
    pd.DataFrame({"region_a": [300.0], "region_b": [150.0]}).to_csv(vol_csv_mr1, index=False)
    pd.DataFrame({"region_a": [310.0], "region_b": [155.0]}).to_csv(vol_csv_mr2, index=False)

    batch_results = pd.DataFrame(
        [
            {"subject_id": "OAS2_0001", "session_id": "MR1", "vol_csv": str(vol_csv_mr1)},
            {"subject_id": "OAS2_0001", "session_id": "MR2", "vol_csv": str(vol_csv_mr2)},
        ]
    )

    demo_df = pd.DataFrame(
        {
            "Subject ID": ["OAS2_0001", "OAS2_0001"],
            "MRI ID": ["OAS2_0001_MR1", "OAS2_0001_MR2"],
            "eTIV": [1500.0, 1600.0],  # deliberately distinct per visit
        }
    )

    volumes = aggregate_volumes(batch_results, demo_df=demo_df)

    # No row explosion: 2 sessions in, 2 rows out (not 2x2=4).
    assert len(volumes) == 2

    mr1_row = volumes[volumes["session_id"] == "MR1"].iloc[0]
    mr2_row = volumes[volumes["session_id"] == "MR2"].iloc[0]

    # Each session's eTIV must come from its OWN visit, not the other visit's.
    assert mr1_row["eTIV"] == 1500.0
    assert mr2_row["eTIV"] == 1600.0

    # Normalization must use the correctly-paired eTIV.
    assert mr1_row["region_a_norm"] == 300.0 / 1500.0
    assert mr2_row["region_a_norm"] == 310.0 / 1600.0
