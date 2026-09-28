from pathlib import Path

import pandas as pd

from oasis2.data.ingest import (
    find_nifti_or_analyze,
    walk_oasis2_directory,
    load_demographic_csv,
    build_subjects_index,
)


def test_find_nifti_or_analyze_finds_hdr_files(tmp_path):
    session_dir = tmp_path / "OAS2_0001_MR1" / "RAW"
    session_dir.mkdir(parents=True)
    (session_dir / "mpr-1.nifti.hdr").touch()
    (session_dir / "mpr-1.nifti.img").touch()
    found = find_nifti_or_analyze(session_dir)
    assert any(p.suffix == ".hdr" for p in found)


def test_walk_oasis2_directory_discovers_sessions(tmp_path):
    for subj, visit in [("0001", "MR1"), ("0001", "MR2"), ("0002", "MR1")]:
        d = tmp_path / f"OAS2_{subj}_{visit}" / "RAW"
        d.mkdir(parents=True)
        (d / "mpr-1.nifti.hdr").touch()
        (d / "mpr-1.nifti.img").touch()
    records = walk_oasis2_directory(tmp_path)
    assert len(records) == 3


def test_load_demographic_csv_reads_expected_columns(tmp_path):
    import pandas as pd
    p = tmp_path / "demo.csv"
    pd.DataFrame({"Subject ID": ["OAS2_0001"], "MRI ID": ["OAS2_0001_MR1"], "Age": [75]}).to_csv(p, index=False)
    df = load_demographic_csv(p)
    assert "Subject ID" in df.columns
    assert len(df) == 1


def test_build_subjects_index_does_not_explode_multivisit_rows(tmp_path):
    """Regression test (Task 13 follow-up): a subject_id-only merge between
    scan_df (one row per session) and demo_df (one row per visit) produces a
    cross-product and mis-pairs sessions with the wrong visit's demographics.
    With 2 scan sessions and 2 demographic visit rows for the same subject,
    the merged index must have exactly 2 rows (not 4), and each session must
    be paired with its OWN visit's demographics, not the other visit's.
    """
    raw_dir = tmp_path / "raw"
    for visit in ("MR1", "MR2"):
        d = raw_dir / f"OAS2_0001_{visit}" / "RAW"
        d.mkdir(parents=True)
        (d / "mpr-1.nifti.hdr").touch()
        (d / "mpr-1.nifti.img").touch()

    demo_csv = tmp_path / "oasis_longitudinal.csv"
    pd.DataFrame(
        {
            "Subject ID": ["OAS2_0001", "OAS2_0001"],
            "MRI ID": ["OAS2_0001_MR1", "OAS2_0001_MR2"],
            "Group": ["Nondemented", "Nondemented"],
            "Visit": [1, 2],
            "MR Delay": [0, 400],
            "M/F": ["F", "F"],
            "Age": [70, 71],
            "EDUC": [16, 16],
            "SES": [2, 2],
            "MMSE": [29, 15],  # deliberately distinct per visit
            "CDR": [0.0, 0.0],
            "eTIV": [1500, 1500],
            "nWBV": [0.75, 0.74],
            "ASF": [1.0, 1.0],
        }
    ).to_csv(demo_csv, index=False)

    out_path = tmp_path / "subjects_index.csv"
    index_df, _ = build_subjects_index(raw_dir, demo_csv=demo_csv, out_path=out_path)

    # No row explosion: 2 scan sessions in, 2 rows out (not 2x2=4).
    assert len(index_df) == 2

    mr1_row = index_df[index_df["session_id"] == "MR1"].iloc[0]
    mr2_row = index_df[index_df["session_id"] == "MR2"].iloc[0]

    # Each session must be paired with its OWN visit's demographics.
    assert mr1_row["MMSE"] == 29
    assert mr2_row["MMSE"] == 15
    assert mr1_row["Age"] == 70
    assert mr2_row["Age"] == 71
