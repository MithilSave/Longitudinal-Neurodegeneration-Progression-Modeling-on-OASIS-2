from pathlib import Path

from oasis2.data.ingest import find_nifti_or_analyze, walk_oasis2_directory, load_demographic_csv


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
