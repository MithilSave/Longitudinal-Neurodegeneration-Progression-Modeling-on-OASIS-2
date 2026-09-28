"""Data ingestion (Phase 0 audit + Phase 1 prep) for the OASIS-2 pipeline.

Ported from OASIS2_Colab.py (lines 279-620): `find_demographic_csv`,
`load_demographic_csv`, `find_nifti_or_analyze`, `walk_oasis2_directory`,
`_process_session`, `convert_analyze_to_nifti`, `batch_convert_analyze`,
`data_quality_report`, `build_subjects_index`.

Responsibilities:
  1. Walk the OASIS-2 download directory and discover subject/session structure.
  2. Load the demographic CSV (oasis_longitudinal.csv or equivalent).
  3. Produce subjects_index.csv: one row per (subject, visit) with file paths
     and demographics merged in, exclusions logged.
  4. Analyze->NIfTI format conversion (nibabel).
  5. Data-quality report: missing values, single-visit subjects, resolution checks.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from oasis2.config import get_outputs_dir
from oasis2.logging_utils import setup_logging

logger = setup_logging("ingest")

# ---------------------------------------------------------------------------
# Constants — expected column names in oasis_longitudinal.csv
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "Subject ID", "MRI ID", "Group", "Visit", "MR Delay",
    "M/F", "Age", "EDUC", "SES", "MMSE", "CDR",
    "eTIV", "nWBV", "ASF",
]

GROUP_VALUES = {"Nondemented", "Demented", "Converted"}


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------

def find_demographic_csv(raw_dir: Path) -> Optional[Path]:
    """
    Search `raw_dir` (recursively) for the OASIS-2 demographics spreadsheet.
    Tries 'oasis_longitudinal*.csv' first, then '*.csv', then the same for
    .xlsx/.xls (the Kaggle mirror of this dataset ships the demographics as
    an Excel file rather than a CSV).
    """
    patterns = [
        "oasis_longitudinal*.csv", "*.csv",
        "oasis_longitudinal*.xlsx", "*.xlsx",
        "oasis_longitudinal*.xls", "*.xls",
    ]
    for pat in patterns:
        candidates = sorted(raw_dir.rglob(pat))
        if candidates:
            logger.info(f"Found demographic spreadsheet ({pat}): {candidates[0]}")
            return candidates[0]
    return None


def load_demographic_csv(csv_path: Path) -> pd.DataFrame:
    """Load and validate the OASIS-2 demographic spreadsheet (CSV or Excel)."""
    csv_path = Path(csv_path)
    if csv_path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows × {len(df.columns)} cols from {csv_path.name}")

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning(f"Expected columns not found: {missing_cols}")
    else:
        logger.info("All expected columns present.")

    # Validate Group values
    if "Group" in df.columns:
        unexpected_groups = set(df["Group"].dropna().unique()) - GROUP_VALUES
        if unexpected_groups:
            logger.warning(f"Unexpected Group values: {unexpected_groups}")
        else:
            logger.info(f"Group values: {df['Group'].value_counts().to_dict()}")

    return df


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def find_nifti_or_analyze(session_dir: Path) -> List[Path]:
    """
    Return NIfTI (.nii, .nii.gz) or Analyze (.hdr) volumes found in a session directory.
    Prefer NIfTI; fall back to Analyze headers.
    """
    niis = list(session_dir.rglob("*.nii.gz")) + list(session_dir.rglob("*.nii"))
    if niis:
        return niis
    hdrs = list(session_dir.rglob("*.hdr"))
    return hdrs


def walk_oasis2_directory(raw_dir: Path) -> List[Dict]:
    """
    Walk the OASIS-2 download directory and return a list of dicts, one per session,
    each containing subject_id, session_id, and scan paths.

    OASIS-2 directory layout (from oasis-brains.org distribution):
        OAS2_XXXX_MR1/
          RAW/
          PROCESSED/MPRAGE_*/mpr_n*/
          FSL_SEG/
    """
    records = []

    # Match folders like OAS2_0001_MR1
    subject_session_re = re.compile(r"(OAS2_\d+)_(MR\d+)", re.IGNORECASE)

    entries = sorted(raw_dir.iterdir())
    for entry in tqdm(entries, desc="Scanning raw data", unit="dir"):
        if not entry.is_dir():
            continue
        m = subject_session_re.search(entry.name)
        if not m:
            # Try one level deeper (some Kaggle mirrors add an extra folder)
            for sub in sorted(entry.iterdir()):
                if not sub.is_dir():
                    continue
                m2 = subject_session_re.search(sub.name)
                if m2:
                    _process_session(sub, m2.group(1), m2.group(2), records)
        else:
            _process_session(entry, m.group(1), m.group(2), records)

    if not records:
        logger.warning(
            "No OASIS-2 subject/session directories found in raw_dir. "
            "Check that the download extracted correctly."
        )
    else:
        logger.info(f"Found {len(records)} sessions across the raw directory.")

    return records


def _process_session(
    session_dir: Path,
    subject_id: str,
    session_id: str,
    records: List[Dict],
) -> None:
    scan_files = find_nifti_or_analyze(session_dir)
    fsl_seg = session_dir / "FSL_SEG"
    processed = session_dir / "PROCESSED"

    records.append(
        {
            "subject_id": subject_id.upper(),
            "session_id": session_id.upper(),
            "session_dir": str(session_dir),
            "n_scans": len(scan_files),
            "scan_files": ";".join(str(p) for p in scan_files),
            "has_fsl_seg": fsl_seg.exists(),
            "has_processed": processed.exists(),
        }
    )


# ---------------------------------------------------------------------------
# Analyze → NIfTI conversion
# ---------------------------------------------------------------------------

def convert_analyze_to_nifti(hdr_path: Path, out_dir: Optional[Path] = None) -> Path:
    """
    Load an Analyze (.hdr/.img) volume with nibabel and re-save as NIfTI (.nii.gz).
    Returns the output path.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required for format conversion: pip install nibabel")

    img = nib.load(str(hdr_path))
    nifti_img = nib.Nifti1Image(img.get_fdata(), img.affine, img.header)

    if out_dir is None:
        out_dir = hdr_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = hdr_path.stem
    out_path = out_dir / f"{stem}.nii.gz"
    nib.save(nifti_img, str(out_path))
    logger.debug(f"Converted {hdr_path.name} → {out_path.name}")
    return out_path


def batch_convert_analyze(raw_dir: Path, out_dir: Optional[Path] = None) -> List[Path]:
    """
    Find all .hdr files in raw_dir and convert them to .nii.gz.
    Skips files that already have a NIfTI counterpart.
    """
    hdr_files = list(raw_dir.rglob("*.hdr"))
    if not hdr_files:
        logger.info("No .hdr files found — no Analyze conversion needed.")
        return []

    logger.info(f"Converting {len(hdr_files)} Analyze files to NIfTI ...")
    converted = []
    for hdr in hdr_files:
        nii_candidate = hdr.with_suffix(".nii.gz")
        if nii_candidate.exists():
            continue
        try:
            out = convert_analyze_to_nifti(hdr, out_dir)
            converted.append(out)
        except Exception as e:
            logger.warning(f"Failed to convert {hdr.name}: {e}")

    logger.info(f"Conversion complete: {len(converted)} new NIfTI files.")
    return converted


# ---------------------------------------------------------------------------
# Data quality report
# ---------------------------------------------------------------------------

def data_quality_report(df: pd.DataFrame, scan_records: List[Dict]) -> Dict:
    """
    Compute a data-quality report dict covering:
    - Missing values per column
    - Single-visit subjects (unusable for longitudinal steps)
    - Group distribution
    - CDR distribution
    """
    report = {}

    # Missing values
    if not df.empty:
        missing = df.isnull().sum()
        report["missing_values"] = missing[missing > 0].to_dict()

        # Subjects with only one visit
        if "Subject ID" in df.columns:
            visit_counts = df.groupby("Subject ID").size()
            single_visit = visit_counts[visit_counts == 1].index.tolist()
            report["single_visit_subjects"] = single_visit
            report["n_single_visit"] = len(single_visit)
            report["n_multi_visit"] = int((visit_counts > 1).sum())
        else:
            report["single_visit_subjects"] = []

        # Group distribution
        if "Group" in df.columns:
            report["group_distribution"] = df["Group"].value_counts().to_dict()

        # CDR distribution
        if "CDR" in df.columns:
            report["cdr_distribution"] = df["CDR"].value_counts().to_dict()

        # Summary stats for numeric columns
        numeric_cols = ["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV"]
        for col in numeric_cols:
            if col in df.columns:
                report.setdefault("column_stats", {})[col] = {
                    "mean": round(float(df[col].mean()), 3),
                    "std": round(float(df[col].std()), 3),
                    "missing": int(df[col].isnull().sum()),
                }

    # Scan file counts
    report["n_sessions_found_on_disk"] = len(scan_records)
    report["sessions_with_fsl_seg"] = sum(1 for r in scan_records if r.get("has_fsl_seg"))

    return report


# ---------------------------------------------------------------------------
# subjects_index.csv builder
# ---------------------------------------------------------------------------

def build_subjects_index(
    raw_dir: Path,
    demo_csv: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main entry point for Phase 0.

    Returns:
        (index_df, quality_report)
        index_df: one row per (subject_id, session_id), file paths + demographics merged.
        quality_report: dict with data quality metrics.
    """
    # Load demographics
    if demo_csv is None:
        demo_csv = find_demographic_csv(raw_dir)

    demo_df = pd.DataFrame()
    if demo_csv is not None and demo_csv.exists():
        demo_df = load_demographic_csv(demo_csv)
    else:
        logger.warning("Demographic CSV not found — index will contain scan paths only.")

    # Walk directory
    scan_records = walk_oasis2_directory(raw_dir)
    scan_df = pd.DataFrame(scan_records)

    # Merge
    if not scan_df.empty and not demo_df.empty:
        # Normalise subject ID for merge
        demo_df["subject_id"] = demo_df["Subject ID"].str.upper().str.strip()

        # NOTE (bugfix, Task 13 follow-up): demo_df is OASIS-2's longitudinal
        # demographics table — one row per (subject, visit) — while scan_df is
        # one row per (subject, session) discovered on disk. Merging on
        # subject_id alone produces a full cross-product per subject (a
        # subject with 2 scan sessions and 2 demographic visits yields 4
        # merged rows instead of 2) and pairs each scan session with the
        # wrong visit's demographics roughly half the time. demo_df's
        # "MRI ID" column already encodes the compound key in the form
        # f"{subject_id}_{session_id}" (e.g. "OAS2_0001_MR1"), matching
        # scan_df's subject_id + session_id, so build the same compound key
        # on both sides and merge on that instead. Same class of bug/fix as
        # `compute_w_scores` in ndm.py (Task 4).
        if "MRI ID" in demo_df.columns:
            demo_df["_merge_key"] = demo_df["MRI ID"].astype(str).str.upper().str.strip()
            scan_df = scan_df.copy()
            scan_df["_merge_key"] = (
                scan_df["subject_id"].astype(str).str.upper().str.strip()
                + "_"
                + scan_df["session_id"].astype(str).str.upper().str.strip()
            )
            demo_merge_cols = [c for c in demo_df.columns if c != "subject_id"]
            index_df = scan_df.merge(demo_df[demo_merge_cols], on="_merge_key", how="left")
            index_df = index_df.drop(columns=["_merge_key"])
        else:
            logger.warning(
                "Demographic CSV has no 'MRI ID' column to build a per-visit "
                "merge key — falling back to a subject_id-only merge, which "
                "will duplicate rows and mis-pair visits for multi-visit "
                "subjects."
            )
            index_df = scan_df.merge(demo_df, on="subject_id", how="left")
    elif not scan_df.empty:
        index_df = scan_df
    elif not demo_df.empty:
        logger.warning(
            "No scan directories found on disk; index will be demographics-only."
        )
        demo_df["subject_id"] = demo_df["Subject ID"].str.upper().str.strip()
        index_df = demo_df.copy()
    else:
        index_df = pd.DataFrame()

    # Quality report
    quality = data_quality_report(demo_df, scan_records)

    # Log single-visit exclusions
    if quality.get("single_visit_subjects"):
        n = quality["n_single_visit"]
        logger.info(
            f"{n} single-visit subjects identified — these will be excluded from "
            "longitudinal phases (NDM, Phase 6)."
        )

    # Save
    if out_path is None:
        out_path = get_outputs_dir() / "subjects_index.csv"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not index_df.empty:
        index_df.to_csv(out_path, index=False)
        logger.info(f"subjects_index.csv written to: {out_path}")
    else:
        logger.warning("Index is empty — nothing to save.")

    return index_df, quality
