import subprocess, sys, os
from pathlib import Path

try:
    gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if gpu.returncode != 0:
        print('[WARNING] No GPU detected!')
        print('  Go to: Runtime > Change runtime type > T4 GPU')
        print('  Phase 1 (SynthSeg) will be ~12x slower on CPU.')
    else:
        lines = [l for l in gpu.stdout.split('\n') if 'GeForce' in l or 'Tesla' in l or 'T4' in l or 'A100' in l]
        print('[OK] GPU:', lines[0].strip() if lines else 'detected')
except FileNotFoundError:
    print('[WARNING] nvidia-smi not found. No GPU detected!')
    print('  Go to: Runtime > Change runtime type > T4 GPU')
    print('  Phase 1 (SynthSeg) will be ~12x slower on CPU.')

# ── 0.2  Mount Google Drive ─────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# Persistent storage (survives disconnects)
DRIVE_BASE = Path('/content/drive/MyDrive/oasis2_pipeline')
DRIVE_BASE.mkdir(parents=True, exist_ok=True)

# Fast local NVMe (ephemeral — recreated each session)
LOCAL_BASE = Path('/content/oasis2')
for sub in ['data/raw', 'data/processed', 'outputs', 'models']:
    (LOCAL_BASE / sub).mkdir(parents=True, exist_ok=True)

print('[OK] Drive mounted at:', DRIVE_BASE)
print('[OK] Local workspace: ', LOCAL_BASE)

# ── 0.3  Install dependencies ────────────────────────────────────────
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'nibabel', 'nilearn', 'SimpleITK', 'networkx',
                'statsmodels', 'kaggle', 'tqdm', 'openpyxl'], check=True)
print('[OK] Dependencies installed.')
print('     (torch, numpy, pandas, sklearn, matplotlib already in Colab)')

"""
utils.py — Shared utilities for the OASIS-2 neurodegeneration pipeline.

Covers:
  - GPU/CPU device detection
  - Logging setup
  - Checkpoint save/load
  - Path helpers
  - Environment detection (Colab vs. local)
"""

import os
import sys
import json
import logging
import pickle
import platform
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

# In a notebook there is no __file__, so we anchor everything to LOCAL_BASE,
# which is defined earlier in this notebook (Section 0, Google Drive mount).
PROJECT_ROOT = LOCAL_BASE


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_data_dir(sub: str = "") -> Path:
    d = PROJECT_ROOT / "data" / sub if sub else PROJECT_ROOT / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_outputs_dir(sub: str = "") -> Path:
    d = PROJECT_ROOT / "outputs" / sub if sub else PROJECT_ROOT / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_models_dir() -> Path:
    d = PROJECT_ROOT / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def is_colab() -> bool:
    """Return True if running in Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def is_gpu_available() -> bool:
    """Return True if a CUDA GPU is accessible."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device():
    """Return a torch.device: cuda if available, else cpu."""
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device
    except ImportError:
        return None


def environment_summary() -> Dict[str, Any]:
    """Return a dict summarising the current runtime environment."""
    info = {
        "platform": platform.system(),
        "python": sys.version,
        "colab": is_colab(),
        "gpu_available": is_gpu_available(),
        "project_root": str(PROJECT_ROOT),
    }
    if is_gpu_available():
        try:
            import torch
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
        except Exception:
            pass
    return info


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    name: str = "oasis2",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure and return a named logger that writes to stdout (and optionally a file)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s  %(name)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(data: Any, name: str, directory: Optional[Path] = None) -> Path:
    """
    Pickle-save `data` to `directory/<name>.pkl`.
    Default directory is outputs/checkpoints/.
    Returns the path written.
    """
    if directory is None:
        directory = get_outputs_dir("checkpoints")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def load_checkpoint(name: str, directory: Optional[Path] = None) -> Any:
    """
    Load and return a previously saved checkpoint.
    Raises FileNotFoundError if the checkpoint does not exist.
    """
    if directory is None:
        directory = get_outputs_dir("checkpoints")
    directory = Path(directory)
    path = directory / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def checkpoint_exists(name: str, directory: Optional[Path] = None) -> bool:
    if directory is None:
        directory = get_outputs_dir("checkpoints")
    return (Path(directory) / f"{name}.pkl").exists()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def save_json(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Numpy helpers
# ---------------------------------------------------------------------------

def z_score(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standardise array along given axis (zero mean, unit variance)."""
    mu = arr.mean(axis=axis, keepdims=True)
    sigma = arr.std(axis=axis, keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (arr - mu) / sigma


# ---------------------------------------------------------------------------
# Colab-specific helpers
# ---------------------------------------------------------------------------

def colab_drive_path(subpath: str = "") -> Optional[Path]:
    """
    Return the path to a project folder on Google Drive when running in Colab.
    Assumes the Drive is mounted at /content/drive/MyDrive.
    """
    if not is_colab():
        return None
    base = Path("/content/drive/MyDrive/oasis2_pipeline")
    base.mkdir(parents=True, exist_ok=True)
    return base / subpath if subpath else base

"""
preprocessing.py — Phase 0 (Data Audit) + Phase 1 prep for the OASIS-2 pipeline.

Responsibilities:
  1. Walk the OASIS-2 download directory and discover subject/session structure.
  2. Load the demographic CSV (oasis_longitudinal.csv or equivalent).
  3. Produce subjects_index.csv: one row per (subject, visit) with file paths
     and demographics merged in, exclusions logged.
  4. Analyze→NIfTI format conversion (nibabel).
  5. Data-quality report: missing values, single-visit subjects, resolution checks.

"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm



logger = setup_logging("preprocessing")

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

"""
segmentation.py — Phase 1: Whole-brain parcellation + regional volume extraction.

Responsibilities:
  1. Run SynthSeg on each T1-weighted NIfTI volume to produce:
       - a parcellation label map  (synthseg_out.nii.gz)
       - a per-region volume CSV   (volumes.csv)
  2. Aggregate per-session volumes into a single regional_volumes.csv.
  3. Normalize volumes by eTIV (provided in demographics).
  4. Fall back to existing FSL_SEG tissue masks where SynthSeg is unavailable.

SynthSeg command used (standalone pip package, no FreeSurfer license needed):
    mri_synthseg --i <t1.nii.gz> --o <out.nii.gz> --vol <volumes.csv> --parc --threads 4

Confirmed runtimes (from Billot et al. 2023):
    ~1-2 min/scan on CPU, ~6-15s/scan on GPU.

"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm



logger = setup_logging("segmentation")


# ---------------------------------------------------------------------------
# SynthSeg availability check
# ---------------------------------------------------------------------------

def synthseg_available() -> bool:
    """Return True if `mri_synthseg` is on PATH (from the standalone pip package)."""
    return shutil.which("mri_synthseg") is not None


def get_synthseg_threads() -> int:
    """Return a reasonable thread count for SynthSeg CPU runs."""
    import multiprocessing
    cores = multiprocessing.cpu_count()
    return min(cores, 8)


# ---------------------------------------------------------------------------
# Run SynthSeg on a single scan
# ---------------------------------------------------------------------------

def run_synthseg_single(
    t1_path: Path,
    out_dir: Path,
    parc: bool = True,
    robust: bool = False,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Run SynthSeg on one T1 NIfTI volume.

    Args:
        t1_path: input T1 volume (.nii or .nii.gz)
        out_dir: directory to write outputs
        parc:    if True, add --parc for cortical parcellation
        robust:  if True, add --robust (slower but handles low-quality scans)

    Returns:
        (seg_path, vol_csv_path) — paths to the label map and volume CSV,
        or (None, None) on failure.
    """
    if not synthseg_available():
        logger.warning("mri_synthseg not found on PATH. Skipping SynthSeg.")
        return None, None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = t1_path.stem.replace(".nii", "")
    seg_path = out_dir / f"{stem}_synthseg.nii.gz"
    vol_csv = out_dir / f"{stem}_volumes.csv"

    cmd = [
        "mri_synthseg",
        "--i", str(t1_path),
        "--o", str(seg_path),
        "--vol", str(vol_csv),
    ]
    if parc:
        cmd.append("--parc")
    if robust:
        cmd.append("--robust")
    if not is_gpu_available():
        cmd += ["--threads", str(get_synthseg_threads()), "--cpu"]

    logger.info(f"Running SynthSeg on: {t1_path.name}")
    logger.debug(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"SynthSeg failed for {t1_path.name}:\n{result.stderr}")
        return None, None

    return seg_path, vol_csv


# ---------------------------------------------------------------------------
# Batch SynthSeg runner
# ---------------------------------------------------------------------------

def run_synthseg_batch(
    subjects_index: pd.DataFrame,
    processed_dir: Optional[Path] = None,
    checkpoint_name: str = "synthseg_batch",
    parc: bool = True,
) -> pd.DataFrame:
    """
    Run SynthSeg on all scans listed in subjects_index.
    Checkpoints after each scan so a partial run is not lost.

    subjects_index must have columns: subject_id, session_id, scan_files
    (scan_files is a semicolon-separated list of paths).

    Returns a DataFrame mapping (subject_id, session_id) → vol_csv_path.
    """
    if processed_dir is None:
        processed_dir = get_data_dir("processed")
    processed_dir = Path(processed_dir)

    # Load existing progress
    if checkpoint_exists(checkpoint_name):
        results = load_checkpoint(checkpoint_name)
        logger.info(f"Resuming from checkpoint: {len(results)} sessions already done.")
    else:
        results = {}

    if "scan_files" not in subjects_index.columns:
        logger.error("subjects_index must have a 'scan_files' column.")
        return pd.DataFrame()

    for _, row in tqdm(subjects_index.iterrows(), total=len(subjects_index),
                        desc="SynthSeg segmentation", unit="scan"):
        subj = row.get("subject_id", "unknown")
        sess = row.get("session_id", "unknown")
        key = f"{subj}_{sess}"

        if key in results:
            continue  # already done

        scan_files_str = row.get("scan_files", "")
        if not scan_files_str:
            logger.warning(f"No scan files for {key}; skipping.")
            results[key] = {"subject_id": subj, "session_id": sess, "vol_csv": None}
            continue

        # Use the first scan file
        t1_path = Path(scan_files_str.split(";")[0])
        out_dir = processed_dir / subj / sess

        _, vol_csv = run_synthseg_single(t1_path, out_dir, parc=parc)

        results[key] = {
            "subject_id": subj,
            "session_id": sess,
            "vol_csv": str(vol_csv) if vol_csv else None,
        }
        save_checkpoint(results, checkpoint_name)

    result_df = pd.DataFrame(list(results.values()))
    return result_df


# ---------------------------------------------------------------------------
# Aggregate per-session volume CSVs into regional_volumes.csv
# ---------------------------------------------------------------------------

def aggregate_volumes(
    batch_results: pd.DataFrame,
    demo_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Load each SynthSeg volume CSV and concatenate into a single DataFrame.
    If demo_df is provided, merges eTIV for normalization.

    Returns regional_volumes DataFrame (one row per session).
    """
    frames = []

    for _, row in batch_results.iterrows():
        vol_csv = row.get("vol_csv")
        if not vol_csv or (isinstance(vol_csv, float) and pd.isna(vol_csv)) or not Path(vol_csv).exists():
            continue
        try:
            df = pd.read_csv(vol_csv)
            df["subject_id"] = row["subject_id"]
            df["session_id"] = row["session_id"]
            frames.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {vol_csv}: {e}")

    if not frames:
        logger.warning("No volume CSVs loaded — regional_volumes will be empty.")
        return pd.DataFrame()

    volumes = pd.concat(frames, ignore_index=True)

    # Normalise by eTIV if available
    if demo_df is not None and "eTIV" in demo_df.columns:
        demo_df = demo_df.copy()
        demo_df["subject_id"] = demo_df.get("Subject ID", demo_df.get("subject_id", "")).str.upper()
        volumes = volumes.merge(demo_df[["subject_id", "eTIV"]], on="subject_id", how="left")
        region_cols = [c for c in volumes.columns
                       if c not in {"subject_id", "session_id", "eTIV"}
                       and pd.api.types.is_numeric_dtype(volumes[c])]
        for col in region_cols:
            volumes[f"{col}_norm"] = volumes[col] / volumes["eTIV"]

    return volumes


# ---------------------------------------------------------------------------
# FSL_SEG fallback (when SynthSeg is unavailable)
# ---------------------------------------------------------------------------

def extract_fsl_seg_volumes(fsl_seg_dir: Path) -> Dict[str, float]:
    """
    Compute GM / WM / CSF volumes from an existing FSL tissue segmentation.
    FSL_SEG typically contains a *_seg.nii.gz with labels: 1=CSF, 2=GM, 3=WM.

    Returns a dict: {tissue_name: voxel_count}.
    NOTE: This is coarser than SynthSeg (3 tissues vs ~37+ regions).
          Only used when SynthSeg is not available.
    """
    try:
        import nibabel as nib
    except ImportError:
        logger.warning("nibabel not installed; cannot read FSL_SEG.")
        return {}

    seg_files = list(fsl_seg_dir.rglob("*_seg.nii.gz"))
    if not seg_files:
        seg_files = list(fsl_seg_dir.rglob("*.nii.gz"))
    if not seg_files:
        return {}

    seg = nib.load(str(seg_files[0]))
    data = seg.get_fdata().astype(int)

    label_map = {1: "CSF", 2: "GM", 3: "WM"}
    volumes = {}
    for label, name in label_map.items():
        volumes[name] = int((data == label).sum())

    return volumes


# ---------------------------------------------------------------------------
# Top-level: build regional_volumes.csv
# ---------------------------------------------------------------------------

def build_regional_volumes(
    subjects_index: pd.DataFrame,
    demo_df: Optional[pd.DataFrame] = None,
    out_path: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Phase 1 main entry point.

    Steps:
      1. Run SynthSeg batch (or load from checkpoint if partially done).
      2. Aggregate volumes.
      3. Normalise by eTIV.
      4. Save to regional_volumes.csv.
    """
    if not synthseg_available():
        logger.warning(
            "SynthSeg (mri_synthseg) not available.\n"
            "  Install the standalone package:  pip install SynthSeg-standalone\n"
            "  or clone from: https://github.com/freesurfer/SynthSeg\n"
            "  The pipeline will continue with FSL_SEG fallback where available."
        )

    batch_results = run_synthseg_batch(subjects_index, processed_dir=processed_dir)
    volumes = aggregate_volumes(batch_results, demo_df=demo_df)

    if out_path is None:
        out_path = get_outputs_dir() / "regional_volumes.csv"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not volumes.empty:
        volumes.to_csv(out_path, index=False)
        logger.info(f"regional_volumes.csv written to: {out_path}")
    else:
        logger.warning("regional_volumes is empty — no SynthSeg outputs found.")

    return volumes


# ---------------------------------------------------------------------------
# Spot-check helper (acceptance criterion from implementation.md §3)
# ---------------------------------------------------------------------------

def spot_check_synthseg_output(
    t1_path: Path,
    seg_path: Path,
    n_slices: int = 3,
    out_dir: Optional[Path] = None,
) -> None:
    """
    Save PNG overlays of the SynthSeg parcellation on the T1 volume
    for visual QC. Requires nilearn and matplotlib.
    """
    try:
        from nilearn import plotting
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("nilearn/matplotlib not available for spot-check visualization.")
        return

    if out_dir is None:
        out_dir = get_outputs_dir("qc_overlays")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = t1_path.stem.replace(".nii", "")
    out_fig = out_dir / f"{stem}_qc.png"

    display = plotting.plot_roi(
        str(seg_path),
        bg_img=str(t1_path),
        title=f"SynthSeg QC: {stem}",
        display_mode="ortho",
        alpha=0.5,
    )
    display.savefig(str(out_fig))
    display.close()
    logger.info(f"QC overlay saved: {out_fig}")

"""
classical_model.py — Phase 2a: Primary classification track.

Task: Binary classification of Nondemented vs. Demented using:
  - Regional brain volumes from Phase 1 (eTIV-normalized)
  - Clinical/demographic features: Age, Sex, Education, SES

Models: Logistic Regression, Linear SVM, Random Forest, Gradient Boosting.

Key constraints (from Yagis et al. 2021 and Wen et al. 2020):
  - Subject-level split ONLY (GroupKFold keyed on subject_id)
  - No slice-level leakage is possible here (volumetric features, not image slices)
  - Expected honest accuracy: ~60–75% — this is the credible primary result

"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.impute import SimpleImputer



logger = setup_logging("classical_model")

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

CLINICAL_FEATURES = ["Age", "EDUC", "SES", "eTIV", "nWBV"]
SEX_COL = "M/F"  # encode as 0/1


def prepare_features(
    regional_volumes: pd.DataFrame,
    demo_df: pd.DataFrame,
    target_col: str = "CDR",
    region_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Build feature matrix X, label vector y, and group vector (subject IDs) for GroupKFold.

    Labeling strategy: CDR > 0  → Demented (1), CDR == 0 → Nondemented (0).
    Subjects with Group == 'Converted' are included with their CDR-derived label.

    Returns:
        X: (n_samples, n_features) float array
        y: (n_samples,) int array {0, 1}
        groups: (n_samples,) array of subject_id strings (for GroupKFold)
        feature_names: list of feature column names
    """
    # Merge regional volumes with demographics on subject_id
    demo_df = demo_df.copy()
    if "subject_id" not in demo_df.columns and "Subject ID" in demo_df.columns:
        demo_df["subject_id"] = demo_df["Subject ID"].str.upper().str.strip()
    if "subject_id" not in demo_df.columns:
        raise ValueError("demo_df must have 'subject_id' or 'Subject ID' column.")

    # Merge regional volumes with demographics on subject_id.
    # Only bring in demo_df columns that aren't already in regional_volumes —
    # otherwise an overlapping column (e.g. if regional_volumes is itself
    # demographics-derived) gets silently suffixed to CDR_x/CDR_y by pandas
    # and the plain "CDR"/"Group" columns below would vanish.
    demo_cols_to_merge = ["subject_id"] + [
        c for c in demo_df.columns if c != "subject_id" and c not in regional_volumes.columns
    ]
    merged = regional_volumes.merge(demo_df[demo_cols_to_merge], on="subject_id", how="left")

    # Sex encoding
    if SEX_COL in merged.columns:
        merged["sex_encoded"] = (merged[SEX_COL].str.upper() == "M").astype(float)
    else:
        merged["sex_encoded"] = 0.0

    # Region cols: use _norm variants if available, else raw
    if region_cols is None:
        norm_cols = [c for c in regional_volumes.columns if c.endswith("_norm")]
        if norm_cols:
            region_cols = norm_cols
        else:
            region_cols = [
                c for c in regional_volumes.columns
                if c not in {"subject_id", "session_id"} and
                pd.api.types.is_numeric_dtype(regional_volumes[c])
            ]

    clinical = [c for c in CLINICAL_FEATURES if c in merged.columns]
    all_feature_cols = region_cols + clinical + ["sex_encoded"]
    all_feature_cols = [c for c in all_feature_cols if c in merged.columns]

    # Target: CDR-derived binary label
    if target_col not in merged.columns and "CDR" in merged.columns:
        target_col = "CDR"

    if target_col in merged.columns:
        valid = merged[target_col].notna()
        merged = merged[valid].copy()
        y = (merged[target_col] > 0).astype(int).values
    elif "Group" in merged.columns:
        logger.warning("CDR not found; using Group column for labels.")
        group_map = {"Nondemented": 0, "Demented": 1, "Converted": 1}
        merged = merged[merged["Group"].isin(group_map)].copy()
        y = merged["Group"].map(group_map).values
    else:
        raise ValueError("No usable label column found (CDR or Group).")

    X = merged[all_feature_cols].values.astype(float)
    groups = merged["subject_id"].values if "subject_id" in merged.columns else np.arange(len(y))

    logger.info(
        f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features. "
        f"Label balance: {y.sum()} positive / {(y == 0).sum()} negative."
    )
    return X, y, groups, all_feature_cols


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_pipelines() -> Dict[str, Pipeline]:
    """Return a dict of named scikit-learn pipelines (imputer + scaler + model)."""
    return {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "LinearSVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, random_state=42,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Cross-validation (subject-level GroupKFold)
# ---------------------------------------------------------------------------

def evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, Dict]:
    """
    Evaluate all classical models using subject-level GroupKFold CV.
    Returns a dict of {model_name: {metric: value}}.
    """
    pipelines = build_pipelines()

    # Ensure we don't request more splits than there are unique subjects
    n_unique = len(np.unique(groups))
    n_splits = min(n_splits, n_unique)
    gkf = GroupKFold(n_splits=n_splits)

    scoring = ["accuracy", "f1_weighted", "roc_auc"]
    # LinearSVC doesn't support predict_proba; remove roc_auc for it
    svm_scoring = ["accuracy", "f1_weighted"]

    results = {}
    model_bar = tqdm(pipelines.items(), desc="Classical models", unit="model")
    for name, pipe in model_bar:
        model_bar.set_description(f"Evaluating {name}")
        logger.info(f"Evaluating {name} ...")
        sc = svm_scoring if name == "LinearSVM" else scoring
        try:
            cv_results = cross_validate(
                pipe, X, y,
                groups=groups,
                cv=gkf,
                scoring=sc,
                return_train_score=False,
            )
            results[name] = {
                metric: {
                    "mean": round(float(cv_results[f"test_{metric}"].mean()), 4),
                    "std": round(float(cv_results[f"test_{metric}"].std()), 4),
                }
                for metric in sc
            }
            logger.info(
                f"  {name}: accuracy={results[name]['accuracy']['mean']:.3f}"
                f" ± {results[name]['accuracy']['std']:.3f}"
            )
            model_bar.set_postfix(acc=f"{results[name]['accuracy']['mean']:.3f}")
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            results[name] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Train final model on full data and save
# ---------------------------------------------------------------------------

def train_and_save_best_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    best_model_name: str = "GradientBoosting",
    out_dir: Optional[Path] = None,
) -> Pipeline:
    """
    Fit the selected model on all available data and save as a checkpoint.
    Returns the fitted pipeline.
    """
    if out_dir is None:
        out_dir = get_models_dir()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipelines = build_pipelines()
    if best_model_name not in pipelines:
        logger.warning(f"{best_model_name} not found; defaulting to GradientBoosting.")
        best_model_name = "GradientBoosting"

    pipe = pipelines[best_model_name]
    pipe.fit(X, y)

    save_checkpoint(
        {"pipeline": pipe, "feature_names": feature_names},
        f"classical_model_{best_model_name}",
        directory=out_dir,
    )
    logger.info(f"Model saved: models/classical_model_{best_model_name}.pkl")
    return pipe


# ---------------------------------------------------------------------------
# Feature importance report
# ---------------------------------------------------------------------------

def feature_importance_report(
    model: Pipeline,
    feature_names: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract feature importances from the fitted model (if available).
    Works with RandomForest, GradientBoosting (feature_importances_)
    and LogisticRegression (coef_).
    """
    clf = model.named_steps.get("clf")
    if clf is None:
        return pd.DataFrame()

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_).flatten()
    else:
        logger.info("Model does not expose feature importances.")
        return pd.DataFrame()

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_phase2a(
    regional_volumes: pd.DataFrame,
    demo_df: pd.DataFrame,
    out_dir: Optional[Path] = None,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    Phase 2a full run: feature prep → cross-validation → train final model → save.
    Returns a summary dict.
    """
    if out_dir is None:
        out_dir = get_outputs_dir("phase2a")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups, feature_names = prepare_features(regional_volumes, demo_df)
    cv_results = evaluate_models(X, y, groups, n_splits=n_splits)

    # Pick best model by mean accuracy
    best_name = max(
        {k: v for k, v in cv_results.items() if "error" not in v},
        key=lambda k: cv_results[k].get("accuracy", {}).get("mean", 0),
        default="GradientBoosting",
    )
    logger.info(f"Best model by accuracy: {best_name}")

    final_model = train_and_save_best_model(X, y, feature_names, best_model_name=best_name)
    importances = feature_importance_report(final_model, feature_names)

    # Save results
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    if not importances.empty:
        importances.to_csv(out_dir / "feature_importances.csv", index=False)

    logger.info(f"Phase 2a results written to: {out_dir}")

    return {
        "cv_results": cv_results,
        "best_model_name": best_name,
        "feature_names": feature_names,
        "feature_importances": importances,
        "final_model": final_model,
    }

"""
cnn_model.py — Phase 2b: Secondary transfer-learning CNN classification track.

Task: Same binary classification (Nondemented vs. Demented) as Phase 2a,
      but using 2D slices from T1 volumes through a pretrained ImageNet backbone.

Key design decisions (from implementation.md §4b, Yagis et al. 2021):
  - NEVER train from scratch on ~150 subjects.
  - Use a pretrained backbone (ResNet-18 or VGG16) as frozen/lightly-tuned feature extractor.
  - Subject-level split ONLY via GroupKFold — slice-level split inflates accuracy ~30 pp.
  - Expected honest subject-level accuracy: 60–75%.
  - If accuracy >> 75% on your own split, check for leakage before celebrating.

Runtime budget:
  - Colab GPU: minutes per fold.
  - CPU-only: design for < 1 hour (small batches, few epochs, early stopping).

"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm



logger = setup_logging("cnn_model")
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Torch imports (graceful failure if not installed)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.models as tv_models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # fallback base class so class defs below don't NameError
    logger.warning(
        "PyTorch/torchvision not installed. "
        "Phase 2b CNN track will not be available. "
        "Phase 2a (classical ML) is the primary track and does not require torch."
    )


# ---------------------------------------------------------------------------
# Slice extraction
# ---------------------------------------------------------------------------

def extract_informative_slices(
    nifti_path: Path,
    n_slices: int = 10,
    axis: int = 2,  # axial
    method: str = "entropy",
) -> Optional[np.ndarray]:
    """
    Extract the most informative 2D slices from a 3D T1 NIfTI volume.

    Args:
        nifti_path: path to .nii or .nii.gz
        n_slices:   number of slices to extract per volume
        axis:       slicing axis (0=sagittal, 1=coronal, 2=axial)
        method:     'entropy' (highest information slices) or 'uniform' (evenly spaced)

    Returns:
        Array of shape (n_slices, H, W) normalized to [0, 1], or None on failure.
    """
    try:
        import nibabel as nib
    except ImportError:
        logger.warning("nibabel not installed; cannot load NIfTI.")
        return None

    try:
        img = nib.load(str(nifti_path))
        data = img.get_fdata()
    except Exception as e:
        logger.warning(f"Failed to load {nifti_path}: {e}")
        return None

    # Guard against unexpected 4D volumes (e.g. trailing singleton or time dim)
    data = np.squeeze(data)
    if data.ndim != 3:
        logger.warning(f"Skipping {nifti_path}: expected 3D volume, got shape {data.shape}")
        return None

    # Normalize volume to [0, 1]
    vmin, vmax = data.min(), data.max()
    if vmax - vmin < 1e-8:
        return None
    data = (data - vmin) / (vmax - vmin)

    n_total = data.shape[axis]

    if method == "entropy":
        # Compute entropy of each slice to pick informative ones
        from scipy.stats import entropy as scipy_entropy

        entropies = []
        for i in range(n_total):
            sl = np.take(data, i, axis=axis).ravel()
            hist, _ = np.histogram(sl, bins=64, range=(0, 1))
            e = scipy_entropy(hist + 1e-10)
            entropies.append(e)

        top_indices = np.argsort(entropies)[-n_slices:]
    else:  # uniform
        top_indices = np.linspace(n_total // 10, 9 * n_total // 10, n_slices, dtype=int)

    slices = [np.take(data, int(i), axis=axis) for i in sorted(top_indices)]
    return np.stack(slices, axis=0)  # (n_slices, H, W)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class OASISSliceDataset(Dataset):
    """
    PyTorch Dataset for 2D MRI slices.

    Each item returns (image_tensor, label, subject_id).
    image_tensor: (3, H, W) — replicated to 3 channels for ImageNet backbone.
    """

    def __init__(
        self,
        slice_arrays: List[np.ndarray],  # list of (n_slices, H, W) arrays
        labels: List[int],
        subject_ids: List[str],
        transform=None,
    ):
        assert len(slice_arrays) == len(labels) == len(subject_ids)
        self.samples = []
        for arr, label, sid in zip(slice_arrays, labels, subject_ids):
            for sl in arr:
                self.samples.append((sl, label, sid))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sl, label, sid = self.samples[idx]
        # Resize to (224, 224) and replicate to 3 channels
        sl_tensor = torch.tensor(sl, dtype=torch.float32).squeeze()
        if sl_tensor.ndim != 2:
            raise ValueError(
                f"Expected 2D slice for subject {sid}, got shape {tuple(sl_tensor.shape)}"
            )
        sl_3ch = sl_tensor.unsqueeze(0).repeat(3, 1, 1)
        if self.transform:
            sl_3ch = self.transform(sl_3ch)
        return sl_3ch, torch.tensor(label, dtype=torch.long), sid


def get_transform(train: bool = True, img_size: int = 224):
    """Return torchvision transforms for train or eval mode."""
    if not TORCH_AVAILABLE:
        return None
    ops = [transforms.Resize((img_size, img_size))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        ]
    ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]))
    return transforms.Compose(ops)


# ---------------------------------------------------------------------------
# Model: pretrained backbone + small classification head
# ---------------------------------------------------------------------------

def build_transfer_model(
    backbone: str = "resnet18",
    freeze_backbone: bool = True,
    num_classes: int = 2,
) -> "nn.Module":
    """
    Build a pretrained backbone with a fine-tuned classification head.

    Args:
        backbone: 'resnet18' or 'vgg16'
        freeze_backbone: if True, backbone weights are frozen (feature extractor only)
        num_classes: number of output classes

    Returns a PyTorch nn.Module.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for the CNN track.")

    if backbone == "resnet18":
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == "vgg16":
        model = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose 'resnet18' or 'vgg16'.")

    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_fold(
    model: "nn.Module",
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    device,
    n_epochs: int = 10,
    lr: float = 1e-3,
    patience: int = 3,
    fold_label: str = "",
) -> Dict:
    """
    Train the model for one CV fold with early stopping.
    Returns a dict with train/val losses and best val accuracy.
    """
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    epochs_no_improve = 0

    epoch_bar = tqdm(range(n_epochs), desc=f"{fold_label} epochs", unit="epoch", leave=False)
    for epoch in epoch_bar:
        # --- Train ---
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"  train", unit="batch", leave=False)
        for images, labels, _ in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = total_loss / max(len(train_loader), 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"  val", unit="batch", leave=False)
            for images, labels, _ in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc = correct / max(total, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(avg_val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_bar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}",
            val_loss=f"{avg_val_loss:.4f}",
            val_acc=f"{val_acc:.3f}",
            best=f"{best_val_acc:.3f}",
        )
        logger.info(
            f"{fold_label} Epoch {epoch+1}/{n_epochs}: "
            f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  val_acc={val_acc:.3f}"
        )

        if epochs_no_improve >= patience:
            logger.info(f"{fold_label} Early stopping at epoch {epoch+1}.")
            break

    history["best_val_acc"] = best_val_acc
    return history


# ---------------------------------------------------------------------------
# Subject-level GroupKFold CV (the hard requirement)
# ---------------------------------------------------------------------------

def run_cnn_cv(
    nifti_paths: List[Path],
    labels: List[int],
    subject_ids: List[str],
    n_splits: int = 5,
    backbone: str = "resnet18",
    n_epochs: int = 10,
    batch_size: int = 16,
    n_slices_per_scan: int = 10,
    out_dir: Optional[Path] = None,
) -> Dict:
    """
    Run subject-level GroupKFold CV for the transfer-learning CNN.

    IMPORTANT: subject_ids are the GroupKFold groups — a subject's slices
    can NEVER appear in both train and val/test splits.

    Returns a dict with per-fold and aggregate accuracy.
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available. Phase 2b cannot run.")
        return {"error": "PyTorch not installed"}

    from sklearn.model_selection import GroupKFold

    if out_dir is None:
        out_dir = get_outputs_dir("phase2b")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"CNN training on device: {device}  [{gpu_name}, {gpu_mem_gb:.1f} GB]")
        print(f"[GPU] Using {gpu_name} ({gpu_mem_gb:.1f} GB) — training will run on GPU.")
    else:
        logger.info(f"CNN training on device: {device}")
        print(
            "[CPU] No CUDA GPU detected — training will be much slower.\n"
            "      In Colab: Runtime > Change runtime type > Hardware accelerator > T4 GPU, "
            "then Runtime > Restart session."
        )

    # Extract slices for all scans
    logger.info(f"Extracting slices from {len(nifti_paths)} scans ...")
    slice_arrays = []
    valid_idx = []
    for i, path in enumerate(tqdm(nifti_paths, desc="Extracting slices", unit="scan")):
        arr = extract_informative_slices(path, n_slices=n_slices_per_scan)
        if arr is not None:
            slice_arrays.append(arr)
            valid_idx.append(i)

    if not slice_arrays:
        logger.error("No valid slices extracted.")
        return {"error": "No valid slices"}

    valid_labels = [labels[i] for i in valid_idx]
    valid_subjects = [subject_ids[i] for i in valid_idx]

    groups_arr = np.array(valid_subjects)
    labels_arr = np.array(valid_labels)

    n_unique = len(np.unique(groups_arr))
    n_splits = min(n_splits, n_unique)
    gkf = GroupKFold(n_splits=n_splits)

    fold_accs = []
    start_time = time.time()

    fold_bar = tqdm(
        enumerate(gkf.split(slice_arrays, labels_arr, groups_arr)),
        total=n_splits, desc="CV folds", unit="fold",
    )
    for fold_idx, (train_idx, val_idx) in fold_bar:
        fold_label = f"[Fold {fold_idx + 1}/{n_splits}]"
        logger.info(f"  {fold_label} ...")
        fold_bar.set_description(f"CV fold {fold_idx + 1}/{n_splits}")

        # CRITICAL: check no subject overlap
        train_subjs = set(groups_arr[train_idx])
        val_subjs = set(groups_arr[val_idx])
        assert len(train_subjs & val_subjs) == 0, "LEAKAGE DETECTED: subject in both train and val!"

        train_arrs = [slice_arrays[i] for i in train_idx]
        val_arrs = [slice_arrays[i] for i in val_idx]

        train_ds = OASISSliceDataset(
            train_arrs, labels_arr[train_idx].tolist(),
            groups_arr[train_idx].tolist(),
            transform=get_transform(train=True),
        )
        val_ds = OASISSliceDataset(
            val_arrs, labels_arr[val_idx].tolist(),
            groups_arr[val_idx].tolist(),
            transform=get_transform(train=False),
        )

        num_workers = 2 if is_gpu_available() else 0
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=is_gpu_available())
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=is_gpu_available())

        model = build_transfer_model(backbone=backbone, freeze_backbone=True)
        model = model.to(device)

        history = train_one_fold(
            model, train_loader, val_loader, device,
            n_epochs=n_epochs, fold_label=fold_label,
        )
        fold_accs.append(history["best_val_acc"])
        logger.info(f"    {fold_label} best val accuracy: {history['best_val_acc']:.3f}")
        fold_bar.set_postfix(best_val_acc=f"{history['best_val_acc']:.3f}")

    wall_time = time.time() - start_time
    result = {
        "backbone": backbone,
        "fold_accuracies": fold_accs,
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "n_splits": n_splits,
        "wall_time_s": round(wall_time, 1),
        "note": (
            "Subject-level GroupKFold split enforced. "
            "Expected honest range: 60-75%. "
            "Values >> 75% should be investigated for leakage."
        ),
    }

    import json
    with open(out_dir / "cnn_cv_results.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.info(
        f"CNN CV: mean accuracy = {result['mean_accuracy']:.3f} "
        f"± {result['std_accuracy']:.3f}  ({wall_time:.0f}s)"
    )
    return result

"""
ndm.py — Phase 3: Network Diffusion Model (NDM) for neurodegeneration spread.

This is the structural analog to the Fisher-Kolmogorov reaction-diffusion PDE
used in the original tumor pipeline. Mathematical family is the same
(diffusion operator on a Laplacian), but applied to brain connectivity graphs
rather than a DTI-derived tissue tensor.

Reference: Raj, Kuceyeski & Weiner (2012). A network diffusion model of disease
           progression in dementia. Neuron, 73(6), 1204-1215.
           Raj et al. (2015). Cell Reports, 10(3), 359-369.
           [ADNI validation using healthy reference connectome — same constraint as OASIS-2]

Model:
    x(t) = exp(-β · L · t) · x(0)

where:
    x(0): initial regional pathology vector (age-normalized atrophy, W-score)
    L:    graph Laplacian of the structural connectome
    β:    scalar diffusivity rate (patient-specific, fitted per subject)
    t:    time in years since baseline visit

Implementation:
    - Eigen-decomposition of L (computed once), then x(t) evaluated analytically.
    - β fitted per subject via scipy.optimize.minimize_scalar.
    - Template connectome used (Desikan-Killiany atlas parcellation, public source).
    - Age-normalized pathology score (W-score, not raw z-score).

"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
import scipy.stats
from tqdm.auto import tqdm



logger = setup_logging("ndm")
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Template connectome
# ---------------------------------------------------------------------------

CONNECTOME_REGIONS = [
    # Desikan-Killiany parcellation regions (subset used in Raj et al.)
    # Left hemisphere
    "ctx-lh-bankssts", "ctx-lh-caudalanteriorcingulate", "ctx-lh-caudalmiddlefrontal",
    "ctx-lh-cuneus", "ctx-lh-entorhinal", "ctx-lh-fusiform",
    "ctx-lh-inferiorparietal", "ctx-lh-inferiortemporal", "ctx-lh-isthmuscingulate",
    "ctx-lh-lateraloccipital", "ctx-lh-lateralorbitofrontal", "ctx-lh-lingual",
    "ctx-lh-medialorbitofrontal", "ctx-lh-middletemporal", "ctx-lh-parahippocampal",
    "ctx-lh-paracentral", "ctx-lh-parsopercularis", "ctx-lh-parsorbitalis",
    "ctx-lh-parstriangularis", "ctx-lh-pericalcarine", "ctx-lh-postcentral",
    "ctx-lh-posteriorcingulate", "ctx-lh-precentral", "ctx-lh-precuneus",
    "ctx-lh-rostralanteriorcingulate", "ctx-lh-rostralmiddlefrontal",
    "ctx-lh-superiorfrontal", "ctx-lh-superiorparietal", "ctx-lh-superiortemporal",
    "ctx-lh-supramarginal", "ctx-lh-frontalpole", "ctx-lh-temporalpole",
    "ctx-lh-transversetemporal", "ctx-lh-insula",
    # Subcortical (bilateral)
    "Left-Hippocampus", "Right-Hippocampus",
    "Left-Amygdala", "Right-Amygdala",
    "Left-Thalamus-Proper", "Right-Thalamus-Proper",
    "Left-Caudate", "Right-Caudate",
    "Left-Putamen", "Right-Putamen",
    # Right hemisphere cortical
    "ctx-rh-bankssts", "ctx-rh-caudalanteriorcingulate", "ctx-rh-caudalmiddlefrontal",
    "ctx-rh-entorhinal", "ctx-rh-inferiorparietal", "ctx-rh-inferiortemporal",
    "ctx-rh-middletemporal", "ctx-rh-parahippocampal", "ctx-rh-precuneus",
    "ctx-rh-superiorfrontal", "ctx-rh-superiortemporal", "ctx-rh-insula",
]

N_REGIONS = len(CONNECTOME_REGIONS)


def build_synthetic_connectome(n_regions: int = N_REGIONS, seed: int = 42) -> np.ndarray:
    """
    Build a synthetic structural connectivity matrix for testing/fallback.

    In practice, this should be replaced with a real template connectome
    (e.g., Budapest Reference Connectome, DSI Studio HCP atlas).

    The structure here approximates known brain connectivity properties:
    - Scale-free-like degree distribution
    - Higher within-hemisphere than between-hemisphere connections
    - Stronger medial temporal lobe connections (critical for AD)
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((n_regions, n_regions))

    # Random sparse base connectivity
    base = rng.exponential(scale=0.3, size=(n_regions, n_regions))
    base = (base + base.T) / 2
    np.fill_diagonal(base, 0)

    # Threshold to get sparsity (~20% density, typical for structural connectomes)
    threshold = np.percentile(base, 80)
    A = np.where(base > threshold, base, 0)

    # Boost medial temporal connections (hippocampus/entorhinal — AD signature)
    mtl_indices = [i for i, r in enumerate(CONNECTOME_REGIONS[:n_regions])
                   if any(k in r.lower() for k in ["hippocampus", "entorhinal", "parahippocampal"])]
    for i in mtl_indices:
        for j in mtl_indices:
            if i != j:
                A[i, j] += 0.5

    np.fill_diagonal(A, 0)
    return A


def load_template_connectome(connectome_path: Optional[Path] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Load a structural connectome adjacency matrix.

    Priority:
    1. Load from connectome_path if provided (must be .npy or .csv).
    2. Try to find a saved template in data/processed/.
    3. Fall back to the synthetic approximate connectome.

    Returns:
        (A, region_names) where A is the (n, n) adjacency matrix.
    """
    if connectome_path is not None and connectome_path.exists():
        try:
            if connectome_path.suffix == ".npy":
                A = np.load(str(connectome_path))
                logger.info(f"Loaded connectome from {connectome_path}: shape {A.shape}")
                return A, CONNECTOME_REGIONS[:A.shape[0]]
            elif connectome_path.suffix == ".csv":
                df = pd.read_csv(connectome_path, index_col=0)
                A = df.values
                regions = list(df.columns)
                logger.info(f"Loaded connectome CSV: {A.shape}, regions={regions[:3]}...")
                return A, regions
        except Exception as e:
            logger.warning(f"Failed to load connectome from {connectome_path}: {e}")

    # Check for a cached template
    cached_path = get_data_dir("processed") / "template_connectome.npy"
    if cached_path.exists():
        A = np.load(str(cached_path))
        logger.info(f"Loaded cached template connectome: {A.shape}")
        return A, CONNECTOME_REGIONS[:A.shape[0]]

    # Fallback: synthetic
    logger.warning(
        "No template connectome found. Using synthetic approximate connectome.\n"
        "For production use, replace with a real template such as:\n"
        "  - Budapest Reference Connectome (http://braingraph.org/cms/download-datasets/)\n"
        "  - DSI Studio HCP-1065 atlas\n"
        "This is consistent with Raj et al. (2015) precedent (healthy-subject reference)."
    )
    A = build_synthetic_connectome(N_REGIONS)
    np.save(str(cached_path), A)
    return A, CONNECTOME_REGIONS


# ---------------------------------------------------------------------------
# Graph Laplacian
# ---------------------------------------------------------------------------

def compute_laplacian(A: np.ndarray) -> np.ndarray:
    """
    Compute the normalized graph Laplacian L = D - A.
    D is the degree matrix (diagonal).
    """
    # Ensure symmetry and non-negative weights
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    A = np.maximum(A, 0)

    degree = A.sum(axis=1)
    D = np.diag(degree)
    L = D - A
    return L


def eigen_decompose_laplacian(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigendecompose the Laplacian: L = V @ diag(eigenvalues) @ V^T.
    Uses scipy.linalg.eigh (symmetric matrix — guaranteed real eigenvalues).

    Returns (eigenvalues, eigenvectors) — compute once, reuse for all t/β.
    """
    eigenvalues, eigenvectors = scipy.linalg.eigh(L)
    # Clip tiny negative eigenvalues (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# NDM propagation
# ---------------------------------------------------------------------------

def ndm_propagate(
    x0: np.ndarray,
    t: float,
    beta: float,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """
    Compute x(t) = exp(-β·L·t)·x(0) using pre-computed eigen-decomposition.

    x(t) = V @ diag(exp(-β·λ_i·t)) @ V^T @ x(0)

    This is O(n^2) after the initial O(n^3) decomposition, vs. O(n^3) per call
    for scipy.linalg.expm.
    """
    decay = np.exp(-beta * eigenvalues * t)
    # x(t) = V · (decay · (V^T · x0))
    x_t = eigenvectors @ (decay * (eigenvectors.T @ x0))
    return x_t


# ---------------------------------------------------------------------------
# Age-normalized pathology score (W-score)
# ---------------------------------------------------------------------------

def compute_w_scores(
    regional_volumes: pd.DataFrame,
    demo_df: pd.DataFrame,
    region_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute age-normalized atrophy scores (W-scores) for each subject/visit.

    Steps (per implementation.md §5.2):
    1. In stably-Nondemented subjects: fit linear regression of each region's
       eTIV-normalized volume against age.
    2. For every subject/visit: compute residual from age-expected volume, then z-score.

    This removes the confound of normal age-related volume loss.

    Returns a DataFrame with the same index as regional_volumes, with W-score columns
    appended (suffix _wscore).
    """
    # Merge demographics
    demo_df = demo_df.copy()
    if "subject_id" not in demo_df.columns and "Subject ID" in demo_df.columns:
        demo_df["subject_id"] = demo_df["Subject ID"].str.upper().str.strip()

    merge_cols = ["subject_id"] + [
        c for c in ["Age", "CDR", "Group"]
        if c in demo_df.columns and c not in regional_volumes.columns
    ]
    merged = regional_volumes.merge(
        demo_df[merge_cols],
        on="subject_id", how="left",
    )

    # Identify region columns (normalized volumes)
    if region_cols is None:
        region_cols = [
            c for c in regional_volumes.columns
            if c.endswith("_norm") or (
                c not in {"subject_id", "session_id"} and
                pd.api.types.is_numeric_dtype(regional_volumes[c])
            )
        ]
        region_cols = [c for c in region_cols if c in merged.columns]

    # Identify nondemented subjects for reference regression
    if "CDR" in merged.columns:
        nondem_mask = merged["CDR"] == 0
    elif "Group" in merged.columns:
        nondem_mask = merged["Group"] == "Nondemented"
    else:
        logger.warning("No CDR or Group column found; using all subjects as reference.")
        nondem_mask = pd.Series(True, index=merged.index)

    if "Age" not in merged.columns:
        logger.warning("Age column not found; W-score will be a plain z-score.")
        w_scores = regional_volumes[region_cols].copy()
        for col in region_cols:
            ref_vals = w_scores.loc[nondem_mask, col].dropna()
            mean, std = ref_vals.mean(), ref_vals.std()
            std = std if std > 1e-10 else 1.0
            w_scores[f"{col}_wscore"] = -(w_scores[col] - mean) / std  # negative: atrophy = positive score
        return w_scores

    from sklearn.linear_model import LinearRegression

    w_df = regional_volumes[["subject_id", "session_id"]].copy()
    age_values = merged["Age"].values.reshape(-1, 1)

    for col in region_cols:
        y = merged[col].values

        # Fit on non-demented subjects only
        mask = nondem_mask.values & ~np.isnan(y) & ~np.isnan(age_values.ravel())
        if mask.sum() < 3:
            logger.warning(f"Too few nondemented samples to fit regression for {col}; using mean.")
            mu = np.nanmean(y[nondem_mask.values])
            sigma = np.nanstd(y[nondem_mask.values])
            sigma = sigma if sigma > 1e-10 else 1.0
            w_df[f"{col}_wscore"] = -(y - mu) / sigma
            continue

        reg = LinearRegression()
        reg.fit(age_values[mask], y[mask])
        predicted = reg.predict(age_values)
        residuals = y - predicted

        # Z-score residuals against the nondemented distribution
        ref_resid = residuals[nondem_mask.values & ~np.isnan(residuals)]
        mu, sigma = ref_resid.mean(), ref_resid.std()
        sigma = sigma if sigma > 1e-10 else 1.0

        # Flip sign: lower volume → higher pathology score
        w_df[f"{col}_wscore"] = -(residuals - mu) / sigma

    return w_df


# ---------------------------------------------------------------------------
# Per-subject β fitting
# ---------------------------------------------------------------------------

def fit_beta(
    x0: np.ndarray,
    x_t: np.ndarray,
    t: float,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    beta_bounds: Tuple[float, float] = (0.001, 10.0),
) -> float:
    """
    Fit β for a single subject by minimising squared error between
    predicted x(t) and observed x(t).

    Uses scipy.optimize.minimize_scalar (bounded 1D optimization).

    Returns the fitted β value.
    """
    valid = ~np.isnan(x0) & ~np.isnan(x_t)
    if valid.sum() < 3:
        return np.nan

    x0_clean = np.where(np.isnan(x0), 0.0, x0)
    x_t_clean = x_t[valid]

    def objective(beta):
        pred = ndm_propagate(x0_clean, t, beta, eigenvalues, eigenvectors)
        return np.mean((pred[valid] - x_t_clean) ** 2)

    result = scipy.optimize.minimize_scalar(
        objective,
        bounds=beta_bounds,
        method="bounded",
        options={"xatol": 1e-6, "maxiter": 200},
    )
    return float(result.x) if result.success else np.nan


def fit_all_betas(
    w_scores: pd.DataFrame,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    demo_df: pd.DataFrame,
    wscore_cols: Optional[List[str]] = None,
    time_col: str = "MR Delay",
) -> pd.DataFrame:
    """
    Fit β for every subject with ≥2 visits.

    `time_col` is the OASIS-2 'MR Delay' column (days since baseline).
    Converts to years internally.

    Returns a DataFrame: subject_id → fitted β.
    """
    if wscore_cols is None:
        wscore_cols = [c for c in w_scores.columns if c.endswith("_wscore")]

    if not wscore_cols:
        logger.error("No W-score columns found in w_scores DataFrame.")
        return pd.DataFrame()

    # Merge time info
    demo_df = demo_df.copy()
    if "subject_id" not in demo_df.columns and "Subject ID" in demo_df.columns:
        demo_df["subject_id"] = demo_df["Subject ID"].str.upper().str.strip()

    if time_col in demo_df.columns:
        merged = w_scores.merge(
            demo_df[["subject_id", "session_id", time_col]],
            on=["subject_id", "session_id"], how="left",
        )
    else:
        # Fall back: assume sessions are ordered chronologically
        merged = w_scores.copy()
        merged["_visit_idx"] = merged.groupby("subject_id").cumcount()
        merged[time_col] = merged["_visit_idx"] * 365  # assume ~1 year between visits

    results = []
    grouped = list(merged.groupby("subject_id"))
    for subj_id, grp in tqdm(grouped, desc="Fitting NDM beta per subject", unit="subj"):
        grp = grp.sort_values(time_col)
        if len(grp) < 2:
            continue  # need at least 2 visits

        # Baseline visit
        row0 = grp.iloc[0]
        x0 = row0[wscore_cols].values.astype(float)

        # Fit β on visit 1 → visit 2 (and optionally validate on visit 3+)
        for i in range(1, len(grp)):
            row_i = grp.iloc[i]
            t_years = float(row_i[time_col]) / 365.0
            if t_years <= 0:
                continue

            x_obs = row_i[wscore_cols].values.astype(float)
            beta = fit_beta(x0, x_obs, t_years, eigenvalues, eigenvectors)

            results.append({
                "subject_id": subj_id,
                "session_id": row_i.get("session_id", f"visit_{i}"),
                "t_years": round(t_years, 3),
                "beta": beta,
                "n_regions_used": int(wscore_cols.__len__()),
            })

    df = pd.DataFrame(results)
    logger.info(
        f"β fitted for {df['subject_id'].nunique()} subjects "
        f"({df['beta'].isna().sum()} failed)."
    )
    return df


# ---------------------------------------------------------------------------
# Future atrophy prediction
# ---------------------------------------------------------------------------

def predict_future_atrophy(
    x0: np.ndarray,
    beta: float,
    t_future_years: float,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    region_names: Optional[List[str]] = None,
) -> pd.Series:
    """
    Predict regional atrophy at a future time point for one subject.

    Returns a pd.Series indexed by region_name (or integer index).
    """
    if np.isnan(beta) or beta <= 0:
        logger.warning("Invalid β value; returning NaN predictions.")
        predicted = np.full(len(x0), np.nan)
    else:
        predicted = ndm_propagate(x0, t_future_years, beta, eigenvalues, eigenvectors)

    index = region_names if region_names is not None else np.arange(len(predicted))
    return pd.Series(predicted, index=index, name="predicted_atrophy")


# ---------------------------------------------------------------------------
# NDM validation: predict visit 3 from visit 1→2 fit
# ---------------------------------------------------------------------------

def validate_ndm(
    w_scores: pd.DataFrame,
    beta_df: pd.DataFrame,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    demo_df: pd.DataFrame,
    wscore_cols: Optional[List[str]] = None,
    time_col: str = "MR Delay",
) -> Dict:
    """
    For subjects with 3+ visits, fit β on visit 1→2 and predict visit 3.
    Report Pearson r and MSE between predicted and actual regional atrophy.

    See implementation.md §5.5 for context on what to expect.
    """
    if wscore_cols is None:
        wscore_cols = [c for c in w_scores.columns if c.endswith("_wscore")]

    demo_df = demo_df.copy()
    if "subject_id" not in demo_df.columns and "Subject ID" in demo_df.columns:
        demo_df["subject_id"] = demo_df["Subject ID"].str.upper().str.strip()

    if time_col in demo_df.columns:
        merged = w_scores.merge(
            demo_df[["subject_id", "session_id", time_col]],
            on=["subject_id", "session_id"], how="left",
        )
    else:
        merged = w_scores.copy()
        merged["_visit_idx"] = merged.groupby("subject_id").cumcount()
        merged[time_col] = merged["_visit_idx"] * 365

    all_predicted = []
    all_actual = []

    for subj_id, grp in tqdm(list(merged.groupby("subject_id")), desc="Validating NDM", unit="subj"):
        if len(grp) < 3:
            continue

        grp = grp.sort_values(time_col)
        subj_betas = beta_df[beta_df["subject_id"] == subj_id]
        if subj_betas.empty or subj_betas["beta"].isna().all():
            continue

        beta = float(subj_betas["beta"].dropna().iloc[0])
        row0 = grp.iloc[0]
        x0 = row0[wscore_cols].values.astype(float)

        row2 = grp.iloc[2]
        t2_years = float(row2[time_col]) / 365.0
        x_actual = row2[wscore_cols].values.astype(float)

        x_pred = ndm_propagate(x0, t2_years, beta, eigenvalues, eigenvectors)

        valid = ~np.isnan(x_actual) & ~np.isnan(x_pred)
        if valid.sum() < 3:
            continue
        all_predicted.extend(x_pred[valid].tolist())
        all_actual.extend(x_actual[valid].tolist())

    if len(all_predicted) < 5:
        logger.warning("Not enough 3-visit subjects for NDM validation.")
        return {"n_subjects_validated": 0, "pearson_r": np.nan, "mse": np.nan}

    r, p = scipy.stats.pearsonr(all_predicted, all_actual)
    mse = np.mean((np.array(all_predicted) - np.array(all_actual)) ** 2)

    result = {
        "n_subjects_validated": len(set()),
        "n_region_observations": len(all_predicted),
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 6),
        "mse": round(float(mse), 6),
        "note": (
            "Raj et al. (2015) reported R≈0.93 on ADNI with their setup. "
            "This OASIS-2 proof-of-concept uses a template connectome and a "
            "smaller, differently-processed dataset — not a replication attempt."
        ),
    }
    logger.info(
        f"NDM validation: Pearson r={result['pearson_r']:.3f}, MSE={result['mse']:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# Full Phase 3 pipeline
# ---------------------------------------------------------------------------

def run_phase3(
    regional_volumes: pd.DataFrame,
    demo_df: pd.DataFrame,
    connectome_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Dict:
    """
    Phase 3 main entry point.

    Steps:
      1. Load template connectome → compute Laplacian → eigen-decompose.
      2. Compute W-scores (age-normalized atrophy).
      3. Fit β per subject.
      4. Validate on 3-visit subjects.
      5. Save results.
    """
    import json

    if out_dir is None:
        out_dir = get_outputs_dir("phase3")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Connectome
    A, region_names = load_template_connectome(connectome_path)
    L = compute_laplacian(A)
    eigenvalues, eigenvectors = eigen_decompose_laplacian(L)
    logger.info(f"Connectome: {A.shape[0]} regions, Laplacian eigenvalue range "
                f"[{eigenvalues.min():.3f}, {eigenvalues.max():.3f}]")

    np.save(str(out_dir / "laplacian_eigenvalues.npy"), eigenvalues)
    np.save(str(out_dir / "laplacian_eigenvectors.npy"), eigenvectors)

    # Step 2: W-scores
    w_scores = compute_w_scores(regional_volumes, demo_df)
    w_scores.to_csv(out_dir / "w_scores.csv", index=False)
    logger.info(f"W-scores computed: {w_scores.shape}")

    # Step 3: Fit β
    beta_df = fit_all_betas(w_scores, eigenvalues, eigenvectors, demo_df)
    if not beta_df.empty:
        beta_df.to_csv(out_dir / "fitted_betas.csv", index=False)

    # Step 4: Validate
    validation = validate_ndm(w_scores, beta_df, eigenvalues, eigenvectors, demo_df)
    with open(out_dir / "ndm_validation.json", "w") as f:
        json.dump(validation, f, indent=2)

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "region_names": region_names,
        "w_scores": w_scores,
        "beta_df": beta_df,
        "validation": validation,
    }

"""
symptom_mapping.py — Phase 4: Neuroanatomical mapping & cognitive deficit prediction.

This is the structural analog to the eloquent-cortex/aphasia mapping section
in the original tumor pipeline — same mechanism, different target regions and deficits.

Mechanism:
  1. Region → cognitive domain lookup table (grounded in standard neuropsychology).
  2. For each subject's predicted future atrophy vector (from Phase 3 NDM),
     compute a weighted deficit score per cognitive domain.
  3. Validate against real outcomes: correlate predicted deficit severity against
     actual MMSE trajectory and CDR change.

This is a genuinely checkable prediction (unlike the tumor doc's symptom section)
because OASIS-2 has longitudinal MMSE and CDR ground truth.

"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats



logger = setup_logging("symptom_mapping")


# ---------------------------------------------------------------------------
# Region → cognitive domain atlas
# ---------------------------------------------------------------------------
# Based on standard neuropsychology literature; partial list representative
# of key AD-relevant regions.

REGION_DOMAIN_MAP = {
    # ------- Episodic Memory -------
    "Left-Hippocampus": {"episodic_memory": 1.0, "learning": 0.8},
    "Right-Hippocampus": {"episodic_memory": 0.9, "learning": 0.7},
    "ctx-lh-entorhinal": {"episodic_memory": 1.0, "learning": 0.9},
    "ctx-rh-entorhinal": {"episodic_memory": 0.9, "learning": 0.8},
    "ctx-lh-parahippocampal": {"episodic_memory": 0.8, "learning": 0.6},
    "ctx-rh-parahippocampal": {"episodic_memory": 0.7, "learning": 0.6},
    "ctx-lh-fusiform": {"episodic_memory": 0.5, "object_recognition": 0.7},
    "ctx-rh-fusiform": {"episodic_memory": 0.4, "object_recognition": 0.8},
    "ctx-lh-isthmuscingulate": {"episodic_memory": 0.7},
    "ctx-rh-isthmuscingulate": {"episodic_memory": 0.6},
    # ------- Executive Function -------
    "ctx-lh-rostralmiddlefrontal": {"executive_function": 1.0, "working_memory": 0.8},
    "ctx-rh-rostralmiddlefrontal": {"executive_function": 0.9, "working_memory": 0.7},
    "ctx-lh-caudalmiddlefrontal": {"executive_function": 0.9},
    "ctx-rh-caudalmiddlefrontal": {"executive_function": 0.8},
    "ctx-lh-superiorfrontal": {"executive_function": 0.8, "working_memory": 0.6},
    "ctx-rh-superiorfrontal": {"executive_function": 0.7},
    "ctx-lh-caudalanteriorcingulate": {"executive_function": 0.7, "attention": 0.8},
    "ctx-rh-caudalanteriorcingulate": {"executive_function": 0.6, "attention": 0.7},
    "ctx-lh-rostralanteriorcingulate": {"executive_function": 0.6, "attention": 0.9},
    "ctx-rh-rostralanteriorcingulate": {"executive_function": 0.5, "attention": 0.8},
    "Left-Caudate": {"executive_function": 0.6},
    "Right-Caudate": {"executive_function": 0.5},
    # ------- Language -------
    "ctx-lh-superiortemporal": {"language": 1.0, "auditory_processing": 0.8},
    "ctx-lh-middletemporal": {"language": 0.9},
    "ctx-lh-inferiortemporal": {"language": 0.7, "semantic_memory": 0.8},
    "ctx-lh-parsopercularis": {"language": 0.9},  # Broca's area
    "ctx-lh-parstriangularis": {"language": 0.8},
    "ctx-lh-parsorbitalis": {"language": 0.7},
    "ctx-lh-bankssts": {"language": 0.8},  # Wernicke-adjacent
    "ctx-rh-superiortemporal": {"language": 0.5, "auditory_processing": 0.9},
    "ctx-rh-middletemporal": {"language": 0.4},
    # ------- Visuospatial -------
    "ctx-lh-inferiorparietal": {"visuospatial": 1.0, "attention": 0.7},
    "ctx-rh-inferiorparietal": {"visuospatial": 0.9, "attention": 0.6},
    "ctx-lh-superiorparietal": {"visuospatial": 0.9},
    "ctx-rh-superiorparietal": {"visuospatial": 0.8},
    "ctx-lh-precuneus": {"visuospatial": 0.8, "episodic_memory": 0.5},
    "ctx-rh-precuneus": {"visuospatial": 0.7, "episodic_memory": 0.5},
    "ctx-lh-lateraloccipital": {"visuospatial": 0.8, "object_recognition": 0.7},
    "ctx-rh-lateraloccipital": {"visuospatial": 0.8, "object_recognition": 0.6},
    "ctx-lh-cuneus": {"visuospatial": 0.7},
    "ctx-lh-lingual": {"visuospatial": 0.6, "object_recognition": 0.5},
    # ------- Emotional/Behavioral -------
    "Left-Amygdala": {"emotional_regulation": 1.0, "behavioral_control": 0.8},
    "Right-Amygdala": {"emotional_regulation": 0.9, "behavioral_control": 0.7},
    "ctx-lh-insula": {"emotional_regulation": 0.7, "interoception": 0.8},
    "ctx-rh-insula": {"emotional_regulation": 0.6, "interoception": 0.7},
    # ------- Attention / Thalamic -------
    "Left-Thalamus-Proper": {"attention": 0.8, "sensory_relay": 0.9},
    "Right-Thalamus-Proper": {"attention": 0.7, "sensory_relay": 0.9},
    "ctx-lh-posteriorcingulate": {"attention": 0.7, "episodic_memory": 0.6},
    "ctx-rh-posteriorcingulate": {"attention": 0.6, "episodic_memory": 0.5},
}

COGNITIVE_DOMAINS = [
    "episodic_memory", "executive_function", "language",
    "visuospatial", "emotional_regulation", "attention",
    "working_memory", "semantic_memory", "learning",
    "object_recognition", "behavioral_control",
]


def get_domain_weights(region_name: str) -> Dict[str, float]:
    """Return the cognitive domain weights for a brain region."""
    # Try exact match first
    if region_name in REGION_DOMAIN_MAP:
        return REGION_DOMAIN_MAP[region_name]
    # Try case-insensitive partial match (for SynthSeg label variants)
    rn_lower = region_name.lower()
    for key, weights in REGION_DOMAIN_MAP.items():
        if key.lower() in rn_lower or rn_lower in key.lower():
            return weights
    return {}


# ---------------------------------------------------------------------------
# Deficit score computation
# ---------------------------------------------------------------------------

def compute_deficit_scores(
    atrophy_vector: np.ndarray,
    region_names: List[str],
    normalize: bool = True,
) -> pd.Series:
    """
    Compute weighted deficit score per cognitive domain.

    deficit_score[domain] = Σ (atrophy_i × weight_{i,domain})
    where the sum is over regions that have non-zero weight for that domain.

    Args:
        atrophy_vector: (n_regions,) array of pathology scores (W-scores or predicted atrophy)
        region_names:   list of region names corresponding to atrophy_vector
        normalize:      if True, normalize scores to [0, 1] range

    Returns:
        pd.Series indexed by cognitive domain name.
    """
    domain_scores = {d: 0.0 for d in COGNITIVE_DOMAINS}
    domain_weight_totals = {d: 0.0 for d in COGNITIVE_DOMAINS}

    for i, (region, atrophy) in enumerate(zip(region_names, atrophy_vector)):
        if np.isnan(atrophy):
            continue
        weights = get_domain_weights(region)
        for domain, w in weights.items():
            if domain in domain_scores:
                domain_scores[domain] += atrophy * w
                domain_weight_totals[domain] += w

    # Normalize by total weights (weighted average, not sum)
    for domain in COGNITIVE_DOMAINS:
        if domain_weight_totals[domain] > 0:
            domain_scores[domain] /= domain_weight_totals[domain]

    scores = pd.Series(domain_scores)

    if normalize:
        # Min-max to [0, 1]
        score_range = scores.max() - scores.min()
        if score_range > 1e-10:
            scores = (scores - scores.min()) / score_range

    return scores


def compute_all_deficit_scores(
    predicted_atrophies: pd.DataFrame,  # rows = subjects, cols = regions
    region_names: List[str],
) -> pd.DataFrame:
    """
    Compute deficit scores for a batch of subjects.

    Returns a DataFrame: one row per subject, one column per cognitive domain.
    """
    rows = []
    for idx, row in predicted_atrophies.iterrows():
        atrophy = row[region_names].values.astype(float) if all(r in row.index for r in region_names) else row.values.astype(float)
        scores = compute_deficit_scores(atrophy, region_names)
        scores.name = idx
        rows.append(scores)

    if not rows:
        return pd.DataFrame(columns=COGNITIVE_DOMAINS)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation: correlate predicted deficits with actual MMSE/CDR outcomes
# ---------------------------------------------------------------------------

def validate_deficit_predictions(
    deficit_scores: pd.DataFrame,
    demo_df: pd.DataFrame,
    outcome_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Correlate predicted cognitive-domain deficit scores with actual MMSE and CDR
    trajectories.

    Returns a dict of {domain: {outcome: pearson_r}}.
    """
    if outcome_cols is None:
        outcome_cols = [c for c in ["MMSE", "CDR", "mmse_delta", "cdr_delta"]
                        if c in demo_df.columns]

    if not outcome_cols:
        logger.warning("No MMSE/CDR outcome columns found for validation.")
        return {}

    demo_df = demo_df.copy()
    if "subject_id" not in demo_df.columns and "Subject ID" in demo_df.columns:
        demo_df["subject_id"] = demo_df["Subject ID"].str.upper().str.strip()

    merged = deficit_scores.join(
        demo_df.set_index("subject_id")[outcome_cols], how="inner"
    )

    if merged.empty:
        logger.warning("No overlapping subjects between deficit scores and demo_df.")
        return {}

    results = {}
    for domain in COGNITIVE_DOMAINS:
        if domain not in merged.columns:
            continue
        for outcome in outcome_cols:
            if outcome not in merged.columns:
                continue
            valid = merged[[domain, outcome]].dropna()
            if len(valid) < 5:
                continue
            r, p = scipy.stats.pearsonr(valid[domain], valid[outcome])
            results.setdefault(domain, {})[outcome] = {
                "pearson_r": round(float(r), 4),
                "p_value": round(float(p), 6),
                "n": len(valid),
            }

    logger.info(f"Validated deficit scores for {len(results)} domains.")
    return results


# ---------------------------------------------------------------------------
# Compute MMSE/CDR deltas from longitudinal data
# ---------------------------------------------------------------------------

def compute_longitudinal_deltas(demo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-subject MMSE and CDR change (last visit - first visit).
    Returns a DataFrame indexed by subject_id with columns mmse_delta, cdr_delta.
    """
    demo_df = demo_df.copy()
    if "subject_id" not in demo_df.columns and "Subject ID" in demo_df.columns:
        demo_df["subject_id"] = demo_df["Subject ID"].str.upper().str.strip()
    if "session_id" not in demo_df.columns and "MRI ID" in demo_df.columns:
        demo_df["session_id"] = demo_df["MRI ID"].str.upper()

    rows = []
    for subj, grp in demo_df.groupby("subject_id"):
        if "MR Delay" in grp.columns:
            grp = grp.sort_values("MR Delay")
        elif "Visit" in grp.columns:
            grp = grp.sort_values("Visit")

        row = {"subject_id": subj}
        for col, delta_col in [("MMSE", "mmse_delta"), ("CDR", "cdr_delta")]:
            if col in grp.columns:
                valid = grp[col].dropna()
                if len(valid) >= 2:
                    row[delta_col] = float(valid.iloc[-1] - valid.iloc[0])
        rows.append(row)

    return pd.DataFrame(rows).set_index("subject_id")


# ---------------------------------------------------------------------------
# Full Phase 4 pipeline
# ---------------------------------------------------------------------------

def run_phase4(
    ndm_results: Dict,
    demo_df: pd.DataFrame,
    out_dir: Optional[Path] = None,
) -> Dict:
    """
    Phase 4 main entry point.

    Uses NDM predicted future atrophy trajectories to compute cognitive deficit
    scores, then validates against real MMSE/CDR outcomes.
    """
    if out_dir is None:
        out_dir = get_outputs_dir("phase4")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    beta_df = ndm_results.get("beta_df", pd.DataFrame())
    eigenvalues = ndm_results.get("eigenvalues")
    eigenvectors = ndm_results.get("eigenvectors")
    region_names = ndm_results.get("region_names", [])
    w_scores = ndm_results.get("w_scores", pd.DataFrame())

    if beta_df.empty or eigenvalues is None:
        logger.error("NDM results missing β values or eigenvalues. Run Phase 3 first.")
        return {}

    # Compute predicted 5-year atrophy for each subject

    wscore_cols = [c for c in w_scores.columns if c.endswith("_wscore")]
    t_future = 5.0  # predict 5 years ahead

    predicted_rows = []
    for _, row in beta_df.drop_duplicates("subject_id").iterrows():
        subj_id = row["subject_id"]
        beta = row["beta"]
        if np.isnan(beta):
            continue

        # Get baseline W-scores for this subject
        subj_ws = w_scores[w_scores["subject_id"] == subj_id]
        if subj_ws.empty or not wscore_cols:
            continue
        x0 = subj_ws.sort_values("session_id").iloc[0][wscore_cols].values.astype(float)

        n = min(len(x0), len(eigenvalues))
        pred = ndm_propagate(x0[:n], t_future, beta, eigenvalues[:n], eigenvectors[:n, :n])

        pred_dict = {"subject_id": subj_id}
        for i, region in enumerate(region_names[:n]):
            pred_dict[region] = pred[i] if i < len(pred) else np.nan
        predicted_rows.append(pred_dict)

    if not predicted_rows:
        logger.warning("No predicted atrophy available for Phase 4.")
        return {}

    predicted_df = pd.DataFrame(predicted_rows).set_index("subject_id")
    predicted_df.to_csv(out_dir / "predicted_atrophy_5yr.csv")

    # Compute deficit scores
    pred_region_cols = [c for c in predicted_df.columns if c in region_names]
    deficit_df = compute_all_deficit_scores(predicted_df, pred_region_cols)
    deficit_df.to_csv(out_dir / "deficit_scores.csv")

    # Compute real outcome deltas
    deltas = compute_longitudinal_deltas(demo_df)
    demo_with_deltas = demo_df.copy()
    if "subject_id" not in demo_with_deltas.columns and "Subject ID" in demo_with_deltas.columns:
        demo_with_deltas["subject_id"] = demo_with_deltas["Subject ID"].str.upper()
    demo_with_deltas = demo_with_deltas.merge(
        deltas.reset_index(), on="subject_id", how="left"
    )

    # Validate
    validation = validate_deficit_predictions(deficit_df, demo_with_deltas,
                                              outcome_cols=["MMSE", "CDR", "mmse_delta", "cdr_delta"])
    with open(out_dir / "deficit_validation.json", "w") as f:
        json.dump(validation, f, indent=2)

    logger.info(f"Phase 4 complete. Outputs in: {out_dir}")
    return {
        "predicted_atrophy": predicted_df,
        "deficit_scores": deficit_df,
        "validation": validation,
    }

"""
risk_model.py — Phase 5: Clinical/Demographic Risk Profiling.

Analog to radiogenomics (IDH/MGMT stratification) in the original tumor pipeline.
OASIS-2 has no genotype data (no APOE, unlike OASIS-3) — this is explicitly a
clinical-feature proxy, as documented in implementation.md §7 and the final report.

Features: Age, Sex, Education, SES, eTIV, nWBV + Phase 1 regional volumes
          + optionally Phase 3's fitted β as a continuous aggressiveness target.

Target options:
  - Binary: predicted future CDR increase (1 = declined, 0 = stable)
  - Continuous: Phase 3 fitted diffusivity β (higher β → faster spread)

Model: Gradient Boosting (primary) + Logistic Regression (interpretable baseline).
       Both trivially fast on CPU.

"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score



logger = setup_logging("risk_model")
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

CLINICAL_FEATURES = ["Age", "EDUC", "SES", "eTIV", "nWBV"]
SEX_COL = "M/F"


def build_risk_features(
    demo_df: pd.DataFrame,
    regional_volumes: Optional[pd.DataFrame] = None,
    beta_df: Optional[pd.DataFrame] = None,
    deficit_scores: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Assemble the risk model feature matrix from clinical + imaging + NDM features.

    Returns (feature_df, feature_names) where feature_df is indexed by subject_id.
    """
    demo_df = demo_df.copy()
    if "subject_id" not in demo_df.columns and "Subject ID" in demo_df.columns:
        demo_df["subject_id"] = demo_df["Subject ID"].str.upper().str.strip()

    # Start with one row per subject (use first visit for demographics)
    if "MR Delay" in demo_df.columns:
        demo_sub = demo_df.sort_values("MR Delay").groupby("subject_id").first().reset_index()
    else:
        demo_sub = demo_df.groupby("subject_id").first().reset_index()

    # Clinical features
    feature_cols = []
    if SEX_COL in demo_sub.columns:
        demo_sub["sex_encoded"] = (demo_sub[SEX_COL].str.upper() == "M").astype(float)
        feature_cols.append("sex_encoded")

    for col in CLINICAL_FEATURES:
        if col in demo_sub.columns:
            feature_cols.append(col)

    feat_df = demo_sub[["subject_id"] + feature_cols].copy()

    # Regional volumes (mean across visits per subject)
    if regional_volumes is not None and not regional_volumes.empty:
        vol_cols = [c for c in regional_volumes.columns
                    if c.endswith("_norm") and c in regional_volumes.columns]
        if not vol_cols:
            vol_cols = [c for c in regional_volumes.columns
                        if c not in {"subject_id", "session_id"}
                        and c not in feat_df.columns  # avoid re-adding clinical cols already present
                        and pd.api.types.is_numeric_dtype(regional_volumes[c])]
        if vol_cols:
            vol_mean = regional_volumes.groupby("subject_id")[vol_cols].mean().reset_index()
            feat_df = feat_df.merge(vol_mean, on="subject_id", how="left")
            feature_cols.extend(vol_cols)

    # Phase 3 fitted β (continuous risk proxy)
    if beta_df is not None and not beta_df.empty and "beta" in beta_df.columns:
        beta_mean = beta_df.groupby("subject_id")["beta"].median().reset_index()
        beta_mean.rename(columns={"beta": "ndm_beta"}, inplace=True)
        feat_df = feat_df.merge(beta_mean, on="subject_id", how="left")
        feature_cols.append("ndm_beta")

    # Phase 4 deficit scores
    if deficit_scores is not None and not deficit_scores.empty:
        ds = deficit_scores.reset_index() if deficit_scores.index.name == "subject_id" else deficit_scores
        if "subject_id" in ds.columns:
            feat_df = feat_df.merge(ds, on="subject_id", how="left")
            feature_cols.extend([c for c in ds.columns if c != "subject_id"])

    feat_df = feat_df.set_index("subject_id")
    feature_cols = [c for c in feature_cols if c in feat_df.columns]
    feat_df = feat_df[feature_cols]

    logger.info(f"Risk feature matrix: {feat_df.shape[0]} subjects × {len(feature_cols)} features")
    return feat_df, feature_cols


# ---------------------------------------------------------------------------
# Target label construction
# ---------------------------------------------------------------------------

def build_risk_labels(
    demo_df: pd.DataFrame,
    beta_df: Optional[pd.DataFrame] = None,
    target: str = "cdr_increase",
) -> pd.Series:
    """
    Build the risk target variable.

    Options:
      'cdr_increase': binary, 1 if CDR increased from first to last visit
      'beta':         continuous, NDM diffusivity β (requires beta_df)
      'demented':     binary, 1 if final Group == 'Demented' or 'Converted'

    Returns a pd.Series indexed by subject_id.
    """
    demo_df = demo_df.copy()
    if "subject_id" not in demo_df.columns and "Subject ID" in demo_df.columns:
        demo_df["subject_id"] = demo_df["Subject ID"].str.upper().str.strip()

    if target == "cdr_increase":
        rows = []
        for subj, grp in demo_df.groupby("subject_id"):
            if "MR Delay" in grp.columns:
                grp = grp.sort_values("MR Delay")
            elif "Visit" in grp.columns:
                grp = grp.sort_values("Visit")
            if "CDR" not in grp.columns or len(grp) < 2:
                continue
            cdr_vals = grp["CDR"].dropna()
            if len(cdr_vals) < 2:
                continue
            label = 1 if cdr_vals.iloc[-1] > cdr_vals.iloc[0] else 0
            rows.append({"subject_id": subj, "target": label})
        df = pd.DataFrame(rows).set_index("subject_id")
        return df["target"]

    elif target == "beta" and beta_df is not None:
        beta_median = beta_df.groupby("subject_id")["beta"].median()
        return beta_median.rename("target")

    elif target == "demented":
        demo_sub = demo_df.groupby("subject_id").last().reset_index()
        if "Group" not in demo_sub.columns:
            logger.warning("Group column not found for 'demented' target.")
            return pd.Series(dtype=float)
        label = (demo_sub["Group"].isin(["Demented", "Converted"])).astype(int)
        return label.set_axis(demo_sub["subject_id"]).rename("target")

    else:
        raise ValueError(f"Unknown target: {target}. Choose 'cdr_increase', 'beta', or 'demented'.")


# ---------------------------------------------------------------------------
# Model pipelines
# ---------------------------------------------------------------------------

def build_risk_pipelines(task: str = "classification") -> Dict[str, Pipeline]:
    if task == "classification":
        return {
            "GradientBoosting": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.05,
                    max_depth=3, random_state=42,
                )),
            ]),
            "LogisticRegression": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=42,
                )),
            ]),
        }
    else:  # regression (for β target)
        return {
            "GradientBoostingRegressor": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("reg", GradientBoostingRegressor(
                    n_estimators=200, learning_rate=0.05,
                    max_depth=3, random_state=42,
                )),
            ]),
            "Ridge": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0)),
            ]),
        }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def evaluate_risk_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    task: str = "classification",
    n_splits: int = 5,
) -> Dict:
    pipelines = build_risk_pipelines(task=task)
    n_splits = min(n_splits, len(np.unique(groups)))

    gkf = GroupKFold(n_splits=n_splits)
    results = {}

    if task == "classification":
        scoring = ["accuracy", "roc_auc", "f1_weighted"]
    else:
        scoring = ["r2", "neg_mean_squared_error"]

    for name, pipe in tqdm(pipelines.items(), desc="Risk models", unit="model"):
        try:
            cv = cross_validate(pipe, X, y, groups=groups, cv=gkf, scoring=scoring)
            results[name] = {
                metric: {
                    "mean": round(float(cv[f"test_{metric}"].mean()), 4),
                    "std": round(float(cv[f"test_{metric}"].std()), 4),
                }
                for metric in scoring
            }
            logger.info(f"  {name}: {results[name]}")
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            results[name] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def risk_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    clf = pipeline.named_steps.get("clf") or pipeline.named_steps.get("reg")
    if clf is None:
        return pd.DataFrame()

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_).flatten()
    else:
        return pd.DataFrame()

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full Phase 5 pipeline
# ---------------------------------------------------------------------------

def run_phase5(
    demo_df: pd.DataFrame,
    regional_volumes: Optional[pd.DataFrame] = None,
    beta_df: Optional[pd.DataFrame] = None,
    deficit_scores: Optional[pd.DataFrame] = None,
    target: str = "cdr_increase",
    out_dir: Optional[Path] = None,
) -> Dict:
    if out_dir is None:
        out_dir = get_outputs_dir("phase5")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_df, feature_names = build_risk_features(
        demo_df, regional_volumes, beta_df, deficit_scores
    )
    y_series = build_risk_labels(demo_df, beta_df, target=target)

    # Align on subject_id
    common = feat_df.index.intersection(y_series.index)
    if len(common) < 10:
        logger.warning(f"Only {len(common)} subjects with both features and labels.")
    feat_df = feat_df.loc[common]
    y_series = y_series.loc[common]

    X = feat_df.values.astype(float)
    y = y_series.values
    groups = np.array(common)

    task = "regression" if target == "beta" else "classification"
    cv_results = evaluate_risk_model(X, y, groups, task=task)

    # Fit final model for feature importances
    pipelines = build_risk_pipelines(task=task)
    best_name = list(pipelines.keys())[0]
    pipe = pipelines[best_name]
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    if valid_mask.sum() > 5:
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X[valid_mask])
        pipe.fit(X_imp, y[valid_mask])
        importances = risk_feature_importance(pipe, feature_names)
        if not importances.empty:
            importances.to_csv(out_dir / "risk_feature_importances.csv", index=False)
            logger.info(f"Top risk features:\n{importances.head(10).to_string()}")
    else:
        importances = pd.DataFrame()

    with open(out_dir / "risk_cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    # Save commentary note
    with open(out_dir / "risk_model_note.txt", "w") as f:
        f.write(
            "IMPORTANT: This is a CLINICAL RISK STRATIFIER, not a genomic/molecular model.\n"
            "OASIS-2 has no APOE genotype or other molecular data.\n"
            "This plays an analogous 'combine non-imaging risk info with imaging' role\n"
            "to the radiogenomics stage in the tumor pipeline, using clinical features\n"
            "(Age, Sex, Education, SES, eTIV, nWBV) as proxies.\n"
            "OASIS-3 includes APOE data and could support a genuinely molecular extension\n"
            "-- see implementation.md §7 for discussion.\n"
        )

    logger.info(f"Phase 5 complete. Outputs in: {out_dir}")
    return {
        "cv_results": cv_results,
        "feature_importances": importances,
        "feature_names": feature_names,
    }

"""
progression_model.py — Phase 6: Sustained Progression vs. Fluctuation Classification.

Analog to pseudoprogression vs. true progression discrimination in the tumor pipeline.

Key design decisions (from implementation.md §8):
  - Target is defined from the CDR TRAJECTORY, not the static Group label.
    "Sustained progression": monotonically non-decreasing CDR across visits that
    crosses into dementia territory (CDR ≥ 0.5).
    "Fluctuation": CDR changes non-monotonically or fluctuates without net sustained increase.
  - Features: longitudinal Δ in regional volumes + Δ MMSE + NDM β (spread rate)
    + demographics.
  - Class imbalance is SEVERE — use class-weighted loss, stratified CV,
    and bootstrap confidence intervals for sensitivity/specificity.
  - DO NOT claim clinical-grade performance from ~150 subjects.
    Report as exploratory pattern-finding.

"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, f1_score,
)
from sklearn.utils.class_weight import compute_class_weight



logger = setup_logging("progression_model")
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# CDR trajectory labeling
# ---------------------------------------------------------------------------

def label_cdr_trajectory(
    demo_df: pd.DataFrame,
    cdr_threshold: float = 0.5,
    min_visits: int = 2,
) -> pd.DataFrame:
    """
    Classify each subject's CDR trajectory as 'sustained_progression' or 'fluctuation'.

    Sustained progression: CDR is monotonically non-decreasing AND reaches >= cdr_threshold.
    Fluctuation: any other pattern (CDR goes up then down, or never reaches threshold).

    Returns a DataFrame with columns: subject_id, trajectory_label, n_visits,
    cdr_values, is_monotone, max_cdr, final_cdr.
    """
    demo_df = demo_df.copy()
    if "subject_id" not in demo_df.columns and "Subject ID" in demo_df.columns:
        demo_df["subject_id"] = demo_df["Subject ID"].str.upper().str.strip()

    rows = []
    for subj, grp in demo_df.groupby("subject_id"):
        if "MR Delay" in grp.columns:
            grp = grp.sort_values("MR Delay")
        elif "Visit" in grp.columns:
            grp = grp.sort_values("Visit")

        if "CDR" not in grp.columns:
            continue
        cdr_vals = grp["CDR"].dropna().values
        if len(cdr_vals) < min_visits:
            continue

        is_monotone = all(cdr_vals[i] <= cdr_vals[i + 1] for i in range(len(cdr_vals) - 1))
        max_cdr = float(cdr_vals.max())
        final_cdr = float(cdr_vals[-1])

        # Sustained progression: monotone AND reaches dementia threshold
        if is_monotone and max_cdr >= cdr_threshold:
            label = 1  # sustained progression
        else:
            label = 0  # fluctuation / stable

        rows.append({
            "subject_id": subj,
            "trajectory_label": label,
            "n_visits": len(cdr_vals),
            "cdr_values": ";".join(str(round(v, 2)) for v in cdr_vals),
            "is_monotone": is_monotone,
            "max_cdr": max_cdr,
            "final_cdr": final_cdr,
            "cdr_range": float(cdr_vals.max() - cdr_vals.min()),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        counts = df["trajectory_label"].value_counts()
        logger.info(
            f"CDR trajectory labels: sustained_progression={counts.get(1, 0)}, "
            f"fluctuation={counts.get(0, 0)}"
        )
        if counts.get(1, 0) < 5:
            logger.warning(
                "Very few sustained-progression cases (<5). "
                "Phase 6 results should be treated as exploratory, not diagnostic."
            )
    return df


# ---------------------------------------------------------------------------
# Longitudinal delta features
# ---------------------------------------------------------------------------

def compute_longitudinal_features(
    demo_df: pd.DataFrame,
    regional_volumes: Optional[pd.DataFrame] = None,
    beta_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build longitudinal CHANGE features per subject:
      - Δ MMSE (last - first)
      - Δ CDR (last - first)
      - Δ regional volumes (last - first, for each region)
      - NDM β (rate of spread — from Phase 3)
      - Clinical demographics (Age, Sex, EDUC, SES, eTIV, nWBV at baseline)

    Returns a DataFrame indexed by subject_id.
    """
    demo_df = demo_df.copy()
    if "subject_id" not in demo_df.columns and "Subject ID" in demo_df.columns:
        demo_df["subject_id"] = demo_df["Subject ID"].str.upper().str.strip()

    feature_rows = []

    for subj, grp in demo_df.groupby("subject_id"):
        if "MR Delay" in grp.columns:
            grp = grp.sort_values("MR Delay")
        elif "Visit" in grp.columns:
            grp = grp.sort_values("Visit")

        if len(grp) < 2:
            continue

        row = {"subject_id": subj}

        # MMSE delta
        if "MMSE" in grp.columns:
            valid_mmse = grp["MMSE"].dropna()
            if len(valid_mmse) >= 2:
                row["delta_mmse"] = float(valid_mmse.iloc[-1] - valid_mmse.iloc[0])

        # CDR delta
        if "CDR" in grp.columns:
            valid_cdr = grp["CDR"].dropna()
            if len(valid_cdr) >= 2:
                row["delta_cdr"] = float(valid_cdr.iloc[-1] - valid_cdr.iloc[0])

        # Baseline demographics
        first = grp.iloc[0]
        for col in ["Age", "EDUC", "SES", "eTIV", "nWBV"]:
            if col in first:
                row[col] = first[col]

        if "M/F" in first:
            row["sex_encoded"] = 1.0 if str(first["M/F"]).upper() == "M" else 0.0

        # Time span (useful proxy for disease duration observed)
        if "MR Delay" in grp.columns:
            row["observation_span_days"] = float(grp["MR Delay"].max() - grp["MR Delay"].min())

        feature_rows.append(row)

    feat_df = pd.DataFrame(feature_rows).set_index("subject_id")

    # Merge regional volume deltas
    if regional_volumes is not None and not regional_volumes.empty:
        vol_cols = [c for c in regional_volumes.columns
                    if c.endswith("_norm") or (
                        c not in {"subject_id", "session_id"} and
                        pd.api.types.is_numeric_dtype(regional_volumes[c])
                    )]
        for subj, grp in regional_volumes.groupby("subject_id"):
            if len(grp) < 2:
                continue
            # Delta: last - first (sorted by session_id alphabetically as proxy)
            grp_sorted = grp.sort_values("session_id") if "session_id" in grp.columns else grp
            for col in vol_cols:
                valid = grp_sorted[col].dropna()
                if len(valid) >= 2:
                    delta_val = float(valid.iloc[-1] - valid.iloc[0])
                    if subj in feat_df.index:
                        feat_df.loc[subj, f"delta_{col}"] = delta_val

    # Merge NDM β
    if beta_df is not None and not beta_df.empty and "beta" in beta_df.columns:
        beta_med = beta_df.groupby("subject_id")["beta"].median()
        feat_df = feat_df.join(beta_med.rename("ndm_beta"), how="left")

    logger.info(f"Longitudinal feature matrix: {feat_df.shape}")
    return feat_df


# ---------------------------------------------------------------------------
# Model pipelines
# ---------------------------------------------------------------------------

def build_progression_pipelines(class_weight: Optional[Dict] = None) -> Dict[str, Pipeline]:
    cw = class_weight or "balanced"
    return {
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05,
                max_depth=2, random_state=42,
            )),
        ]),
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight=cw, random_state=42,
            )),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=100, class_weight=cw, random_state=42,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals for sensitivity/specificity
# ---------------------------------------------------------------------------

def bootstrap_sens_spec(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict:
    """
    Compute sensitivity (TPR) and specificity (TNR) with bootstrap 95% CIs.
    """
    sensitivities = []
    specificities = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel() if len(np.unique(yt)) > 1 else (0, 0, 0, 0)
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        sensitivities.append(sens)
        specificities.append(spec)

    alpha = (1 - confidence) / 2

    def ci(vals):
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) < 10:
            return {"mean": np.nan, "lower": np.nan, "upper": np.nan}
        return {
            "mean": round(float(np.mean(vals)), 4),
            "lower": round(float(np.percentile(vals, 100 * alpha)), 4),
            "upper": round(float(np.percentile(vals, 100 * (1 - alpha))), 4),
        }

    return {
        "sensitivity": ci(sensitivities),
        "specificity": ci(specificities),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Cross-validation (Leave-One-Subject-Out or repeated stratified k-fold)
# ---------------------------------------------------------------------------

def evaluate_progression_model(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_splits: int = 5,
    use_logo: bool = False,
) -> Dict:
    """
    Evaluate progression classifiers with leave-one-subject-out (LOGO) or
    stratified k-fold CV.

    Per implementation.md §8: class imbalance is severe. Use class-weighted loss
    and report sensitivity/specificity with bootstrap CIs.
    """
    pipelines = build_progression_pipelines(class_weight="balanced")
    results = {}

    if use_logo or len(np.unique(subject_ids)) <= n_splits * 2:
        # Leave-one-subject-out (best for very small n)
        cv_strategy = LeaveOneGroupOut()
        cv_kwargs = {"groups": subject_ids}
        logger.info("Using Leave-One-Subject-Out CV (small n).")
    else:
        n_splits = min(n_splits, len(np.unique(subject_ids)) // 2)
        cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_kwargs = {}
        logger.info(f"Using StratifiedKFold CV (n_splits={n_splits}).")

    for name, pipe in pipelines.items():
        all_true = []
        all_pred = []

        if use_logo or len(np.unique(subject_ids)) <= n_splits * 2:
            for train_idx, test_idx in cv_strategy.split(X, y, **cv_kwargs):
                pipe_clone = sklearn_clone(pipe)
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                if len(np.unique(y_tr)) < 2:
                    continue
                try:
                    pipe_clone.fit(X_tr, y_tr)
                    preds = pipe_clone.predict(X_te)
                    all_true.extend(y_te.tolist())
                    all_pred.extend(preds.tolist())
                except Exception as e:
                    logger.debug(f"  fold failed: {e}")
        else:
            for train_idx, test_idx in cv_strategy.split(X, y):
                pipe_clone = sklearn_clone(pipe)
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                if len(np.unique(y_tr)) < 2:
                    continue
                try:
                    pipe_clone.fit(X_tr, y_tr)
                    preds = pipe_clone.predict(X_te)
                    all_true.extend(y_te.tolist())
                    all_pred.extend(preds.tolist())
                except Exception as e:
                    logger.debug(f"  fold failed: {e}")

        if len(all_true) < 5:
            results[name] = {"error": "insufficient folds"}
            continue

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)

        ci_result = bootstrap_sens_spec(all_true, all_pred)
        acc = float((all_true == all_pred).mean())

        try:
            auc = float(roc_auc_score(all_true, all_pred))
        except Exception:
            auc = np.nan

        results[name] = {
            "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
            **ci_result,
            "n_positive": int(all_true.sum()),
            "n_negative": int((all_true == 0).sum()),
        }
        logger.info(
            f"  {name}: accuracy={acc:.3f}, "
            f"sensitivity={ci_result['sensitivity']['mean']:.3f} "
            f"[{ci_result['sensitivity']['lower']:.3f}–{ci_result['sensitivity']['upper']:.3f}], "
            f"specificity={ci_result['specificity']['mean']:.3f}"
        )

    return results


def sklearn_clone(estimator):
    """Safely clone a scikit-learn estimator."""
    from sklearn.base import clone
    return clone(estimator)


# ---------------------------------------------------------------------------
# Full Phase 6 pipeline
# ---------------------------------------------------------------------------

def run_phase6(
    demo_df: pd.DataFrame,
    regional_volumes: Optional[pd.DataFrame] = None,
    beta_df: Optional[pd.DataFrame] = None,
    out_dir: Optional[Path] = None,
) -> Dict:
    if out_dir is None:
        out_dir = get_outputs_dir("phase6")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Label CDR trajectories
    trajectory_df = label_cdr_trajectory(demo_df)
    trajectory_df.to_csv(out_dir / "cdr_trajectories.csv", index=False)

    if trajectory_df.empty or trajectory_df["trajectory_label"].nunique() < 2:
        logger.warning("Not enough trajectory diversity for Phase 6 classification.")
        return {"error": "insufficient class diversity"}

    # Step 2: Build features
    feat_df = compute_longitudinal_features(demo_df, regional_volumes, beta_df)

    # Align on subject_id
    traj_indexed = trajectory_df.set_index("subject_id")
    common = feat_df.index.intersection(traj_indexed.index)
    if len(common) < 10:
        logger.warning(f"Only {len(common)} subjects with both features and labels.")
        return {"error": "insufficient subjects", "n_subjects": len(common)}

    X = feat_df.loc[common].values.astype(float)
    y = traj_indexed.loc[common, "trajectory_label"].values
    subject_ids = np.array(common)
    feature_names = list(feat_df.columns)

    feat_df.loc[common].to_csv(out_dir / "progression_features.csv")

    # Step 3: Evaluate
    n_pos = y.sum()
    n_neg = (y == 0).sum()
    logger.info(f"Phase 6 class distribution: sustained={n_pos}, fluctuation={n_neg}")

    use_logo = len(common) < 30
    cv_results = evaluate_progression_model(X, y, subject_ids, use_logo=use_logo)

    with open(out_dir / "progression_cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    # Step 4: Limitations report
    limitations = {
        "n_subjects": len(common),
        "n_sustained_progression": int(n_pos),
        "n_fluctuation": int(n_neg),
        "warning": (
            "EXPLORATORY ONLY. Sample size is too small for a validated diagnostic tool. "
            "Sustained-progression class is likely << 50 subjects. "
            "Results should not be used clinically without independent replication. "
            "See implementation.md §8 for full discussion."
        ),
    }
    with open(out_dir / "limitations.json", "w") as f:
        json.dump(limitations, f, indent=2)

    logger.info(f"Phase 6 complete. Outputs in: {out_dir}")
    return {
        "trajectory_df": trajectory_df,
        "cv_results": cv_results,
        "feature_names": feature_names,
        "limitations": limitations,
    }

# ── Environment summary + GPU check ────────────────────────────────────
# All pipeline functions were defined directly above (Section 1) — nothing
# to import, they're already in this notebook's namespace.
logger = setup_logging('colab')
env = environment_summary()
print('Environment:')
for k, v in env.items():
    print(f'  {k}: {v}')

import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"\n[GPU] {gpu_name} detected ({gpu_mem_gb:.1f} GB). Phase 2b CNN training will use the GPU.")
else:
    print(
        "\n[WARNING] No GPU detected — PyTorch will fall back to CPU, which is "
        "much slower for Phase 2b (CNN).\n"
        "          Fix: Runtime menu -> Change runtime type -> Hardware accelerator -> "
        "T4 GPU -> Save, then Runtime -> Restart session and re-run all cells."
    )

# ── 2.1  Kaggle credentials ──────────────────────────────────────────
import shutil, json as _json
from google.colab import userdata

KAGGLE_JSON_DRIVE = DRIVE_BASE / 'kaggle.json'
KAGGLE_DIR = Path(os.path.expanduser('~/.kaggle'))
KAGGLE_DIR.mkdir(exist_ok=True)

has_secrets = False
try:
    kaggle_user = userdata.get('KAGGLE_USERNAME')
    kaggle_key = userdata.get('KAGGLE_KEY')
    if kaggle_user and kaggle_key:
        dest = KAGGLE_DIR / 'kaggle.json'
        dest.write_text(_json.dumps({'username': kaggle_user, 'key': kaggle_key}))
        dest.chmod(0o600)
        print('[OK] Kaggle credentials loaded from Colab Secrets.')
        has_secrets = True
except Exception as e:
    # Covers SecretNotFoundError (secret not set) and NotebookAccessError
    # (secret exists but access wasn't granted) alike.
    print(f'[INFO] Colab Secrets not available/usable ({type(e).__name__}); falling back to Drive/upload.')

if not has_secrets:
    if KAGGLE_JSON_DRIVE.exists():
        shutil.copy(str(KAGGLE_JSON_DRIVE), str(KAGGLE_DIR / 'kaggle.json'))
        (KAGGLE_DIR / 'kaggle.json').chmod(0o600)
        print('[OK] Kaggle credentials loaded from Drive.')
    else:
        print('First time: upload your kaggle.json file (or set Colab secrets and restart cell).')
        from google.colab import files
        uploaded = files.upload()
        if 'kaggle.json' in uploaded:
            dest = KAGGLE_DIR / 'kaggle.json'
            dest.write_bytes(uploaded['kaggle.json'])
            dest.chmod(0o600)
            shutil.copy(str(dest), str(KAGGLE_JSON_DRIVE))  # save to Drive
            print('[OK] Credentials saved to Drive for future sessions.')
        else:
            print('[ERROR] kaggle.json not uploaded. Cannot download dataset.')

# ── 2.2  Download dataset (skips if already on Drive) ────────────────
RAW_DRIVE = DRIVE_BASE / 'data' / 'raw'
RAW_DRIVE.mkdir(parents=True, exist_ok=True)
RAW_LOCAL = LOCAL_BASE / 'data' / 'raw'

KAGGLE_DATASET = 'nadiatriki/oasis-2-longitudinal-scan-data'

existing = list(RAW_DRIVE.rglob('oasis_longitudinal*.csv')) + list(RAW_DRIVE.rglob('oasis_longitudinal*.xlsx'))
if existing:
    print(f'[OK] Dataset already on Drive: {existing[0].name}')
else:
    print(f'Downloading {KAGGLE_DATASET} from Kaggle (this takes a few minutes)...')
    from kaggle.api.kaggle_api_extended import KaggleApi

    try:
        api = KaggleApi()
        api.authenticate()
        # quiet=False shows a live tqdm progress bar for each file as it downloads.
        api.dataset_download_files(
            KAGGLE_DATASET, path=str(RAW_DRIVE), unzip=True, quiet=False,
        )
        print('[OK] Download complete.')
    except Exception as e:
        print(f'[ERROR] Kaggle download failed: {type(e).__name__}: {e}')
        print('        Double-check your kaggle.json credentials (previous cell) and that')
        print(f'        you have accepted the dataset terms at:')
        print(f'        https://www.kaggle.com/datasets/{KAGGLE_DATASET}')

# Always show what actually landed on Drive, so failures are obvious immediately.
all_files = sorted(RAW_DRIVE.rglob('*'))
print(f'\n{len(all_files)} items now under {RAW_DRIVE}:')
for p in all_files[:30]:
    print(' ', p.relative_to(RAW_DRIVE))
if len(all_files) > 30:
    print(f'  ... and {len(all_files) - 30} more')

# Symlink Drive path to local fast path
if RAW_LOCAL.exists() and not RAW_LOCAL.is_symlink():
    shutil.rmtree(str(RAW_LOCAL))
if not RAW_LOCAL.exists():
    RAW_LOCAL.symlink_to(RAW_DRIVE)
print(f'\n[OK] Linked to local: {RAW_LOCAL}')

# ── 2.3  Verify download ─────────────────────────────────────────────
import pandas as pd

raw_dir = LOCAL_BASE / 'data' / 'raw'
csvs = sorted(raw_dir.rglob('oasis_longitudinal*.csv')) or sorted(raw_dir.rglob('*.csv'))
xlsxs = sorted(raw_dir.rglob('oasis_longitudinal*.xlsx')) or sorted(raw_dir.rglob('*.xlsx'))
nii_count = len(list(raw_dir.rglob('*.nii.gz'))) + len(list(raw_dir.rglob('*.hdr')))

print(f'CSV files:   {len(csvs)}')
print(f'Excel files: {len(xlsxs)}')
print(f'Scan files:  {nii_count}')

demo_csv = (csvs or xlsxs or [None])[0]
if demo_csv is None:
    all_files = sorted(raw_dir.rglob('*'))
    raise FileNotFoundError(
        'No demographics CSV/XLSX found in data/raw/.\n'
        f'Contents of {raw_dir}:\n' +
        '\n'.join(f'  {p.relative_to(raw_dir)}' for p in all_files[:30]) +
        '\nPlease check the download cell above for a [ERROR] message.'
    )

demo_df = pd.read_excel(demo_csv) if demo_csv.suffix.lower() in ('.xlsx', '.xls') else pd.read_csv(demo_csv)
demo_df.columns = [c.strip() for c in demo_df.columns]
if 'subject_id' not in demo_df.columns and 'Subject ID' in demo_df.columns:
    demo_df['subject_id'] = demo_df['Subject ID'].str.upper().str.strip()
print(f'Demographics: {demo_df.shape[0]} rows x {demo_df.shape[1]} cols  (from {demo_csv.name})')
print(demo_df.head(3).to_string())

import json, numpy as np, matplotlib.pyplot as plt

outputs_dir = LOCAL_BASE / 'outputs'
outputs_dir.mkdir(parents=True, exist_ok=True)
(DRIVE_BASE / 'outputs').mkdir(parents=True, exist_ok=True)

demo_csv_path = find_demographic_csv(raw_dir)
index_df, quality = build_subjects_index(
    raw_dir=raw_dir,
    demo_csv=demo_csv_path,
    out_path=outputs_dir / 'subjects_index.csv',
)
index_df.to_csv(DRIVE_BASE / 'outputs' / 'subjects_index.csv', index=False)
print(f'Index shape: {index_df.shape}')
print(json.dumps(quality, indent=2, default=str))

# Phase 0 — distribution plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('OASIS-2 Dataset Overview', fontsize=14, fontweight='bold')
if 'Group' in demo_df.columns:
    gc = demo_df.groupby('subject_id')['Group'].last().value_counts()
    axes[0].bar(gc.index, gc.values, color=['#4CAF50','#F44336','#FF9800'][:len(gc)])
    axes[0].set_title('Subjects by Group'); axes[0].set_ylabel('Count')
if 'Age' in demo_df.columns:
    demo_df.groupby('subject_id')['Age'].first().hist(
        bins=15, ax=axes[1], color='#9C27B0', alpha=0.8)
    axes[1].set_title('Age Distribution'); axes[1].set_xlabel('Age')
if 'CDR' in demo_df.columns:
    cc = demo_df['CDR'].value_counts().sort_index()
    axes[2].bar(cc.index.astype(str), cc.values, color='#2196F3')
    axes[2].set_title('CDR Score Distribution'); axes[2].set_xlabel('CDR')
plt.tight_layout()
fig.savefig(str(outputs_dir / 'phase0_overview.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Phase 0 complete.')

# Check for SynthSeg (mri_synthseg). The real SynthSeg package is not on PyPI
# under a pip-installable name and requires its own model weights, so we don't
# attempt an automatic install here (it would just fail). If you want true
# SynthSeg parcellation, follow https://github.com/BBillot/SynthSeg to install
# it manually in this runtime, then re-run this cell.
import shutil as _sh
if _sh.which('mri_synthseg'):
    print('[OK] mri_synthseg available.')
else:
    print('[INFO] mri_synthseg not found on PATH.')
    print('       Phase 1 will fall back to FSL_SEG volumes where available,')
    print('       or to demographic-only features otherwise.')
    print('       Phases 2-6 will still run normally using whatever features are available.')


processed_dir = DRIVE_BASE / 'data' / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

regional_volumes = build_regional_volumes(
    subjects_index=index_df,
    demo_df=demo_df,
    out_path=outputs_dir / 'regional_volumes.csv',
    processed_dir=processed_dir,
)

if regional_volumes.empty:
    print('[WARN] No SynthSeg output — using demographics only for Phases 2-6.')
    # IMPORTANT: keep this to just the join key. Copying demo_df here would make
    # every downstream regional_volumes.merge(demo_df, on='subject_id') collide
    # with itself (pandas would suffix duplicate columns as e.g. CDR_x/CDR_y and
    # the plain "CDR"/"Group" columns the phases look for would disappear).
    key_cols = [c for c in ['subject_id', 'session_id'] if c in demo_df.columns]
    regional_volumes = demo_df[key_cols].drop_duplicates().reset_index(drop=True)
else:
    regional_volumes.to_csv(DRIVE_BASE / 'outputs' / 'regional_volumes.csv', index=False)
    print(f'[OK] regional_volumes: {regional_volumes.shape}')
print('Phase 1 complete.')


p2a_results = run_phase2a(
    regional_volumes=regional_volumes,
    demo_df=demo_df,
    out_dir=outputs_dir / 'phase2a',
    n_splits=5,
)

print('=== Phase 2a CV Results ===')
print(json.dumps(p2a_results['cv_results'], indent=2))
print(f'Best model: {p2a_results["best_model_name"]}')

(DRIVE_BASE / 'outputs' / 'phase2a').mkdir(parents=True, exist_ok=True)
with open(DRIVE_BASE / 'outputs' / 'phase2a' / 'cv_results.json', 'w') as f:
    json.dump(p2a_results['cv_results'], f, indent=2)
print('Phase 2a complete.')

# Phase 2a — Feature importance plot
imp = p2a_results.get('feature_importances', pd.DataFrame())
if not imp.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    top = imp.head(15)
    ax.barh(top['feature'], top['importance'],
            color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top))))
    ax.set_xlabel('Importance'); ax.invert_yaxis()
    ax.set_title(f'Phase 2a Feature Importances — {p2a_results["best_model_name"]}')
    plt.tight_layout()
    fig.savefig(str(outputs_dir / 'phase2a' / 'feature_importances.png'), dpi=150)
    plt.show()

import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')

nifti_paths, labels_cnn, sids_cnn = [], [], []
if 'scan_files' in index_df.columns and 'CDR' in demo_df.columns:
    cdr_map = demo_df.groupby('subject_id')['CDR'].first().to_dict()
    for _, row in index_df.iterrows():
        scans = str(row.get('scan_files', ''))
        if not scans: continue
        p = Path(scans.split(';')[0])
        if not p.exists(): continue
        sid = str(row.get('subject_id', ''))
        cdr = cdr_map.get(sid, float('nan'))
        if not pd.isna(cdr):
            nifti_paths.append(p); labels_cnn.append(int(cdr > 0)); sids_cnn.append(sid)

print(f'NIfTI scans available: {len(nifti_paths)}')
RUN_CNN = len(nifti_paths) >= 10
if not RUN_CNN:
    print('Skipping Phase 2b — insufficient NIfTI files.')
    print('Phase 2a (classical ML) is the primary result.')

if RUN_CNN:
    cnn_results = run_cnn_cv(
        nifti_paths=nifti_paths, labels=labels_cnn, subject_ids=sids_cnn,
        n_splits=5, backbone='resnet18', n_epochs=10,
        batch_size=16, n_slices_per_scan=10,
        out_dir=outputs_dir / 'phase2b',
    )
    print(json.dumps(cnn_results, indent=2))
    (DRIVE_BASE / 'outputs' / 'phase2b').mkdir(parents=True, exist_ok=True)
    with open(DRIVE_BASE / 'outputs' / 'phase2b' / 'cnn_results.json', 'w') as f:
        json.dump(cnn_results, f, indent=2)
else:
    cnn_results = {'skipped': True}
print('Phase 2b complete.')


p3_results = run_phase3(
    regional_volumes=regional_volumes,
    demo_df=demo_df,
    out_dir=outputs_dir / 'phase3',
)

beta_df     = p3_results['beta_df']
eigenvalues = p3_results['eigenvalues']
eigenvectors= p3_results['eigenvectors']
w_scores    = p3_results['w_scores']

print(f'Beta estimates: {len(beta_df)} subjects')
print('NDM validation:', json.dumps(p3_results['validation'], indent=2))

(DRIVE_BASE / 'outputs' / 'phase3').mkdir(parents=True, exist_ok=True)
if not beta_df.empty:
    beta_df.to_csv(DRIVE_BASE / 'outputs' / 'phase3' / 'fitted_betas.csv', index=False)
    w_scores.to_csv(DRIVE_BASE / 'outputs' / 'phase3' / 'w_scores.csv', index=False)
np.save(str(DRIVE_BASE / 'outputs' / 'phase3' / 'eigenvalues.npy'), eigenvalues)
np.save(str(DRIVE_BASE / 'outputs' / 'phase3' / 'eigenvectors.npy'), eigenvectors)
print('Phase 3 complete.')

# Phase 3 — visualizations
if not beta_df.empty and 'beta' in beta_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase 3: Network Diffusion Model', fontsize=13, fontweight='bold')
    valid_b = beta_df['beta'].dropna()
    axes[0].hist(valid_b, bins=20, color='#9C27B0', alpha=0.85, edgecolor='white')
    axes[0].axvline(valid_b.median(), color='red', linestyle='--',
                    label=f'Median beta={valid_b.median():.3f}')
    axes[0].set_xlabel('Diffusivity beta'); axes[0].set_ylabel('Count')
    axes[0].set_title('Fitted beta per subject'); axes[0].legend()
    # 5-year prediction for highest-beta subject
    top = beta_df.dropna(subset=['beta']).sort_values('beta', ascending=False).iloc[0]
    wscore_cols = [c for c in w_scores.columns if c.endswith('_wscore')]
    ws = w_scores[w_scores['subject_id'] == top['subject_id']]
    if not ws.empty and wscore_cols:
        x0 = ws.iloc[0][wscore_cols].values.astype(float)
        n  = min(len(x0), len(eigenvalues))
        x5 = ndm_propagate(x0[:n], 5.0, top['beta'], eigenvalues[:n], eigenvectors[:n,:n])
        axes[1].bar(range(n), x5, color=['#F44336' if v>0 else '#4CAF50' for v in x5], alpha=0.8)
        axes[1].axhline(0, color='black', linewidth=1)
        axes[1].set_xlabel('Brain Region Index'); axes[1].set_ylabel('Atrophy (W-score)')
        axes[1].set_title(f'5-yr Prediction — {top["subject_id"]} (beta={top["beta"]:.3f})')
    plt.tight_layout()
    fig.savefig(str(outputs_dir / 'phase3' / 'ndm_results.png'), dpi=150, bbox_inches='tight')
    plt.show()


p3_results['region_names'] = CONNECTOME_REGIONS

p4_results = run_phase4(
    ndm_results=p3_results,
    demo_df=demo_df,
    out_dir=outputs_dir / 'phase4',
)

deficit_df = p4_results.get('deficit_scores', pd.DataFrame())
print(f'Deficit scores: {deficit_df.shape}')
if not deficit_df.empty:
    print(deficit_df.mean().sort_values(ascending=False).round(3))

(DRIVE_BASE / 'outputs' / 'phase4').mkdir(parents=True, exist_ok=True)
if not deficit_df.empty:
    deficit_df.to_csv(DRIVE_BASE / 'outputs' / 'phase4' / 'deficit_scores.csv')

print('Validation (Pearson r with MMSE/CDR):')
for domain, outcomes in p4_results.get('validation', {}).items():
    for outcome, m in outcomes.items():
        print(f'  {domain:25s} x {outcome}: r={m["pearson_r"]:+.3f}')
print('Phase 4 complete.')

# Phase 4 — deficit profile visualization
if not deficit_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    mean_d = deficit_df.mean().sort_values(ascending=False)
    axes[0].bar(range(len(mean_d)), mean_d.values,
                color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(mean_d))))
    axes[0].set_xticks(range(len(mean_d)))
    axes[0].set_xticklabels(mean_d.index, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylabel('Mean deficit'); axes[0].set_title('Population-avg Deficit Profiles')
    if len(deficit_df) > 1:
        s = deficit_df.sample(min(20, len(deficit_df)), random_state=42)
        im = axes[1].imshow(s.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[1].set_yticks(range(len(s))); axes[1].set_yticklabels(s.index, fontsize=7)
        axes[1].set_xticks(range(len(deficit_df.columns)))
        axes[1].set_xticklabels(deficit_df.columns, rotation=45, ha='right', fontsize=8)
        plt.colorbar(im, ax=axes[1], label='Deficit score')
        axes[1].set_title('Per-subject deficit profiles (sample 20)')
    plt.tight_layout()
    fig.savefig(str(outputs_dir / 'phase4' / 'deficit_profiles.png'), dpi=150)
    plt.show()


p5_results = run_phase5(
    demo_df=demo_df,
    regional_volumes=regional_volumes if not regional_volumes.empty else None,
    beta_df=beta_df if not beta_df.empty else None,
    deficit_scores=deficit_df if not deficit_df.empty else None,
    target='cdr_increase',
    out_dir=outputs_dir / 'phase5',
)

print('=== Phase 5 Risk CV Results ===')
print(json.dumps(p5_results['cv_results'], indent=2))

(DRIVE_BASE / 'outputs' / 'phase5').mkdir(parents=True, exist_ok=True)
with open(DRIVE_BASE / 'outputs' / 'phase5' / 'risk_cv_results.json', 'w') as f:
    json.dump(p5_results['cv_results'], f, indent=2)

imp5 = p5_results.get('feature_importances', pd.DataFrame())
if not imp5.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    top = imp5.head(15)
    ax.barh(top['feature'], top['importance'],
            color=plt.cm.plasma(np.linspace(0.2, 0.85, len(top))))
    ax.invert_yaxis(); ax.set_xlabel('Importance')
    ax.set_title('Phase 5: Risk Feature Importances (clinical proxy)')
    plt.tight_layout()
    fig.savefig(str(outputs_dir / 'phase5' / 'risk_importances.png'), dpi=150)
    plt.show()
print('Phase 5 complete.')


traj_preview = label_cdr_trajectory(demo_df)
print('Trajectory distribution:')
print(traj_preview['trajectory_label'].value_counts())

p6_results = run_phase6(
    demo_df=demo_df,
    regional_volumes=regional_volumes if not regional_volumes.empty else None,
    beta_df=beta_df if not beta_df.empty else None,
    out_dir=outputs_dir / 'phase6',
)

print('=== Phase 6 CV Results ===')
print(json.dumps(p6_results.get('cv_results', {}), indent=2))
print('Limitations:', json.dumps(p6_results.get('limitations', {}), indent=2))

(DRIVE_BASE / 'outputs' / 'phase6').mkdir(parents=True, exist_ok=True)
with open(DRIVE_BASE / 'outputs' / 'phase6' / 'progression_cv_results.json', 'w') as f:
    json.dump(p6_results.get('cv_results', {}), f, indent=2)
if 'trajectory_df' in p6_results:
    p6_results['trajectory_df'].to_csv(
        DRIVE_BASE / 'outputs' / 'phase6' / 'cdr_trajectories.csv', index=False)

# Visualization
traj_df = p6_results.get('trajectory_df', pd.DataFrame())
if not traj_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase 6: CDR Trajectory Classification', fontsize=13, fontweight='bold')
    counts = traj_df['trajectory_label'].value_counts()
    axes[0].bar(['Fluctuation (0)', 'Sustained (1)'],
                [counts.get(0,0), counts.get(1,0)], color=['#4CAF50','#F44336'])
    axes[0].set_ylabel('Subjects'); axes[0].set_title('Class distribution')
    if 'CDR' in demo_df.columns:
        dm = demo_df.merge(traj_df[['subject_id','trajectory_label']], on='subject_id', how='left')
        for lbl, col in [(1,'#F44336'),(0,'#4CAF50')]:
            for sid in traj_df[traj_df['trajectory_label']==lbl]['subject_id'].values[:8]:
                sd = dm[dm['subject_id']==sid].sort_values('MR Delay') if 'MR Delay' in dm.columns else dm[dm['subject_id']==sid]
                if 'CDR' in sd.columns:
                    axes[1].plot(sd.get('MR Delay', pd.Series(range(len(sd))))/365,
                                 sd['CDR'], color=col, alpha=0.55, linewidth=1.8)
        from matplotlib.lines import Line2D
        axes[1].legend(handles=[Line2D([0],[0],color='#F44336',linewidth=2,label='Sustained'),
                                 Line2D([0],[0],color='#4CAF50',linewidth=2,label='Fluctuation')])
        axes[1].set_xlabel('Years'); axes[1].set_ylabel('CDR')
        axes[1].set_title('Sample CDR trajectories')
    plt.tight_layout()
    fig.savefig(str(outputs_dir / 'phase6' / 'trajectory_plot.png'), dpi=150)
    plt.show()
print('Phase 6 complete.')

# Aggregate summary table
rows = []
for model, m in p2a_results.get('cv_results', {}).items():
    if 'accuracy' in m:
        rows.append({'Phase':'2a Classification','Method':model,
                     'Metric':f"Acc {m['accuracy']['mean']:.3f}+/-{m['accuracy']['std']:.3f}",
                     'Note':'GroupKFold, honest 60-75%'})
v3 = p3_results.get('validation',{})
if v3 and 'pearson_r' in v3:
    rows.append({'Phase':'3 NDM','Method':'NDM',
                 'Metric':f"Pearson r={v3['pearson_r']}", 'Note':'Raj2015: r~0.93 ADNI'})
for model, m in p6_results.get('cv_results',{}).items():
    if 'accuracy' in m:
        s=m.get('sensitivity',{}).get('mean','?')
        sp=m.get('specificity',{}).get('mean','?')
        rows.append({'Phase':'6 Progression','Method':model,
                     'Metric':f"Sens={s}, Spec={sp}", 'Note':'EXPLORATORY'})
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_csv(outputs_dir / 'aggregate_metrics.csv', index=False)
summary.to_csv(DRIVE_BASE / 'outputs' / 'aggregate_metrics.csv', index=False)

# Per-subject report
SUBJECT_ID = None  # set to e.g. 'OAS2_0001' or leave None for auto-select
if SUBJECT_ID is None and not beta_df.empty:
    SUBJECT_ID = beta_df['subject_id'].dropna().iloc[0]
if SUBJECT_ID:
    print(f'=== Per-Subject Report: {SUBJECT_ID} ===')
    sd = demo_df[demo_df['subject_id']==SUBJECT_ID]
    if not sd.empty:
        r = sd.iloc[0]
        print(f'  Age={r.get("Age","?")}  Group={r.get("Group","?")}  EDUC={r.get("EDUC","?")}')
    sb = beta_df[beta_df['subject_id']==SUBJECT_ID]['beta'].dropna()
    if len(sb): print(f'  NDM beta={sb.iloc[0]:.4f}')
    if not deficit_df.empty and SUBJECT_ID in deficit_df.index:
        print('  Top deficits:', dict(deficit_df.loc[SUBJECT_ID].sort_values(ascending=False).head(3).round(3)))
    tr = traj_preview[traj_preview['subject_id']==SUBJECT_ID]
    if not tr.empty:
        lbl = 'Sustained Progression' if tr['trajectory_label'].values[0]==1 else 'Fluctuation'
        print(f'  CDR trajectory: {lbl}')
        print(f'  CDR values: {tr["cdr_values"].values[0]}')

# Methods adaptation note
note = """
METHODS ADAPTATION NOTE
========================
Structural analogs between this OASIS-2 pipeline and the tumor source pipeline:

  nnU-Net/Swin UNETR (3D seg)  ->  SynthSeg T1 parcellation + volumetric ML
  Fisher-Kolmogorov PDE + DTI  ->  Network Diffusion Model (Raj et al. 2012)
  PINN / adjoint inverse        ->  scipy.optimize least-squares beta fit
  Eloquent-cortex mapping       ->  Region->cognitive domain atlas
  Radiogenomics (IDH/MGMT)      ->  Clinical risk proxy (no genomics in OASIS-2)
  Pseudoprogression vs. TP      ->  CDR: sustained monotone vs. fluctuation
"""
print(note)
with open(outputs_dir / 'methods_note.txt', 'w') as f: f.write(note)
with open(DRIVE_BASE / 'outputs' / 'methods_note.txt', 'w') as f: f.write(note)
print(f'All outputs saved to Drive: {DRIVE_BASE / "outputs"}')
print('Pipeline complete.')
