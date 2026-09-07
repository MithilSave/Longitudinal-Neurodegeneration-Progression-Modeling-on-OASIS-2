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
dl_models.py — Deep Learning Models for Longitudinal Neurodegeneration Modeling.

Replaces legacy sklearn/CNN classifiers with three post-2022 architectures:
  1. ST-GNN-ODE: Continuous-Time Spatio-Temporal Graph Neural ODE
  2. ND-VAE:     Physics-Constrained Network Diffusion VAE
  3. TADM:       Temporally-Aware Trajectory Diffusion Model

All models operate on connectome graph structure and handle irregular
longitudinal visit intervals natively.
"""

import sys
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

logger = setup_logging("dl_models")
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# PyTorch imports
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Longitudinal Graph Dataset
# ---------------------------------------------------------------------------

class LongitudinalGraphDataset(Dataset):
    """
    Build DL-ready tensors from OASIS-2 demographics + connectome.

    Each sample is one subject with K_i visits:
      - node_features: (K_i, N, D) — regional atrophy features per visit
      - times: (K_i,) — elapsed years from baseline
      - covariates: (C,) — static clinical covariates
      - mmse: (K_i,) — MMSE scores per visit
      - cdr: (K_i,) — CDR labels per visit (0, 0.5, 1, 2 → 0,1,2,3)
      - adj: (N, N) — normalized adjacency matrix
    """
    CDR_MAP = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}

    def __init__(self, subjects_data, adj_matrix):
        self.subjects = subjects_data
        self.adj = torch.tensor(adj_matrix, dtype=torch.float32)
        self.n_regions = adj_matrix.shape[0]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        s = self.subjects[idx]
        return {
            'node_features': s['node_features'],
            'times': s['times'],
            'covariates': s['covariates'],
            'mmse': s['mmse'],
            'cdr': s['cdr'],
            'adj': self.adj,
            'subject_id': s['subject_id'],
        }


def collate_longitudinal(batch):
    """Custom collate — pad to max visits in batch."""
    max_k = max(b['times'].shape[0] for b in batch)
    n_regions = batch[0]['node_features'].shape[1]
    n_feat = batch[0]['node_features'].shape[2]
    n_cov = batch[0]['covariates'].shape[0]
    B = len(batch)

    node_features = torch.zeros(B, max_k, n_regions, n_feat)
    times = torch.zeros(B, max_k)
    covariates = torch.zeros(B, n_cov)
    mmse = torch.zeros(B, max_k)
    cdr = torch.full((B, max_k), -1, dtype=torch.long)
    mask = torch.zeros(B, max_k, dtype=torch.bool)
    adj = batch[0]['adj'].unsqueeze(0)  # shared across batch

    for i, b in enumerate(batch):
        k = b['times'].shape[0]
        node_features[i, :k] = b['node_features']
        times[i, :k] = b['times']
        covariates[i] = b['covariates']
        mmse[i, :k] = b['mmse']
        cdr[i, :k] = b['cdr']
        mask[i, :k] = True

    return {
        'node_features': node_features,
        'times': times,
        'covariates': covariates,
        'mmse': mmse,
        'cdr': cdr,
        'mask': mask,
        'adj': adj,
    }


def build_dl_datasets(
    demo_df: pd.DataFrame,
    w_scores: pd.DataFrame,
    adj_matrix: np.ndarray,
    region_names: List[str],
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders from demographics + w-scores + connectome.
    Subject-level 70/15/15 split.
    """
    rng = np.random.default_rng(seed)

    # Normalize adjacency: A_hat = D^{-1/2} (A + I) D^{-1/2}
    N = adj_matrix.shape[0]
    A_hat = adj_matrix + np.eye(N)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A_hat.sum(axis=1) + 1e-8))
    adj_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

    # Identify w-score columns
    wscore_cols = [c for c in w_scores.columns if c.endswith('_wscore')]
    if not wscore_cols:
        logger.warning("No w-score columns found. Using synthetic node features.")
        wscore_cols = []

    # Build per-subject data
    subjects_data = []
    demo_grouped = demo_df.copy()
    if 'subject_id' not in demo_grouped.columns and 'Subject ID' in demo_grouped.columns:
        demo_grouped['subject_id'] = demo_grouped['Subject ID'].str.upper().str.strip()

    for sid, grp in demo_grouped.groupby('subject_id'):
        grp = grp.sort_values('MR Delay' if 'MR Delay' in grp.columns else 'Visit')
        if len(grp) < 2:
            continue

        # Time vector (years from baseline)
        if 'MR Delay' in grp.columns:
            t_days = grp['MR Delay'].fillna(0).values.astype(float)
            t_years = t_days / 365.25
        else:
            t_years = np.arange(len(grp), dtype=float)
        t_years = t_years - t_years[0]  # baseline = 0

        K = len(grp)

        # Node features from w-scores
        if wscore_cols:
            ws_sub = w_scores[w_scores['subject_id'] == sid]
            if len(ws_sub) >= 1:
                x_vals = ws_sub.iloc[0][wscore_cols].values.astype(float)
                x_vals = np.nan_to_num(x_vals, nan=0.0)
                n_ws = min(len(x_vals), N)
                # Repeat baseline w-scores for each visit, modulated by time
                node_feat = np.zeros((K, N, 1))
                for ki in range(K):
                    decay = np.exp(-0.1 * t_years[ki])
                    node_feat[ki, :n_ws, 0] = x_vals[:n_ws] * decay
            else:
                node_feat = np.zeros((K, N, 1))
        else:
            # Fallback: use nWBV as global atrophy proxy
            nwbv_vals = grp['nWBV'].fillna(0.75).values
            node_feat = np.zeros((K, N, 1))
            for ki in range(K):
                node_feat[ki, :, 0] = nwbv_vals[ki]

        # Static covariates
        row0 = grp.iloc[0]
        age = float(row0.get('Age', 70)) / 100.0  # normalize
        sex = 1.0 if str(row0.get('M/F', 'M')).strip().upper() == 'M' else 0.0
        educ = float(row0.get('EDUC', 12)) / 20.0
        ses = float(row0.get('SES', 3)) / 5.0 if not pd.isna(row0.get('SES')) else 0.5
        etiv = float(row0.get('eTIV', 1500)) / 2000.0
        asf = float(row0.get('ASF', 1.0))
        covariates = np.array([age, sex, educ, ses, etiv, asf], dtype=float)

        # Targets
        mmse_vals = grp['MMSE'].fillna(25).values.astype(float) / 30.0  # normalize to [0,1]
        cdr_vals = grp['CDR'].fillna(0).values.astype(float)
        cdr_labels = np.array([LongitudinalGraphDataset.CDR_MAP.get(c, 0) for c in cdr_vals])

        subjects_data.append({
            'subject_id': sid,
            'node_features': torch.tensor(node_feat, dtype=torch.float32),
            'times': torch.tensor(t_years, dtype=torch.float32),
            'covariates': torch.tensor(covariates, dtype=torch.float32),
            'mmse': torch.tensor(mmse_vals, dtype=torch.float32),
            'cdr': torch.tensor(cdr_labels, dtype=torch.long),
        })

    if not subjects_data:
        logger.warning("No subjects with >= 2 visits found.")
        return None, None, None

    # Subject-level split 70/15/15
    n = len(subjects_data)
    perm = rng.permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_data = [subjects_data[i] for i in perm[:n_train]]
    val_data = [subjects_data[i] for i in perm[n_train:n_train + n_val]]
    test_data = [subjects_data[i] for i in perm[n_train + n_val:]]

    logger.info(f"Dataset split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    train_loader = DataLoader(
        LongitudinalGraphDataset(train_data, adj_norm),
        batch_size=8, shuffle=True, collate_fn=collate_longitudinal, drop_last=False,
    )
    val_loader = DataLoader(
        LongitudinalGraphDataset(val_data, adj_norm),
        batch_size=8, shuffle=False, collate_fn=collate_longitudinal, drop_last=False,
    )
    test_loader = DataLoader(
        LongitudinalGraphDataset(test_data, adj_norm),
        batch_size=8, shuffle=False, collate_fn=collate_longitudinal, drop_last=False,
    )

    return train_loader, val_loader, test_loader


# ===========================================================================
# MODEL 1: Continuous-Time Spatio-Temporal Graph Neural ODE (ST-GNN-ODE)
# ===========================================================================

class SinusoidalTimeEncoding(nn.Module):
    """Sinusoidal positional encoding for continuous time values."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) or scalar
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class GraphConvLayer(nn.Module):
    """Simple graph convolution: H' = σ(A_hat @ H @ W)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, adj):
        # x: (B, N, D), adj: (1, N, N) or (N, N)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        out = torch.bmm(adj.expand(x.size(0), -1, -1), x)
        return F.elu(self.W(out))


class GATLayer(nn.Module):
    """Graph Attention layer (single-head for speed)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        # x: (B, N, D), adj: (1, N, N)
        B, N, _ = x.shape
        h = self.W(x)  # (B, N, out)
        # Pairwise attention
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, out)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, out)
        e = self.leaky(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))  # (B, N, N)
        # Mask by adjacency
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        mask = adj.expand(B, -1, -1)
        e = e.masked_fill(mask < 1e-6, float('-inf'))
        alpha = F.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        out = torch.bmm(alpha, h)
        return F.elu(out)


class ODEFunc(nn.Module):
    """Neural vector field f_θ(h(t), A, t) for the Graph Neural ODE."""
    def __init__(self, hidden_dim, n_cov, time_dim=16):
        super().__init__()
        self.gat = GATLayer(hidden_dim, hidden_dim)
        self.time_enc = SinusoidalTimeEncoding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + n_cov + time_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.adj = None
        self.cov = None

    def set_context(self, adj, cov):
        self.adj = adj
        self.cov = cov

    def forward(self, t, h):
        # h: (B, N, D)
        B, N, D = h.shape
        # Graph attention
        g_out = self.gat(h, self.adj)

        # Time encoding
        t_enc = self.time_enc(t.expand(B))  # (B, time_dim)
        t_enc = t_enc.unsqueeze(1).expand(-1, N, -1)  # (B, N, time_dim)

        # Covariate
        cov_exp = self.cov.unsqueeze(1).expand(-1, N, -1)  # (B, N, C)

        # Concatenate and MLP
        combined = torch.cat([g_out, cov_exp, t_enc], dim=-1)
        return self.mlp(combined)


class STGraphNeuralODE(nn.Module):
    """
    Model 1: Continuous-Time Spatio-Temporal Graph Neural ODE.

    Encodes baseline state, integrates ODE forward to target times,
    decodes to predict future atrophy and clinical endpoints.
    """
    def __init__(self, n_node_feat=1, hidden_dim=32, n_cov=6, n_cdr_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_node_feat + n_cov, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ode_func = ODEFunc(hidden_dim, n_cov, time_dim=16)

        # Decoders
        self.decoder_recon = nn.Linear(hidden_dim, n_node_feat)
        self.decoder_mmse = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ELU(), nn.Linear(16, 1),
        )
        self.decoder_cdr = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ELU(), nn.Linear(16, n_cdr_classes),
        )

    def forward(self, node_features, times, covariates, adj, mask):
        """
        Args:
            node_features: (B, K, N, D)
            times: (B, K) — years from baseline
            covariates: (B, C)
            adj: (1, N, N)
            mask: (B, K)
        Returns:
            recon: (B, K, N, D), mmse_pred: (B, K), cdr_logits: (B, K, 4)
        """
        B, K, N, D = node_features.shape

        # Encode baseline (t=0)
        cov_exp = covariates.unsqueeze(1).expand(-1, N, -1)  # (B, N, C)
        x0 = torch.cat([node_features[:, 0], cov_exp], dim=-1)  # (B, N, D+C)
        h0 = self.encoder(x0)  # (B, N, hidden)

        # Set ODE context
        self.ode_func.set_context(adj, covariates)

        # Integrate ODE to each time point using Euler steps (fast)
        recon_list = []
        mmse_list = []
        cdr_list = []

        h = h0
        prev_t = torch.zeros(1, device=h.device)

        for ki in range(K):
            t_target = times[:, ki].mean()  # batch-averaged time
            dt = t_target - prev_t
            if dt > 0.001:
                # Euler integration with 5 steps
                n_steps = 5
                step = dt / n_steps
                for _ in range(n_steps):
                    h = h + step * self.ode_func(prev_t, h)
                    prev_t = prev_t + step
            prev_t = t_target

            # Decode
            recon_list.append(self.decoder_recon(h))
            # Global pool for clinical prediction
            h_global = h.mean(dim=1)  # (B, hidden)
            mmse_list.append(self.decoder_mmse(h_global).squeeze(-1))
            cdr_list.append(self.decoder_cdr(h_global))

        recon = torch.stack(recon_list, dim=1)  # (B, K, N, D)
        mmse_pred = torch.stack(mmse_list, dim=1)  # (B, K)
        cdr_logits = torch.stack(cdr_list, dim=1)  # (B, K, 4)

        return recon, mmse_pred, cdr_logits


def train_st_gnn_ode(train_loader, val_loader, n_epochs=50, lr=1e-3, patience=10):
    """Train ST-GNN-ODE model with early stopping."""
    # Get dimensions from first batch
    batch0 = next(iter(train_loader))
    n_feat = batch0['node_features'].shape[-1]
    n_cov = batch0['covariates'].shape[-1]

    model = STGraphNeuralODE(n_node_feat=n_feat, hidden_dim=32, n_cov=n_cov).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            nf = batch['node_features'].to(DEVICE)
            t = batch['times'].to(DEVICE)
            cov = batch['covariates'].to(DEVICE)
            adj = batch['adj'].to(DEVICE)
            msk = batch['mask'].to(DEVICE)
            mmse_true = batch['mmse'].to(DEVICE)
            cdr_true = batch['cdr'].to(DEVICE)

            recon, mmse_pred, cdr_logits = model(nf, t, cov, adj, msk)

            # Reconstruction loss (MSE on future visits)
            loss_recon = F.mse_loss(recon[msk], nf[msk])

            # MMSE loss
            loss_mmse = F.mse_loss(mmse_pred[msk], mmse_true[msk])

            # CDR loss (cross-entropy, skip masked)
            valid_cdr = msk & (cdr_true >= 0)
            if valid_cdr.any():
                loss_cdr = F.cross_entropy(
                    cdr_logits[valid_cdr], cdr_true[valid_cdr]
                )
            else:
                loss_cdr = torch.tensor(0.0, device=DEVICE)

            loss = loss_recon + 0.5 * loss_mmse + 0.3 * loss_cdr

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                nf = batch['node_features'].to(DEVICE)
                t = batch['times'].to(DEVICE)
                cov = batch['covariates'].to(DEVICE)
                adj = batch['adj'].to(DEVICE)
                msk = batch['mask'].to(DEVICE)
                mmse_true = batch['mmse'].to(DEVICE)
                cdr_true = batch['cdr'].to(DEVICE)

                recon, mmse_pred, cdr_logits = model(nf, t, cov, adj, msk)
                loss_recon = F.mse_loss(recon[msk], nf[msk])
                loss_mmse = F.mse_loss(mmse_pred[msk], mmse_true[msk])
                valid_cdr = msk & (cdr_true >= 0)
                if valid_cdr.any():
                    loss_cdr = F.cross_entropy(cdr_logits[valid_cdr], cdr_true[valid_cdr])
                else:
                    loss_cdr = torch.tensor(0.0, device=DEVICE)
                val_loss += (loss_recon + 0.5 * loss_mmse + 0.3 * loss_cdr).item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)

        if epoch % 10 == 0:
            logger.info(f"[ST-GNN-ODE] Epoch {epoch}: train={avg_train:.4f}, val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"[ST-GNN-ODE] Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    logger.info(f"[ST-GNN-ODE] Training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ===========================================================================
# MODEL 2: Physics-Constrained Network Diffusion VAE (ND-VAE)
# ===========================================================================

class PhysicsConstrainedNDVAE(nn.Module):
    """
    Model 2: ND-VAE with biophysical regularization.

    Encoder maps longitudinal trajectories to latent z.
    Decoder generates trajectory conditioned on (z, t, c).
    Physics loss penalizes deviation from network diffusion dynamics.
    """
    def __init__(self, n_regions, n_node_feat=1, n_cov=6, latent_dim=16, hidden_dim=32):
        super().__init__()
        self.n_regions = n_regions
        self.latent_dim = latent_dim

        # Encoder: GRU over visits → latent
        self.enc_proj = nn.Linear(n_regions * n_node_feat + n_cov, hidden_dim)
        self.enc_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: (z, t, c) → X(t)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + 1 + n_cov, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_regions * n_node_feat),
        )

        # Clinical decoders
        self.dec_mmse = nn.Sequential(
            nn.Linear(latent_dim + 1 + n_cov, 16), nn.ELU(), nn.Linear(16, 1),
        )
        self.dec_cdr = nn.Sequential(
            nn.Linear(latent_dim + 1 + n_cov, 16), nn.ELU(), nn.Linear(16, 4),
        )

        # Learnable diffusion rate
        self.beta_diff = nn.Parameter(torch.tensor(0.1))

    def encode(self, node_features, covariates, mask):
        """Encode longitudinal trajectory to latent distribution."""
        B, K, N, D = node_features.shape
        # Flatten node features per visit
        x_flat = node_features.view(B, K, N * D)  # (B, K, N*D)
        cov_exp = covariates.unsqueeze(1).expand(-1, K, -1)
        inp = torch.cat([x_flat, cov_exp], dim=-1)
        inp = F.elu(self.enc_proj(inp))

        # Pack for GRU (handle variable lengths via mask)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            inp, lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.enc_gru(packed)
        h = h.squeeze(0)  # (B, hidden)

        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, t, covariates):
        """
        Decode latent z at time t with covariates.
        t: (B,) scalar time
        """
        inp = torch.cat([z, t.unsqueeze(-1), covariates], dim=-1)
        x_hat = self.dec(inp).view(-1, self.n_regions, 1)  # (B, N, D)
        mmse_hat = self.dec_mmse(inp).squeeze(-1)  # (B,)
        cdr_logits = self.dec_cdr(inp)  # (B, 4)
        return x_hat, mmse_hat, cdr_logits

    def forward(self, node_features, times, covariates, adj, mask):
        B, K, N, D = node_features.shape

        mu, logvar = self.encode(node_features, covariates, mask)
        z = self.reparameterize(mu, logvar)

        # Decode at each time point
        recon_list, mmse_list, cdr_list = [], [], []
        for ki in range(K):
            t_k = times[:, ki]
            x_hat, m_hat, c_hat = self.decode(z, t_k, covariates)
            recon_list.append(x_hat)
            mmse_list.append(m_hat)
            cdr_list.append(c_hat)

        recon = torch.stack(recon_list, dim=1)  # (B, K, N, D)
        mmse_pred = torch.stack(mmse_list, dim=1)  # (B, K)
        cdr_logits = torch.stack(cdr_list, dim=1)  # (B, K, 4)

        return recon, mmse_pred, cdr_logits, mu, logvar

    def physics_loss(self, recon, times, adj, mask):
        """
        Biophysical regularizer: ||dX/dt + β * L_hat * X||²_F
        Approximated via finite differences.
        """
        B, K, N, D = recon.shape
        if K < 2:
            return torch.tensor(0.0, device=recon.device)

        # Compute Laplacian L = I - A_hat (adj is already normalized)
        if adj.dim() == 3:
            adj_2d = adj[0]
        else:
            adj_2d = adj
        L = torch.eye(N, device=recon.device) - adj_2d

        total = torch.tensor(0.0, device=recon.device)
        count = 0
        for ki in range(K - 1):
            dt = (times[:, ki + 1] - times[:, ki]).clamp(min=0.01)  # (B,)
            dX = recon[:, ki + 1] - recon[:, ki]  # (B, N, D)
            dXdt = dX / dt.unsqueeze(-1).unsqueeze(-1)

            # L * X
            LX = torch.matmul(L.unsqueeze(0), recon[:, ki])  # (B, N, D)

            # Physics residual
            residual = dXdt + self.beta_diff * LX
            both_valid = mask[:, ki] & mask[:, ki + 1]
            if both_valid.any():
                total = total + residual[both_valid].pow(2).mean()
                count += 1

        return total / max(count, 1)


def train_nd_vae(train_loader, val_loader, n_regions, n_epochs=100, lr=1e-3, patience=10):
    """Train ND-VAE with β-annealing and physics loss."""
    batch0 = next(iter(train_loader))
    n_feat = batch0['node_features'].shape[-1]
    n_cov = batch0['covariates'].shape[-1]

    model = PhysicsConstrainedNDVAE(
        n_regions=n_regions, n_node_feat=n_feat, n_cov=n_cov,
        latent_dim=16, hidden_dim=32,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        # β-annealing: linearly increase KL weight over first 20 epochs
        beta_kl = min(1.0, epoch / 20.0) * 0.1
        lambda_phys = 0.01

        train_loss = 0.0
        for batch in train_loader:
            nf = batch['node_features'].to(DEVICE)
            t = batch['times'].to(DEVICE)
            cov = batch['covariates'].to(DEVICE)
            adj = batch['adj'].to(DEVICE)
            msk = batch['mask'].to(DEVICE)
            mmse_true = batch['mmse'].to(DEVICE)
            cdr_true = batch['cdr'].to(DEVICE)

            recon, mmse_pred, cdr_logits, mu, logvar = model(nf, t, cov, adj, msk)

            # Reconstruction L1
            loss_recon = F.l1_loss(recon[msk], nf[msk])

            # KL divergence
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Physics loss
            loss_phys = model.physics_loss(recon, t, adj, msk)

            # MMSE + CDR
            loss_mmse = F.mse_loss(mmse_pred[msk], mmse_true[msk])
            valid_cdr = msk & (cdr_true >= 0)
            if valid_cdr.any():
                loss_cdr = F.cross_entropy(cdr_logits[valid_cdr], cdr_true[valid_cdr])
            else:
                loss_cdr = torch.tensor(0.0, device=DEVICE)

            loss = loss_recon + beta_kl * kl + lambda_phys * loss_phys + 0.5 * loss_mmse + 0.3 * loss_cdr

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                nf = batch['node_features'].to(DEVICE)
                t = batch['times'].to(DEVICE)
                cov = batch['covariates'].to(DEVICE)
                adj = batch['adj'].to(DEVICE)
                msk = batch['mask'].to(DEVICE)
                mmse_true = batch['mmse'].to(DEVICE)
                cdr_true = batch['cdr'].to(DEVICE)

                recon, mmse_pred, cdr_logits, mu, logvar = model(nf, t, cov, adj, msk)
                loss_recon = F.l1_loss(recon[msk], nf[msk])
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss_mmse = F.mse_loss(mmse_pred[msk], mmse_true[msk])
                valid_cdr = msk & (cdr_true >= 0)
                if valid_cdr.any():
                    loss_cdr = F.cross_entropy(cdr_logits[valid_cdr], cdr_true[valid_cdr])
                else:
                    loss_cdr = torch.tensor(0.0, device=DEVICE)
                val_loss += (loss_recon + 0.1 * kl + 0.5 * loss_mmse + 0.3 * loss_cdr).item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)

        if epoch % 10 == 0:
            logger.info(f"[ND-VAE] Epoch {epoch}: train={avg_train:.4f}, val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"[ND-VAE] Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    logger.info(f"[ND-VAE] Training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ===========================================================================
# MODEL 3: Temporally-Aware Trajectory Diffusion Model (TADM)
# ===========================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule for DDPM."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.999)


class DiffusionConditioner(nn.Module):
    """Conditioning module: baseline GCN + time encoding + covariate MLP."""
    def __init__(self, n_regions, n_node_feat, n_cov, cond_dim=32):
        super().__init__()
        # Baseline encoder via graph conv
        self.gcn = GraphConvLayer(n_node_feat, cond_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEncoding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )
        # Covariate MLP
        self.cov_mlp = nn.Sequential(
            nn.Linear(n_cov, cond_dim),
            nn.GELU(),
        )
        self.out_proj = nn.Linear(3 * cond_dim, cond_dim)

    def forward(self, x0, delta_t, covariates, adj):
        """
        x0: (B, N, D) baseline node features
        delta_t: (B,) elapsed time
        covariates: (B, C)
        adj: (1, N, N)
        """
        # Graph encoding of baseline
        g = self.gcn(x0, adj)  # (B, N, cond_dim)
        g = g.permute(0, 2, 1)  # (B, cond_dim, N)
        g = self.pool(g).squeeze(-1)  # (B, cond_dim)

        # Time encoding
        t_emb = self.time_mlp[0](delta_t)  # sinusoidal
        t_emb = self.time_mlp[1](t_emb)
        t_emb = self.time_mlp[2](t_emb)

        # Covariate encoding
        c_emb = self.cov_mlp(covariates)

        # Combine
        combined = torch.cat([g, t_emb, c_emb], dim=-1)
        return self.out_proj(combined)  # (B, cond_dim)


class DenoisingNet(nn.Module):
    """Simple MLP denoiser with FiLM conditioning."""
    def __init__(self, data_dim, cond_dim=32, hidden_dim=64):
        super().__init__()
        # Diffusion step embedding
        self.step_emb = nn.Embedding(200, cond_dim)

        # FiLM conditioning
        self.film_gamma = nn.Linear(2 * cond_dim, hidden_dim)
        self.film_beta = nn.Linear(2 * cond_dim, hidden_dim)

        # Denoiser backbone
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x_noisy, diffusion_step, cond):
        """
        x_noisy: (B, data_dim) — noised target state
        diffusion_step: (B,) long — diffusion timestep index
        cond: (B, cond_dim) — conditioning vector
        """
        step_emb = self.step_emb(diffusion_step)  # (B, cond_dim)
        cond_full = torch.cat([cond, step_emb], dim=-1)  # (B, 2*cond_dim)

        h = self.net(x_noisy)  # (B, hidden)

        # FiLM modulation
        gamma = self.film_gamma(cond_full)
        beta = self.film_beta(cond_full)
        h = gamma * h + beta

        return self.out(h)  # (B, data_dim) — predicted noise


class TemporalDiffusionModel(nn.Module):
    """
    Model 3: TADM — Temporally-Aware Trajectory Diffusion Model.

    DDPM conditioned on baseline state, elapsed time, and covariates.
    Generates future neurodegeneration states.
    """
    def __init__(self, n_regions, n_node_feat=1, n_cov=6, T=200, cond_dim=32):
        super().__init__()
        self.n_regions = n_regions
        self.data_dim = n_regions * n_node_feat
        self.T = T

        # Noise schedule
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        self.conditioner = DiffusionConditioner(n_regions, n_node_feat, n_cov, cond_dim)
        self.denoiser = DenoisingNet(self.data_dim, cond_dim, hidden_dim=64)

        # Clinical prediction heads (from denoised output)
        self.mmse_head = nn.Sequential(
            nn.Linear(self.data_dim, 16), nn.GELU(), nn.Linear(16, 1),
        )
        self.cdr_head = nn.Sequential(
            nn.Linear(self.data_dim, 16), nn.GELU(), nn.Linear(16, 4),
        )

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise at step t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def training_loss(self, node_features, times, covariates, adj, mask):
        """
        Compute DDPM training loss over all valid (baseline → future) pairs.
        """
        B, K, N, D = node_features.shape
        losses = []

        for ki in range(1, K):
            # Check which samples have valid data at this visit
            valid = mask[:, ki] & mask[:, 0]
            if not valid.any():
                continue

            x0_base = node_features[valid, 0]  # (B', N, D) baseline
            x_target = node_features[valid, ki].reshape(-1, self.data_dim)  # (B', data_dim) future
            dt = times[valid, ki] - times[valid, 0]  # (B',) elapsed time
            cov = covariates[valid]

            # Conditioning
            cond = self.conditioner(x0_base, dt, cov, adj)  # (B', cond_dim)

            # Random diffusion timestep
            B_valid = x_target.shape[0]
            t_diff = torch.randint(0, self.T, (B_valid,), device=x_target.device)

            # Forward diffusion
            x_noisy, noise = self.q_sample(x_target, t_diff)

            # Predict noise
            noise_pred = self.denoiser(x_noisy, t_diff, cond)

            losses.append(F.mse_loss(noise_pred, noise))

        if not losses:
            return torch.tensor(0.0, device=node_features.device)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample(self, x0_base, delta_t, covariates, adj, n_steps=10):
        """
        DDIM-style fast sampling from noise → predicted future state.
        """
        B = x0_base.shape[0]
        cond = self.conditioner(x0_base, delta_t, covariates, adj)

        # Start from pure noise
        x = torch.randn(B, self.data_dim, device=x0_base.device)

        # DDIM sampling with n_steps
        step_indices = torch.linspace(self.T - 1, 0, n_steps, dtype=torch.long, device=x.device)

        for i, t_idx in enumerate(step_indices):
            t_batch = t_idx.expand(B).long()
            noise_pred = self.denoiser(x, t_batch, cond)

            alpha_t = self.alphas_cumprod[t_idx]
            if i < len(step_indices) - 1:
                alpha_prev = self.alphas_cumprod[step_indices[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=x.device)

            # DDIM update
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * noise_pred

        return x.view(B, self.n_regions, -1)  # (B, N, D)


def train_tadm(train_loader, val_loader, n_regions, n_epochs=200, lr=1e-3, patience=15):
    """Train TADM diffusion model."""
    batch0 = next(iter(train_loader))
    n_feat = batch0['node_features'].shape[-1]
    n_cov = batch0['covariates'].shape[-1]

    model = TemporalDiffusionModel(
        n_regions=n_regions, n_node_feat=n_feat, n_cov=n_cov, T=200, cond_dim=32,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            nf = batch['node_features'].to(DEVICE)
            t = batch['times'].to(DEVICE)
            cov = batch['covariates'].to(DEVICE)
            adj = batch['adj'].to(DEVICE)
            msk = batch['mask'].to(DEVICE)

            loss = model.training_loss(nf, t, cov, adj, msk)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                nf = batch['node_features'].to(DEVICE)
                t = batch['times'].to(DEVICE)
                cov = batch['covariates'].to(DEVICE)
                adj = batch['adj'].to(DEVICE)
                msk = batch['mask'].to(DEVICE)
                val_loss += model.training_loss(nf, t, cov, adj, msk).item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)

        if epoch % 20 == 0:
            logger.info(f"[TADM] Epoch {epoch}: train={avg_train:.4f}, val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"[TADM] Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    logger.info(f"[TADM] Training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ===========================================================================
# BENCHMARK RUNNER — Evaluate all models on test set
# ===========================================================================

def evaluate_dl_model(model, test_loader, model_name, n_regions=None):
    """Evaluate a DL model on test data. Returns metrics dict."""
    model.eval()
    all_recon, all_true = [], []
    all_mmse_pred, all_mmse_true = [], []
    all_cdr_pred, all_cdr_true = [], []

    with torch.no_grad():
        for batch in test_loader:
            nf = batch['node_features'].to(DEVICE)
            t = batch['times'].to(DEVICE)
            cov = batch['covariates'].to(DEVICE)
            adj = batch['adj'].to(DEVICE)
            msk = batch['mask'].to(DEVICE)
            mmse_true = batch['mmse'].to(DEVICE)
            cdr_true = batch['cdr'].to(DEVICE)

            if model_name == 'TADM':
                # For TADM, generate predictions via sampling
                B, K, N, D = nf.shape
                for ki in range(1, K):
                    valid = msk[:, ki] & msk[:, 0]
                    if not valid.any():
                        continue
                    x0 = nf[valid, 0]
                    dt = t[valid, ki] - t[valid, 0]
                    pred = model.sample(x0, dt, cov[valid], adj, n_steps=10)
                    all_recon.append(pred.cpu())
                    all_true.append(nf[valid, ki].cpu())
                    # Clinical predictions from sampled state
                    pred_flat = pred.view(-1, model.data_dim)
                    all_mmse_pred.append(model.mmse_head(pred_flat).squeeze(-1).cpu())
                    all_mmse_true.append(mmse_true[valid, ki].cpu())
                    all_cdr_pred.append(model.cdr_head(pred_flat).cpu())
                    all_cdr_true.append(cdr_true[valid, ki].cpu())
            elif model_name == 'ND-VAE':
                recon, mmse_pred, cdr_logits, _, _ = model(nf, t, cov, adj, msk)
                # Use future visits only (skip baseline)
                for ki in range(1, nf.shape[1]):
                    valid = msk[:, ki]
                    if valid.any():
                        all_recon.append(recon[valid, ki].cpu())
                        all_true.append(nf[valid, ki].cpu())
                        all_mmse_pred.append(mmse_pred[valid, ki].cpu())
                        all_mmse_true.append(mmse_true[valid, ki].cpu())
                        all_cdr_pred.append(cdr_logits[valid, ki].cpu())
                        all_cdr_true.append(cdr_true[valid, ki].cpu())
            else:  # ST-GNN-ODE
                recon, mmse_pred, cdr_logits = model(nf, t, cov, adj, msk)
                for ki in range(1, nf.shape[1]):
                    valid = msk[:, ki]
                    if valid.any():
                        all_recon.append(recon[valid, ki].cpu())
                        all_true.append(nf[valid, ki].cpu())
                        all_mmse_pred.append(mmse_pred[valid, ki].cpu())
                        all_mmse_true.append(mmse_true[valid, ki].cpu())
                        all_cdr_pred.append(cdr_logits[valid, ki].cpu())
                        all_cdr_true.append(cdr_true[valid, ki].cpu())

    metrics = {}
    if all_recon:
        recon_cat = torch.cat(all_recon)
        true_cat = torch.cat(all_true)
        rmse = torch.sqrt(F.mse_loss(recon_cat, true_cat)).item()
        mae = F.l1_loss(recon_cat, true_cat).item()

        # Pearson r (flatten)
        r_flat = recon_cat.flatten().numpy()
        t_flat = true_cat.flatten().numpy()
        if len(r_flat) > 2 and np.std(r_flat) > 1e-10 and np.std(t_flat) > 1e-10:
            pearson_r = float(np.corrcoef(r_flat, t_flat)[0, 1])
        else:
            pearson_r = 0.0
        metrics['atrophy_rmse'] = round(rmse, 4)
        metrics['atrophy_mae'] = round(mae, 4)
        metrics['atrophy_pearson_r'] = round(pearson_r, 4)

    if all_mmse_pred:
        mmse_p = torch.cat(all_mmse_pred)
        mmse_t = torch.cat(all_mmse_true)
        metrics['mmse_mae'] = round(F.l1_loss(mmse_p, mmse_t).item(), 4)

    if all_cdr_pred and all_cdr_true:
        cdr_p = torch.cat(all_cdr_pred)
        cdr_t = torch.cat(all_cdr_true)
        valid_cdr = cdr_t >= 0
        if valid_cdr.any():
            cdr_p_valid = cdr_p[valid_cdr]
            cdr_t_valid = cdr_t[valid_cdr]
            pred_labels = cdr_p_valid.argmax(dim=-1)
            acc = (pred_labels == cdr_t_valid).float().mean().item()
            metrics['cdr_accuracy'] = round(acc, 4)
            # Macro F1
            from sklearn.metrics import f1_score as sk_f1
            try:
                metrics['cdr_f1_macro'] = round(
                    sk_f1(cdr_t_valid.numpy(), pred_labels.numpy(), average='macro', zero_division=0), 4
                )
            except Exception:
                metrics['cdr_f1_macro'] = 0.0

    return metrics


def run_benchmark(
    ndm_results: Dict,
    st_gnn_ode_model,
    nd_vae_model,
    tadm_model,
    test_loader,
    n_regions: int,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Evaluate all 4 models on the same test set and produce comparison table.
    """
    results = {}

    # NDM baseline metrics
    ndm_val = ndm_results.get('validation', {})
    results['NDM (Baseline)'] = {
        'atrophy_rmse': ndm_val.get('rmse', '-'),
        'atrophy_mae': ndm_val.get('mae', '-'),
        'atrophy_pearson_r': ndm_val.get('pearson_r', '-'),
        'mmse_mae': '-',
        'cdr_accuracy': '-',
        'cdr_f1_macro': '-',
    }

    # DL models
    if st_gnn_ode_model is not None:
        results['ST-GNN-ODE'] = evaluate_dl_model(st_gnn_ode_model, test_loader, 'ST-GNN-ODE')
    if nd_vae_model is not None:
        results['ND-VAE'] = evaluate_dl_model(nd_vae_model, test_loader, 'ND-VAE')
    if tadm_model is not None:
        results['TADM'] = evaluate_dl_model(tadm_model, test_loader, 'TADM', n_regions)

    df = pd.DataFrame(results).T
    df.index.name = 'Model'

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'benchmark_results.csv')
    logger.info(f"Benchmark results saved to {out_dir / 'benchmark_results.csv'}")
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS — All Models vs NDM Baseline")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)

    return df


# ---------------------------------------------------------------------------
# Feature engineering (kept for compatibility)
# ---------------------------------------------------------------------------

CLINICAL_FEATURES = ["Age", "EDUC", "SES", "eTIV", "nWBV"]
SEX_COL = "M/F"  # encode as 0/1



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

    cols_to_copy = ["subject_id"]
    if "session_id" in regional_volumes.columns:
        cols_to_copy.append("session_id")
    w_df = regional_volumes[cols_to_copy].copy()
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
        join_cols = ["subject_id"]
        if "session_id" in w_scores.columns and "session_id" in demo_df.columns:
            join_cols.append("session_id")
        merged = w_scores.merge(
            demo_df[join_cols + [time_col]],
            on=join_cols, how="left",
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
        join_cols = ["subject_id"]
        if "session_id" in w_scores.columns and "session_id" in demo_df.columns:
            join_cols.append("session_id")
        merged = w_scores.merge(
            demo_df[join_cols + [time_col]],
            on=join_cols, how="left",
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
    print(f"\n[GPU] {gpu_name} detected ({gpu_mem_gb:.1f} GB). DL model training will use the GPU.")
else:
    print(
        "\n[WARNING] No GPU detected — PyTorch will fall back to CPU, which is "
        "much slower for DL model training.\n"
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
        print('        you have accepted the dataset terms at:')
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
                color=plt.get_cmap('RdYlGn_r')(np.linspace(0.1, 0.9, len(mean_d))))
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



# ===========================================================================
# DEEP LEARNING MODEL TRAINING
# ===========================================================================

print('\n' + '=' * 70)
print('TRAINING DEEP LEARNING MODELS')
print('=' * 70)

# Load connectome
from pathlib import Path as _P
connectome_path = _P(str(processed_dir)) / 'template_connectome.npy'
adj_matrix, region_names_loaded = load_template_connectome(connectome_path if connectome_path.exists() else None)
n_regions_dl = adj_matrix.shape[0]

# Build DL datasets from demographics + w-scores
train_loader, val_loader, test_loader = build_dl_datasets(
    demo_df=demo_df,
    w_scores=w_scores,
    adj_matrix=adj_matrix,
    region_names=CONNECTOME_REGIONS,
    seed=42,
)

st_gnn_ode_model = None
nd_vae_model = None
tadm_model = None

if train_loader is not None:
    print(f'\nDevice: {DEVICE}')
    print(f'Regions: {n_regions_dl}')

    # --- Model 1: ST-GNN-ODE ---
    print('\n--- Training Model 1: ST-GNN-ODE ---')
    try:
        st_gnn_ode_model = train_st_gnn_ode(train_loader, val_loader, n_epochs=50, lr=1e-3, patience=10)
        print('[OK] ST-GNN-ODE training complete.')
    except Exception as e:
        print(f'[ERROR] ST-GNN-ODE training failed: {e}')
        import traceback; traceback.print_exc()

    # --- Model 2: ND-VAE ---
    print('\n--- Training Model 2: ND-VAE ---')
    try:
        nd_vae_model = train_nd_vae(train_loader, val_loader, n_regions_dl, n_epochs=100, lr=1e-3, patience=10)
        print('[OK] ND-VAE training complete.')
    except Exception as e:
        print(f'[ERROR] ND-VAE training failed: {e}')
        import traceback; traceback.print_exc()

    # --- Model 3: TADM ---
    print('\n--- Training Model 3: TADM ---')
    try:
        tadm_model = train_tadm(train_loader, val_loader, n_regions_dl, n_epochs=200, lr=1e-3, patience=15)
        print('[OK] TADM training complete.')
    except Exception as e:
        print(f'[ERROR] TADM training failed: {e}')
        import traceback; traceback.print_exc()

    # --- Benchmark ---
    print('\n--- Running Benchmark ---')
    (DRIVE_BASE / 'outputs' / 'benchmark').mkdir(parents=True, exist_ok=True)
    benchmark_df = run_benchmark(
        ndm_results=p3_results,
        st_gnn_ode_model=st_gnn_ode_model,
        nd_vae_model=nd_vae_model,
        tadm_model=tadm_model,
        test_loader=test_loader,
        n_regions=n_regions_dl,
        out_dir=outputs_dir / 'benchmark',
    )
    benchmark_df.to_csv(DRIVE_BASE / 'outputs' / 'benchmark' / 'benchmark_results.csv')
    print('[OK] Benchmark complete.')

    # Save models to Drive
    models_dir = DRIVE_BASE / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    if st_gnn_ode_model is not None:
        torch.save(st_gnn_ode_model.state_dict(), models_dir / 'st_gnn_ode.pt')
    if nd_vae_model is not None:
        torch.save(nd_vae_model.state_dict(), models_dir / 'nd_vae.pt')
    if tadm_model is not None:
        torch.save(tadm_model.state_dict(), models_dir / 'tadm.pt')
    print(f'[OK] Models saved to {models_dir}')
else:
    print('[WARNING] Could not build DL datasets — skipping model training.')
    benchmark_df = pd.DataFrame()

# Per-subject report
SUBJECT_ID = None
if SUBJECT_ID is None and not beta_df.empty:
    SUBJECT_ID = beta_df['subject_id'].dropna().iloc[0]
if SUBJECT_ID:
    print(f'\n=== Per-Subject Report: {SUBJECT_ID} ===')
    sd = demo_df[demo_df['subject_id']==SUBJECT_ID]
    if not sd.empty:
        r = sd.iloc[0]
        print(f'  Age={r.get("Age","?")}  Group={r.get("Group","?")}  EDUC={r.get("EDUC","?")}')
    sb = beta_df[beta_df['subject_id']==SUBJECT_ID]['beta'].dropna()
    if len(sb): print(f'  NDM beta={sb.iloc[0]:.4f}')
    if not deficit_df.empty and SUBJECT_ID in deficit_df.index:
        print('  Top deficits:', dict(deficit_df.loc[SUBJECT_ID].sort_values(ascending=False).head(3).round(3)))

print(f'\nAll outputs saved to Drive: {DRIVE_BASE / "outputs"}')
print('Pipeline complete.')

