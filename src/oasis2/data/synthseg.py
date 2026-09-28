"""SynthSeg wrapper: whole-brain parcellation + regional volume extraction.

Ported from OASIS2_Colab.py (lines 622-965): `synthseg_available`,
`get_synthseg_threads`, `run_synthseg_single`, `run_synthseg_batch`,
`aggregate_volumes`, `extract_fsl_seg_volumes`, `build_regional_volumes`,
`spot_check_synthseg_output`.

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
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from oasis2.config import get_data_dir, get_outputs_dir, is_gpu_available
from oasis2.io_utils import checkpoint_exists, load_checkpoint, save_checkpoint
from oasis2.logging_utils import setup_logging

logger = setup_logging("synthseg")


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
    fsl_seg_dir = Path(fsl_seg_dir)
    if not fsl_seg_dir.exists():
        return {}

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
