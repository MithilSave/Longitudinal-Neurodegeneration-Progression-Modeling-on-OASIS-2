"""Network Diffusion Model (NDM) baseline: W-scores, per-subject beta fitting,
and the full Phase 3 pipeline.

Ported from OASIS2_Colab.py (lines 2349-2769).

NOTE (bugfix): `fit_beta` calls `scipy.optimize.minimize_scalar`, but the
original script never imports `scipy.optimize` anywhere (the only `scipy`
import in the whole 3549-line script is `import scipy.stats`, at line 2796,
which does not populate the `scipy.optimize` submodule attribute). This meant
`fit_all_betas` / `run_phase3` would crash with
`AttributeError: module 'scipy' has no attribute 'optimize'` the first time a
subject's beta was fit — the same class of bug as the missing `scipy.linalg`
import fixed in `connectome.py` (Task 3). This module adds explicit
`import scipy.optimize` and `import scipy.stats` to fix that.

NOTE (bugfix): `compute_w_scores` originally merged `regional_volumes` with
`demo_df` on `subject_id` alone (source line ~2376-2379). OASIS-2 demographics
are longitudinal — one row per session/visit per subject — so a subject-only
merge produces a row explosion (each of a subject's session rows in
`regional_volumes` gets duplicated against every one of that subject's rows in
`demo_df`). The subsequent per-column residual array (computed on the
exploded `merged` frame) then has more rows than `w_df` (built from the
original, non-exploded `regional_volumes`), which crashes with
`ValueError: Length of values (...) does not match length of index (...)`.
`fit_all_betas` and `validate_ndm` already join on `subject_id` + `session_id`
for exactly this reason; this module applies the same join-key fix to
`compute_w_scores` for consistency and correctness.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from tqdm.auto import tqdm

from oasis2.config import get_outputs_dir
from oasis2.logging_utils import setup_logging
from oasis2.connectome import (
    load_template_connectome,
    compute_laplacian,
    eigen_decompose_laplacian,
    ndm_propagate,
)

logger = setup_logging("ndm")


# ---------------------------------------------------------------------------
# W-scores
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

    join_cols = ["subject_id"]
    if "session_id" in regional_volumes.columns and "session_id" in demo_df.columns:
        join_cols.append("session_id")
    merge_cols = join_cols + [
        c for c in ["Age", "CDR", "Group"]
        if c in demo_df.columns and c not in regional_volumes.columns
    ]
    merged = regional_volumes.merge(
        demo_df[merge_cols],
        on=join_cols, how="left",
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

    # Guard: the connectome's eigen-decomposition must cover the same number
    # of regions as the observed W-score vector. In production these match by
    # construction (both are keyed to the same brain atlas); a mismatch here
    # means the wrong connectome was supplied for these regional measurements
    # (e.g. a placeholder/synthetic connectome with a different region count),
    # so skip this subject/visit rather than let ndm_propagate raise.
    if eigenvectors.shape[0] != x0.shape[0]:
        logger.warning(
            f"Region count mismatch: connectome has {eigenvectors.shape[0]} regions "
            f"but W-scores have {x0.shape[0]}; skipping beta fit."
        )
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
    validated_subjects: set = set()

    # Guard: beta_df may be empty (no columns) when no W-score regions were found
    if beta_df.empty or "subject_id" not in beta_df.columns:
        logger.warning("beta_df is empty or missing 'subject_id' — skipping NDM validation.")
        return {"n_subjects_validated": 0, "pearson_r": float("nan"), "mse": float("nan")}

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
        validated_subjects.add(subj_id)

    if len(all_predicted) < 5:
        logger.warning("Not enough 3-visit subjects for NDM validation.")
        return {"n_subjects_validated": 0, "pearson_r": np.nan, "mse": np.nan}

    r, p = scipy.stats.pearsonr(all_predicted, all_actual)
    mse = np.mean((np.array(all_predicted) - np.array(all_actual)) ** 2)

    result = {
        "n_subjects_validated": len(validated_subjects),
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

    # Guard: if no _wscore columns were produced (empty regional_volumes), skip gracefully
    wscore_cols = [c for c in w_scores.columns if c.endswith("_wscore")]
    if not wscore_cols:
        logger.warning(
            "No W-score columns in w_scores — regional_volumes has no region data. "
            "Phase 3 NDM cannot run. Returning empty results. "
            "Run Phase 1 (SynthSeg / FSL segmentation) first to generate regional volumes."
        )
        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "region_names": region_names,
            "w_scores": w_scores,
            "beta_df": pd.DataFrame(),
            "validation": {"n_subjects_validated": 0, "pearson_r": float("nan"), "mse": float("nan")},
        }

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
