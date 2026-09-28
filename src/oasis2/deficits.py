"""Phase 4: cognitive deficit mapping.

Ported from OASIS2_Colab.py (lines 2804-3139): a region -> cognitive-domain
atlas, deficit-score computation from predicted regional atrophy, and
validation against real MMSE/CDR outcomes.

NOTE (bugfix): `get_domain_weights` in the source script only accepts a
required `region_name` argument (single-region lookup). The plan's own test
for this module (`test_get_domain_weights_returns_nonempty_dict`) calls it
with no arguments at all and expects a non-empty dict back. To reconcile
this without changing the single-region lookup behaviour used internally by
`compute_deficit_scores`, `region_name` is given a default of `None`: calling
`get_domain_weights()` with no region returns the whole `REGION_DOMAIN_MAP`
atlas (which is itself a non-empty dict), while `get_domain_weights("...")`
keeps its original per-region behaviour.

NOTE (bugfix): `run_phase4`'s source implementation returns a bare `{}` when
no subject has a non-NaN fitted beta (`if not predicted_rows: return {}`).
With the small synthetic fixtures used in tests (8 regions) run through
`ndm.run_phase3` with its *default* connectome (56 regions, see
`connectome.py` / `ndm.py`), the region-count mismatch guard in
`ndm.fit_beta` makes every subject's fitted beta NaN, so `predicted_rows` is
always empty in that scenario and Phase 4 would report nothing at all -
including callers that only want to know the (empty) `deficit_scores`
shape. This mirrors the same "no data available" case `ndm.run_phase3`
already handles by returning a full, consistently-shaped dict of empty
results rather than `run_phase4` returning a plain `{}`. `run_phase4` here
applies the same pattern: it always returns a dict with `predicted_atrophy`,
`deficit_scores`, and `validation` keys, populated with empty
DataFrames/dict when there is no predictable atrophy.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats

from oasis2.config import get_outputs_dir
from oasis2.connectome import ndm_propagate
from oasis2.logging_utils import setup_logging

logger = setup_logging("deficits")


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


def get_domain_weights(region_name: Optional[str] = None) -> Dict:
    """Return the cognitive domain weights for a brain region.

    If `region_name` is omitted (None), returns the full region -> domain
    weight atlas (`REGION_DOMAIN_MAP`) instead of a single region's weights.
    """
    if region_name is None:
        return REGION_DOMAIN_MAP
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
        return {
            "predicted_atrophy": pd.DataFrame(),
            "deficit_scores": pd.DataFrame(columns=COGNITIVE_DOMAINS),
            "validation": {},
        }

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
