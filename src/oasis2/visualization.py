"""Plotting functions for pipeline outputs.

Wraps the inline plotting code that previously lived directly in the
OASIS2_Colab.py script tail (lines ~3291-3441), which referenced
script-level globals (demo_df, outputs_dir, beta_df, ...) instead of taking
explicit parameters, into standalone, individually-testable functions.
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oasis2.logging_utils import setup_logging

logger = setup_logging("visualization")


def plot_phase0_overview(demo_df: pd.DataFrame, outputs_dir: Path) -> Path:
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('OASIS-2 Dataset Overview', fontsize=14, fontweight='bold')
    if 'Group' in demo_df.columns:
        gc = demo_df.groupby('subject_id')['Group'].last().value_counts()
        axes[0].bar(gc.index, gc.values, color=['#4CAF50', '#F44336', '#FF9800'][:len(gc)])
        axes[0].set_title('Subjects by Group'); axes[0].set_ylabel('Count')
    if 'Age' in demo_df.columns:
        demo_df.groupby('subject_id')['Age'].first().hist(bins=15, ax=axes[1], color='#9C27B0', alpha=0.8)
        axes[1].set_title('Age Distribution'); axes[1].set_xlabel('Age')
    if 'CDR' in demo_df.columns:
        cc = demo_df['CDR'].value_counts().sort_index()
        axes[2].bar(cc.index.astype(str), cc.values, color='#2196F3')
        axes[2].set_title('CDR Score Distribution'); axes[2].set_xlabel('CDR')
    plt.tight_layout()
    out_path = outputs_dir / 'phase0_overview.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Phase 0 overview plot saved to: {out_path}")
    return out_path


def plot_phase3_ndm_results(
    beta_df: pd.DataFrame, w_scores: pd.DataFrame, eigenvalues: np.ndarray,
    eigenvectors: np.ndarray, outputs_dir: Path,
) -> Optional[Path]:
    from oasis2.connectome import ndm_propagate

    if beta_df.empty or 'beta' not in beta_df.columns or beta_df['beta'].dropna().empty:
        logger.warning(
            "beta_df is empty, missing 'beta' column, or has no non-NaN beta values "
            "— skipping Phase 3 plot"
        )
        return None
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase 3: Network Diffusion Model', fontsize=13, fontweight='bold')
    valid_b = beta_df['beta'].dropna()
    axes[0].hist(valid_b, bins=20, color='#9C27B0', alpha=0.85, edgecolor='white')
    axes[0].axvline(valid_b.median(), color='red', linestyle='--', label=f'Median beta={valid_b.median():.3f}')
    axes[0].set_xlabel('Diffusivity beta'); axes[0].set_ylabel('Count')
    axes[0].set_title('Fitted beta per subject'); axes[0].legend()

    top = beta_df.dropna(subset=['beta']).sort_values('beta', ascending=False).iloc[0]
    wscore_cols = [c for c in w_scores.columns if c.endswith('_wscore')]
    ws = w_scores[w_scores['subject_id'] == top['subject_id']]
    if not ws.empty and wscore_cols:
        x0 = ws.iloc[0][wscore_cols].values.astype(float)
        n = min(len(x0), len(eigenvalues))
        x5 = ndm_propagate(x0[:n], 5.0, top['beta'], eigenvalues[:n], eigenvectors[:n, :n])
        axes[1].bar(range(n), x5, color=['#F44336' if v > 0 else '#4CAF50' for v in x5], alpha=0.8)
        axes[1].axhline(0, color='black', linewidth=1)
        axes[1].set_xlabel('Brain Region Index'); axes[1].set_ylabel('Atrophy (W-score)')
        axes[1].set_title(f'5-yr Prediction — {top["subject_id"]} (beta={top["beta"]:.3f})')
    plt.tight_layout()
    out_path = outputs_dir / 'ndm_results.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Phase 3 NDM plot saved to: {out_path}")
    return out_path


def plot_phase4_deficit_profiles(deficit_df: pd.DataFrame, outputs_dir: Path) -> Optional[Path]:
    if deficit_df.empty:
        logger.warning("deficit_df is empty — skipping Phase 4 plot")
        return None
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    mean_d = deficit_df.mean().sort_values(ascending=False)
    axes[0].bar(range(len(mean_d)), mean_d.values, color=plt.get_cmap('RdYlGn_r')(np.linspace(0.1, 0.9, len(mean_d))))
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
    out_path = outputs_dir / 'deficit_profiles.png'
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logger.info(f"Phase 4 deficit profiles plot saved to: {out_path}")
    return out_path
