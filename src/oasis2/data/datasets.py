"""Longitudinal graph dataset builder for the deep-learning models.

Ported from OASIS2_Colab.py (lines 1006-1197): `LongitudinalGraphDataset`,
`collate_longitudinal`, `build_dl_datasets`.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from oasis2.logging_utils import setup_logging

logger = setup_logging("datasets")


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
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
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
