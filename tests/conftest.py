"""Shared synthetic-data fixtures for oasis2 tests.

No real MRI data is available in this environment (see
docs/superpowers/specs/2026-09-28-py-pipeline-conversion-design.md §1), so every
fixture here is a small, fully synthetic stand-in with the same shapes the real
data pipeline produces.
"""
import numpy as np
import pandas as pd
import pytest
import torch

N_REGIONS = 8
N_COV = 6


@pytest.fixture
def synthetic_connectome():
    """An (8, 8) symmetric non-negative adjacency matrix, zero diagonal."""
    rng = np.random.default_rng(0)
    A = rng.random((N_REGIONS, N_REGIONS))
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    return A


@pytest.fixture
def synthetic_demo_df():
    """Longitudinal demographics for 6 synthetic subjects, 2-4 visits each."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(6):
        sid = f"OAS2_{i:04d}"
        n_visits = rng.integers(2, 5)
        age0 = rng.uniform(65, 85)
        group = "Nondemented" if i % 3 == 0 else "Demented"
        cdr0 = 0.0 if group == "Nondemented" else rng.choice([0.5, 1.0])
        for v in range(n_visits):
            rows.append({
                "subject_id": sid,
                "session_id": f"{sid}_MR{v+1}",
                "Subject ID": sid,
                "MRI ID": f"{sid}_MR{v+1}",
                "Visit": v + 1,
                "MR Delay": v * 365,
                "Group": group,
                "M/F": "F" if i % 2 == 0 else "M",
                "Age": age0 + v,
                "EDUC": rng.uniform(8, 20),
                "SES": rng.integers(1, 5),
                "MMSE": max(0, 30 - v * rng.uniform(0, 2) - (5 if group == "Demented" else 0)),
                "CDR": min(2.0, cdr0 + v * 0.25 if group == "Demented" else 0.0),
                "eTIV": rng.uniform(1300, 1700),
                "nWBV": rng.uniform(0.65, 0.82) - v * 0.005,
                "ASF": rng.uniform(0.9, 1.3),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_regional_volumes(synthetic_demo_df):
    """One row per (subject, session) with N_REGIONS normalized volume columns."""
    rng = np.random.default_rng(2)
    df = synthetic_demo_df[["subject_id", "session_id"]].copy()
    for r in range(N_REGIONS):
        df[f"region_{r}_norm"] = rng.uniform(0.3, 1.0, size=len(df))
    return df


@pytest.fixture
def synthetic_batch():
    """A dict shaped like a `collate_longitudinal` batch: B=4 subjects, K=3 visits, N=8 regions, D=1."""
    B, K, N, D = 4, 3, N_REGIONS, 1
    g = torch.Generator().manual_seed(0)
    node_features = torch.rand(B, K, N, D, generator=g)
    times = torch.stack([torch.arange(K, dtype=torch.float32)] * B)
    covariates = torch.rand(B, N_COV, generator=g)
    mmse = torch.rand(B, K, generator=g) * 30
    cdr = torch.randint(0, 4, (B, K), generator=g)
    mask = torch.ones(B, K, dtype=torch.bool)
    adj = torch.rand(1, N, N, generator=g)
    adj = (adj + adj.transpose(1, 2)) / 2
    return {
        "node_features": node_features, "times": times, "covariates": covariates,
        "mmse": mmse, "cdr": cdr, "mask": mask, "adj": adj,
    }
