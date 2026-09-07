"""
OASIS2_Local_Pipeline.py — Local version of the OASIS-2 Longitudinal
Neurodegeneration Progression Modeling pipeline.

Implements 3 deep learning models + 1 NDM baseline for longitudinal
brain atrophy modeling:
  1. NDM Baseline (Raj et al., 2012)
  2. ST-GNN-ODE: Continuous-Time Graph Neural ODE
  3. ND-VAE: Physics-Constrained Network Diffusion VAE
  4. TADM: Temporally-Aware Trajectory Diffusion Model
"""

import subprocess, sys, os
import math
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings('ignore')

try:
    gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if gpu.returncode != 0:
        print('[WARNING] No GPU detected.')
    else:
        lines = [l for l in gpu.stdout.split('\n')
                 if any(x in l for x in ['GeForce','Tesla','T4','A100','RTX','GTX'])]
        print('[OK] GPU:', lines[0].strip() if lines else 'detected')
except FileNotFoundError:
    print('[WARNING] nvidia-smi not found — running on CPU.')

# ── Local path setup ──────────────
PROJECT_ROOT = Path(os.getcwd())
LOCAL_BASE = PROJECT_ROOT

# Ensure output directories exist
for sub in ['data/raw', 'data/processed', 'outputs', 'models',
            'outputs/phase0', 'outputs/phase3', 'outputs/phase4',
            'outputs/benchmark', 'outputs/checkpoints', 'reports']:
    (LOCAL_BASE / sub).mkdir(parents=True, exist_ok=True)

print('[OK] Project root:', PROJECT_ROOT)

# Imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import expm
from sklearn.metrics import f1_score as sk_f1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'[OK] Device: {DEVICE}')


# ===========================================================================
# PHASE 0: DATA AUDIT
# ===========================================================================

def run_phase_0_audit():
    print("\n--- Phase 0: Data Audit ---")
    data_dir = PROJECT_ROOT / 'data' / 'raw'
    csv_paths = list(data_dir.rglob('oasis_longitudinal*.csv')) + list(data_dir.rglob('oasis_longitudinal*.xlsx'))

    if not csv_paths:
        print("[WARNING] Demographic CSV not found in data/raw. Creating dummy data.")
        np.random.seed(42)
        subjects = [f'OAS2_{i:04d}' for i in range(1, 51)]
        records = []
        for sub in subjects:
            n_visits = np.random.randint(2, 5)
            baseline_age = np.random.randint(65, 85)
            for v in range(n_visits):
                records.append({
                    'Subject ID': sub, 'MRI ID': f'{sub}_MR{v+1}', 'Visit': v+1,
                    'MR Delay': v * 365 + np.random.randint(-30, 30) if v > 0 else 0,
                    'M/F': np.random.choice(['M', 'F']),
                    'Age': baseline_age + v, 'EDUC': np.random.randint(8, 20),
                    'SES': np.random.randint(1, 5),
                    'MMSE': np.random.randint(20, 31),
                    'CDR': np.random.choice([0.0, 0.5, 1.0]),
                    'eTIV': np.random.normal(1500, 100), 'nWBV': np.random.normal(0.75, 0.05),
                    'ASF': np.random.normal(1.1, 0.1)
                })
        df = pd.DataFrame(records)
    else:
        csv_path = csv_paths[0]
        if csv_path.suffix.lower() in ('.xlsx', '.xls'):
            df = pd.read_excel(csv_path)
        else:
            df = pd.read_csv(csv_path)

    df.columns = [c.strip() for c in df.columns]
    if 'subject_id' not in df.columns and 'Subject ID' in df.columns:
        df['subject_id'] = df['Subject ID'].str.strip()

    visit_counts = df.groupby('subject_id').size()
    multi_visit = visit_counts[visit_counts >= 2].index
    df_longitudinal = df[df['subject_id'].isin(multi_visit)].copy()

    print(f"Total subjects: {df['subject_id'].nunique()}")
    print(f"Subjects with >=2 visits: {len(multi_visit)}")

    out_path = PROJECT_ROOT / 'outputs' / 'phase0' / 'subjects_index.csv'
    df_longitudinal.to_csv(out_path, index=False)
    print(f"[OK] Audit complete. Saved index to {out_path}")
    return df_longitudinal


# ===========================================================================
# CONNECTOME & NDM BASELINE (Phase 3)
# ===========================================================================

N_REGIONS = 56  # Desikan-Killiany parcellation

def build_synthetic_connectome(n_regions=N_REGIONS, seed=42):
    """Build synthetic structural connectivity matrix."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n_regions, n_regions))
    # Scale-free-like connectivity
    for i in range(n_regions):
        n_conn = rng.integers(3, min(10, n_regions))
        targets = rng.choice(n_regions, size=n_conn, replace=False)
        for j in targets:
            if i != j:
                w = rng.uniform(0.1, 1.0)
                A[i, j] = w
                A[j, i] = w
    return A


def load_connectome():
    """Load or build connectome adjacency matrix."""
    conn_path = PROJECT_ROOT / 'data' / 'processed' / 'template_connectome.npy'
    if conn_path.exists():
        A = np.load(str(conn_path))
        print(f"[OK] Loaded connectome: {A.shape}")
    else:
        A = build_synthetic_connectome()
        np.save(str(conn_path), A)
        print(f"[OK] Built synthetic connectome: {A.shape}")
    return A


def compute_laplacian(A):
    """Compute normalized symmetric Laplacian."""
    A_hat = A + np.eye(A.shape[0])
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A_hat.sum(axis=1) + 1e-8))
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A_hat @ D_inv_sqrt
    return L


def ndm_propagate(x0, t, beta, L):
    """NDM: x(t) = exp(-β*L*t) * x(0)."""
    return expm(-beta * L * t) @ x0


def run_phase3_ndm(df, A):
    """Phase 3: Baseline NDM — fit β per subject, compute w-scores."""
    print("\n--- Phase 3: Network Diffusion Model (Baseline) ---")
    L = compute_laplacian(A)
    N = A.shape[0]

    # Compute w-scores from nWBV
    w_scores_data = []
    for sid, grp in df.groupby('subject_id'):
        grp = grp.sort_values('MR Delay' if 'MR Delay' in grp.columns else 'Visit')
        nwbv = grp['nWBV'].fillna(0.75).values
        ws = np.zeros(N)
        ws[:] = (nwbv[0] - 0.75) / 0.05  # simple z-score from population mean
        w_scores_data.append({'subject_id': sid, **{f'r{i}_wscore': ws[i] for i in range(N)}})

    w_scores = pd.DataFrame(w_scores_data)

    # Fit β per subject
    beta_records = []
    for sid, grp in df.groupby('subject_id'):
        grp = grp.sort_values('MR Delay' if 'MR Delay' in grp.columns else 'Visit')
        if len(grp) < 2:
            continue
        nwbv = grp['nWBV'].fillna(0.75).values
        x0 = np.full(N, (nwbv[0] - 0.75) / 0.05)

        if 'MR Delay' in grp.columns:
            t_years = grp['MR Delay'].fillna(0).values / 365.25
        else:
            t_years = np.arange(len(grp), dtype=float)

        # Simple grid search for β
        best_beta, best_err = 0.01, float('inf')
        for beta in np.linspace(0.001, 0.5, 50):
            err = 0
            for ki in range(1, len(grp)):
                dt = max(t_years[ki] - t_years[0], 0.01)
                x_pred = ndm_propagate(x0, dt, beta, L)
                x_true = np.full(N, (nwbv[ki] - 0.75) / 0.05)
                err += np.mean((x_pred - x_true) ** 2)
            if err < best_err:
                best_err = err
                best_beta = beta

        beta_records.append({'subject_id': sid, 'beta': best_beta, 'fit_error': best_err})

    beta_df = pd.DataFrame(beta_records)
    print(f"[OK] NDM fitted β for {len(beta_df)} subjects. Median β={beta_df['beta'].median():.4f}")

    # Validation: predict and compute Pearson r
    all_pred, all_true = [], []
    for _, row in beta_df.iterrows():
        sid = row['subject_id']
        beta = row['beta']
        grp = df[df['subject_id'] == sid].sort_values('MR Delay' if 'MR Delay' in df.columns else 'Visit')
        nwbv = grp['nWBV'].fillna(0.75).values
        x0 = np.full(N, (nwbv[0] - 0.75) / 0.05)
        if 'MR Delay' in grp.columns:
            t_years = grp['MR Delay'].fillna(0).values / 365.25
        else:
            t_years = np.arange(len(grp), dtype=float)

        for ki in range(1, len(grp)):
            dt = max(t_years[ki] - t_years[0], 0.01)
            x_pred = ndm_propagate(x0, dt, beta, L)
            x_true = np.full(N, (nwbv[ki] - 0.75) / 0.05)
            all_pred.extend(x_pred)
            all_true.extend(x_true)

    pearson_r = np.corrcoef(all_pred, all_true)[0, 1] if all_pred else 0.0
    rmse = np.sqrt(np.mean((np.array(all_pred) - np.array(all_true))**2)) if all_pred else 0.0
    mae = np.mean(np.abs(np.array(all_pred) - np.array(all_true))) if all_pred else 0.0

    validation = {'pearson_r': round(float(pearson_r), 4), 'rmse': round(float(rmse), 4), 'mae': round(float(mae), 4)}
    print(f"[OK] NDM validation: Pearson r={validation['pearson_r']}, RMSE={validation['rmse']}")

    return {
        'beta_df': beta_df, 'w_scores': w_scores, 'L': L,
        'validation': validation, 'eigenvalues': np.linalg.eigvalsh(L),
    }


# ===========================================================================
# DL DATASET BUILDER
# ===========================================================================

CDR_MAP = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}


class LongitudinalGraphDataset(Dataset):
    def __init__(self, subjects_data, adj_matrix):
        self.subjects = subjects_data
        self.adj = torch.tensor(adj_matrix, dtype=torch.float32)

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
        }


def collate_fn(batch):
    max_k = max(b['times'].shape[0] for b in batch)
    N = batch[0]['node_features'].shape[1]
    D = batch[0]['node_features'].shape[2]
    C = batch[0]['covariates'].shape[0]
    B = len(batch)

    nf = torch.zeros(B, max_k, N, D)
    t = torch.zeros(B, max_k)
    cov = torch.zeros(B, C)
    mmse = torch.zeros(B, max_k)
    cdr = torch.full((B, max_k), -1, dtype=torch.long)
    mask = torch.zeros(B, max_k, dtype=torch.bool)

    for i, b in enumerate(batch):
        k = b['times'].shape[0]
        nf[i, :k] = b['node_features']
        t[i, :k] = b['times']
        cov[i] = b['covariates']
        mmse[i, :k] = b['mmse']
        cdr[i, :k] = b['cdr']
        mask[i, :k] = True

    return {'node_features': nf, 'times': t, 'covariates': cov,
            'mmse': mmse, 'cdr': cdr, 'mask': mask, 'adj': batch[0]['adj'].unsqueeze(0)}


def build_datasets(df, w_scores, A, seed=42):
    """Build train/val/test DataLoaders."""
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    A_hat = A + np.eye(N)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A_hat.sum(axis=1) + 1e-8))
    adj_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

    wscore_cols = [c for c in w_scores.columns if c.endswith('_wscore')]
    subjects_data = []

    for sid, grp in df.groupby('subject_id'):
        grp = grp.sort_values('MR Delay' if 'MR Delay' in grp.columns else 'Visit')
        if len(grp) < 2:
            continue

        if 'MR Delay' in grp.columns:
            t_years = grp['MR Delay'].fillna(0).values.astype(float) / 365.25
        else:
            t_years = np.arange(len(grp), dtype=float)
        t_years = t_years - t_years[0]
        K = len(grp)

        # Node features
        if wscore_cols:
            ws_sub = w_scores[w_scores['subject_id'] == sid]
            if len(ws_sub) >= 1:
                x_vals = ws_sub.iloc[0][wscore_cols].values.astype(float)
                x_vals = np.nan_to_num(x_vals, nan=0.0)
                n_ws = min(len(x_vals), N)
                node_feat = np.zeros((K, N, 1))
                for ki in range(K):
                    node_feat[ki, :n_ws, 0] = x_vals[:n_ws] * np.exp(-0.1 * t_years[ki])
            else:
                node_feat = np.zeros((K, N, 1))
        else:
            nwbv_vals = grp['nWBV'].fillna(0.75).values
            node_feat = np.zeros((K, N, 1))
            for ki in range(K):
                node_feat[ki, :, 0] = nwbv_vals[ki]

        # Covariates
        r0 = grp.iloc[0]
        age = float(r0.get('Age', 70)) / 100.0
        sex = 1.0 if str(r0.get('M/F', 'M')).strip().upper() == 'M' else 0.0
        educ = float(r0.get('EDUC', 12)) / 20.0
        ses = float(r0.get('SES', 3)) / 5.0 if not pd.isna(r0.get('SES')) else 0.5
        etiv = float(r0.get('eTIV', 1500)) / 2000.0
        asf = float(r0.get('ASF', 1.0))
        covariates = np.array([age, sex, educ, ses, etiv, asf], dtype=float)

        mmse_vals = grp['MMSE'].fillna(25).values.astype(float) / 30.0
        cdr_vals = grp['CDR'].fillna(0).values.astype(float)
        cdr_labels = np.array([CDR_MAP.get(c, 0) for c in cdr_vals])

        subjects_data.append({
            'node_features': torch.tensor(node_feat, dtype=torch.float32),
            'times': torch.tensor(t_years, dtype=torch.float32),
            'covariates': torch.tensor(covariates, dtype=torch.float32),
            'mmse': torch.tensor(mmse_vals, dtype=torch.float32),
            'cdr': torch.tensor(cdr_labels, dtype=torch.long),
        })

    if not subjects_data:
        return None, None, None

    n = len(subjects_data)
    perm = rng.permutation(n)
    n_train, n_val = int(0.7 * n), int(0.15 * n)

    def make_loader(data, shuffle=False):
        return DataLoader(LongitudinalGraphDataset(data, adj_norm),
                         batch_size=8, shuffle=shuffle, collate_fn=collate_fn)

    return (make_loader([subjects_data[i] for i in perm[:n_train]], True),
            make_loader([subjects_data[i] for i in perm[n_train:n_train+n_val]]),
            make_loader([subjects_data[i] for i in perm[n_train+n_val:]]))


# ===========================================================================
# MODEL 1: ST-GNN-ODE
# ===========================================================================

class SinusoidalTimeEnc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        if t.dim() == 0: t = t.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
    def forward(self, x, adj):
        B, N, _ = x.shape
        h = self.W(x)
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        e = F.leaky_relu(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1), 0.2)
        if adj.dim() == 2: adj = adj.unsqueeze(0)
        e = e.masked_fill(adj.expand(B, -1, -1) < 1e-6, float('-inf'))
        alpha = torch.nan_to_num(F.softmax(e, dim=-1), nan=0.0)
        return F.elu(torch.bmm(alpha, h))


class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
    def forward(self, x, adj):
        if adj.dim() == 2: adj = adj.unsqueeze(0)
        return F.elu(self.W(torch.bmm(adj.expand(x.size(0), -1, -1), x)))


class ODEFunc(nn.Module):
    def __init__(self, hid, n_cov, t_dim=16):
        super().__init__()
        self.gat = GATLayer(hid, hid)
        self.time_enc = SinusoidalTimeEnc(t_dim)
        self.mlp = nn.Sequential(nn.Linear(hid + n_cov + t_dim, hid), nn.ELU(), nn.Linear(hid, hid))
        self.adj = self.cov = None
    def set_ctx(self, adj, cov): self.adj, self.cov = adj, cov
    def forward(self, t, h):
        B, N, D = h.shape
        g = self.gat(h, self.adj)
        te = self.time_enc(t.expand(B)).unsqueeze(1).expand(-1, N, -1)
        ce = self.cov.unsqueeze(1).expand(-1, N, -1)
        return self.mlp(torch.cat([g, ce, te], dim=-1))


class STGraphNeuralODE(nn.Module):
    def __init__(self, nf=1, hid=32, nc=6):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(nf + nc, hid), nn.ELU(), nn.Linear(hid, hid))
        self.ode = ODEFunc(hid, nc)
        self.dec_r = nn.Linear(hid, nf)
        self.dec_m = nn.Sequential(nn.Linear(hid, 16), nn.ELU(), nn.Linear(16, 1))
        self.dec_c = nn.Sequential(nn.Linear(hid, 16), nn.ELU(), nn.Linear(16, 4))

    def forward(self, nf, times, cov, adj, mask):
        B, K, N, D = nf.shape
        ce = cov.unsqueeze(1).expand(-1, N, -1)
        h = self.enc(torch.cat([nf[:, 0], ce], dim=-1))
        self.ode.set_ctx(adj, cov)
        rl, ml, cl = [], [], []
        prev_t = torch.zeros(1, device=h.device)
        for ki in range(K):
            tt = times[:, ki].mean()
            dt = tt - prev_t
            if dt > 0.001:
                step = dt / 5
                for _ in range(5):
                    h = h + step * self.ode(prev_t, h)
                    prev_t = prev_t + step
            prev_t = tt
            rl.append(self.dec_r(h))
            hg = h.mean(dim=1)
            ml.append(self.dec_m(hg).squeeze(-1))
            cl.append(self.dec_c(hg))
        return torch.stack(rl, 1), torch.stack(ml, 1), torch.stack(cl, 1)


# ===========================================================================
# MODEL 2: ND-VAE
# ===========================================================================

class PhysicsNDVAE(nn.Module):
    def __init__(self, nr, nf=1, nc=6, zd=16, hid=32):
        super().__init__()
        self.nr, self.zd = nr, zd
        self.enc_proj = nn.Linear(nr * nf + nc, hid)
        self.enc_gru = nn.GRU(hid, hid, batch_first=True)
        self.enc_mu = nn.Linear(hid, zd)
        self.enc_lv = nn.Linear(hid, zd)
        self.dec = nn.Sequential(nn.Linear(zd + 1 + nc, hid), nn.ELU(), nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, nr * nf))
        self.dec_m = nn.Sequential(nn.Linear(zd + 1 + nc, 16), nn.ELU(), nn.Linear(16, 1))
        self.dec_c = nn.Sequential(nn.Linear(zd + 1 + nc, 16), nn.ELU(), nn.Linear(16, 4))
        self.beta_diff = nn.Parameter(torch.tensor(0.1))

    def forward(self, nf, times, cov, adj, mask):
        B, K, N, D = nf.shape
        xf = nf.view(B, K, N * D)
        ce = cov.unsqueeze(1).expand(-1, K, -1)
        inp = F.elu(self.enc_proj(torch.cat([xf, ce], -1)))
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(inp, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.enc_gru(packed)
        mu, lv = self.enc_mu(h.squeeze(0)), self.enc_lv(h.squeeze(0))
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

        rl, ml, cl = [], [], []
        for ki in range(K):
            d = torch.cat([z, times[:, ki:ki+1], cov], -1)
            rl.append(self.dec(d).view(-1, N, D))
            ml.append(self.dec_m(d).squeeze(-1))
            cl.append(self.dec_c(d))
        return torch.stack(rl, 1), torch.stack(ml, 1), torch.stack(cl, 1), mu, lv

    def physics_loss(self, recon, times, adj, mask):
        B, K, N, D = recon.shape
        if K < 2: return torch.tensor(0.0, device=recon.device)
        adj2d = adj[0] if adj.dim() == 3 else adj
        L = torch.eye(N, device=recon.device) - adj2d
        total, cnt = 0.0, 0
        for ki in range(K - 1):
            dt = (times[:, ki+1] - times[:, ki]).clamp(min=0.01)
            dXdt = (recon[:, ki+1] - recon[:, ki]) / dt.unsqueeze(-1).unsqueeze(-1)
            LX = torch.matmul(L.unsqueeze(0), recon[:, ki])
            res = dXdt + self.beta_diff * LX
            v = mask[:, ki] & mask[:, ki+1]
            if v.any(): total = total + res[v].pow(2).mean(); cnt += 1
        return total / max(cnt, 1)


# ===========================================================================
# MODEL 3: TADM (Diffusion Model)
# ===========================================================================

def cosine_schedule(T, s=0.008):
    x = torch.linspace(0, T, T + 1)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    return torch.clamp(1 - ac[1:] / ac[:-1], 0.0001, 0.999)


class TADM(nn.Module):
    def __init__(self, nr, nf=1, nc=6, T=200, cd=32):
        super().__init__()
        self.nr, self.dd, self.T = nr, nr * nf, T
        betas = cosine_schedule(T)
        alphas = 1 - betas
        ac = torch.cumprod(alphas, 0)
        self.register_buffer('ac', ac)
        self.register_buffer('sac', torch.sqrt(ac))
        self.register_buffer('somac', torch.sqrt(1 - ac))

        self.gcn = GraphConvLayer(nf, cd)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.t_mlp = nn.Sequential(SinusoidalTimeEnc(cd), nn.Linear(cd, cd), nn.GELU())
        self.c_mlp = nn.Sequential(nn.Linear(nc, cd), nn.GELU())
        self.cond_proj = nn.Linear(3 * cd, cd)
        self.step_emb = nn.Embedding(T, cd)
        self.net = nn.Sequential(nn.Linear(self.dd, 64), nn.GELU())
        self.fg = nn.Linear(2 * cd, 64)
        self.fb = nn.Linear(2 * cd, 64)
        self.out = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, self.dd))
        self.mmse_head = nn.Sequential(nn.Linear(self.dd, 16), nn.GELU(), nn.Linear(16, 1))
        self.cdr_head = nn.Sequential(nn.Linear(self.dd, 16), nn.GELU(), nn.Linear(16, 4))

    def condition(self, x0, dt, cov, adj):
        g = self.pool(self.gcn(x0, adj).permute(0, 2, 1)).squeeze(-1)
        te = self.t_mlp[2](self.t_mlp[1](self.t_mlp[0](dt)))
        ce = self.c_mlp(cov)
        return self.cond_proj(torch.cat([g, te, ce], -1))

    def training_loss(self, nf, times, cov, adj, mask):
        B, K, N, D = nf.shape
        losses = []
        for ki in range(1, K):
            v = mask[:, ki] & mask[:, 0]
            if not v.any(): continue
            x0b = nf[v, 0]; xt = nf[v, ki].reshape(-1, self.dd)
            dt = times[v, ki] - times[v, 0]
            cond = self.condition(x0b, dt, cov[v], adj)
            Bv = xt.shape[0]
            td = torch.randint(0, self.T, (Bv,), device=xt.device)
            noise = torch.randn_like(xt)
            xn = self.sac[td].view(-1, 1) * xt + self.somac[td].view(-1, 1) * noise
            se = self.step_emb(td)
            cf = torch.cat([cond, se], -1)
            h = self.net(xn)
            h = self.fg(cf) * h + self.fb(cf)
            pred = self.out(h)
            losses.append(F.mse_loss(pred, noise))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=nf.device)

    @torch.no_grad()
    def sample(self, x0b, dt, cov, adj, steps=10):
        B = x0b.shape[0]
        cond = self.condition(x0b, dt, cov, adj)
        x = torch.randn(B, self.dd, device=x0b.device)
        idxs = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=x.device)
        for i, ti in enumerate(idxs):
            tb = ti.expand(B).long()
            se = self.step_emb(tb)
            cf = torch.cat([cond, se], -1)
            h = self.net(x); h = self.fg(cf) * h + self.fb(cf); np_ = self.out(h)
            at = self.ac[ti]
            ap = self.ac[idxs[i+1]] if i < len(idxs) - 1 else torch.tensor(1.0, device=x.device)
            x0p = (x - torch.sqrt(1 - at) * np_) / torch.sqrt(at)
            x = torch.sqrt(ap) * x0p + torch.sqrt(1 - ap) * np_
        return x.view(B, self.nr, -1)


# ===========================================================================
# TRAINING LOOPS
# ===========================================================================

def train_model(model, train_ld, val_ld, n_epochs, lr=1e-3, patience=10, name="Model", is_vae=False, is_diffusion=False, n_regions=None):
    """Unified training loop for all 3 DL models."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_vl, best_st, no_imp = float('inf'), None, 0

    for ep in range(n_epochs):
        model.train(); tl = 0.0
        beta_kl = min(1.0, ep / 20.0) * 0.1 if is_vae else 0

        for b in train_ld:
            nf, t, cov, adj, msk = [b[k].to(DEVICE) for k in ['node_features','times','covariates','adj','mask']]
            mt, ct = b['mmse'].to(DEVICE), b['cdr'].to(DEVICE)

            if is_diffusion:
                loss = model.training_loss(nf, t, cov, adj, msk)
            else:
                if is_vae:
                    recon, mp, cl, mu, lv = model(nf, t, cov, adj, msk)
                    lr_ = F.l1_loss(recon[msk], nf[msk])
                    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                    lp = model.physics_loss(recon, t, adj, msk)
                else:
                    recon, mp, cl = model(nf, t, cov, adj, msk)
                    lr_ = F.mse_loss(recon[msk], nf[msk])
                    kl, lp = 0, 0

                lm = F.mse_loss(mp[msk], mt[msk])
                vc = msk & (ct >= 0)
                lc = F.cross_entropy(cl[vc], ct[vc]) if vc.any() else torch.tensor(0.0, device=DEVICE)
                loss = lr_ + beta_kl * kl + 0.01 * lp + 0.5 * lm + 0.3 * lc

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()

        sched.step()

        model.eval(); vl = 0.0
        with torch.no_grad():
            for b in val_ld:
                nf, t, cov, adj, msk = [b[k].to(DEVICE) for k in ['node_features','times','covariates','adj','mask']]
                mt, ct = b['mmse'].to(DEVICE), b['cdr'].to(DEVICE)
                if is_diffusion:
                    vl += model.training_loss(nf, t, cov, adj, msk).item()
                else:
                    if is_vae:
                        recon, mp, cl, mu, lv = model(nf, t, cov, adj, msk)
                    else:
                        recon, mp, cl = model(nf, t, cov, adj, msk)
                    vl += F.mse_loss(recon[msk], nf[msk]).item()

        avg_t = tl / max(len(train_ld), 1)
        avg_v = vl / max(len(val_ld), 1)
        if ep % 10 == 0:
            print(f"  [{name}] Epoch {ep}: train={avg_t:.4f}, val={avg_v:.4f}")

        if avg_v < best_vl:
            best_vl = avg_v
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  [{name}] Early stop at epoch {ep}")
                break

    if best_st: model.load_state_dict(best_st)
    model.to(DEVICE)
    print(f"  [{name}] Done. Best val loss: {best_vl:.4f}")
    return model


# ===========================================================================
# BENCHMARK
# ===========================================================================

def evaluate_model(model, test_ld, name, is_vae=False, is_diffusion=False):
    model.eval()
    recs, trues, mps, mts, cps, cts = [], [], [], [], [], []
    with torch.no_grad():
        for b in test_ld:
            nf, t, cov, adj, msk = [b[k].to(DEVICE) for k in ['node_features','times','covariates','adj','mask']]
            mt, ct = b['mmse'].to(DEVICE), b['cdr'].to(DEVICE)

            if is_diffusion:
                B, K, N, D = nf.shape
                for ki in range(1, K):
                    v = msk[:, ki] & msk[:, 0]
                    if not v.any(): continue
                    pred = model.sample(nf[v, 0], t[v, ki] - t[v, 0], cov[v], adj, 10)
                    recs.append(pred.cpu()); trues.append(nf[v, ki].cpu())
                    pf = pred.view(-1, model.dd)
                    mps.append(model.mmse_head(pf).squeeze(-1).cpu()); mts.append(mt[v, ki].cpu())
                    cps.append(model.cdr_head(pf).cpu()); cts.append(ct[v, ki].cpu())
            else:
                if is_vae:
                    recon, mp, cl, _, _ = model(nf, t, cov, adj, msk)
                else:
                    recon, mp, cl = model(nf, t, cov, adj, msk)
                for ki in range(1, nf.shape[1]):
                    v = msk[:, ki]
                    if v.any():
                        recs.append(recon[v, ki].cpu()); trues.append(nf[v, ki].cpu())
                        mps.append(mp[v, ki].cpu()); mts.append(mt[v, ki].cpu())
                        cps.append(cl[v, ki].cpu()); cts.append(ct[v, ki].cpu())

    metrics = {}
    if recs:
        rc, tc = torch.cat(recs), torch.cat(trues)
        metrics['atrophy_rmse'] = round(torch.sqrt(F.mse_loss(rc, tc)).item(), 4)
        metrics['atrophy_mae'] = round(F.l1_loss(rc, tc).item(), 4)
        rf, tf = rc.flatten().numpy(), tc.flatten().numpy()
        metrics['atrophy_pearson_r'] = round(float(np.corrcoef(rf, tf)[0, 1]) if np.std(rf) > 1e-10 else 0.0, 4)
    if mps:
        metrics['mmse_mae'] = round(F.l1_loss(torch.cat(mps), torch.cat(mts)).item(), 4)
    if cps:
        cp, ct_ = torch.cat(cps), torch.cat(cts)
        v = ct_ >= 0
        if v.any():
            pl = cp[v].argmax(-1)
            metrics['cdr_accuracy'] = round((pl == ct_[v]).float().mean().item(), 4)
            try: metrics['cdr_f1_macro'] = round(sk_f1(ct_[v].numpy(), pl.numpy(), average='macro', zero_division=0), 4)
            except: metrics['cdr_f1_macro'] = 0.0
    return metrics


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("OASIS-2 LONGITUDINAL NEURODEGENERATION MODELING (LOCAL)")
    print("=" * 70)

    # Phase 0
    df = run_phase_0_audit()

    # Phase 3: NDM Baseline
    A = load_connectome()
    p3 = run_phase3_ndm(df, A)
    beta_df, w_scores = p3['beta_df'], p3['w_scores']

    # Build DL datasets
    print("\n--- Building DL Datasets ---")
    train_ld, val_ld, test_ld = build_datasets(df, w_scores, A)

    if train_ld is None:
        print("[ERROR] Could not build datasets. Exiting.")
        sys.exit(1)

    N = A.shape[0]

    # Model 1: ST-GNN-ODE
    print("\n--- Training Model 1: ST-GNN-ODE ---")
    m1 = STGraphNeuralODE(nf=1, hid=32, nc=6).to(DEVICE)
    m1 = train_model(m1, train_ld, val_ld, n_epochs=50, name="ST-GNN-ODE")

    # Model 2: ND-VAE
    print("\n--- Training Model 2: ND-VAE ---")
    m2 = PhysicsNDVAE(nr=N, nf=1, nc=6).to(DEVICE)
    m2 = train_model(m2, train_ld, val_ld, n_epochs=100, name="ND-VAE", is_vae=True)

    # Model 3: TADM
    print("\n--- Training Model 3: TADM ---")
    m3 = TADM(nr=N, nf=1, nc=6).to(DEVICE)
    m3 = train_model(m3, train_ld, val_ld, n_epochs=200, name="TADM", is_diffusion=True, patience=15)

    # Benchmark
    print("\n--- Benchmark ---")
    results = {
        'NDM (Baseline)': {
            'atrophy_rmse': p3['validation']['rmse'],
            'atrophy_mae': p3['validation']['mae'],
            'atrophy_pearson_r': p3['validation']['pearson_r'],
            'mmse_mae': '-', 'cdr_accuracy': '-', 'cdr_f1_macro': '-',
        },
        'ST-GNN-ODE': evaluate_model(m1, test_ld, 'ST-GNN-ODE'),
        'ND-VAE': evaluate_model(m2, test_ld, 'ND-VAE', is_vae=True),
        'TADM': evaluate_model(m3, test_ld, 'TADM', is_diffusion=True),
    }
    bench_df = pd.DataFrame(results).T
    bench_df.index.name = 'Model'
    bench_df.to_csv(PROJECT_ROOT / 'outputs' / 'benchmark' / 'benchmark_results.csv')

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS — All Models vs NDM Baseline")
    print("=" * 70)
    print(bench_df.to_string())
    print("=" * 70)

    # Save models
    torch.save(m1.state_dict(), PROJECT_ROOT / 'models' / 'st_gnn_ode.pt')
    torch.save(m2.state_dict(), PROJECT_ROOT / 'models' / 'nd_vae.pt')
    torch.save(m3.state_dict(), PROJECT_ROOT / 'models' / 'tadm.pt')
    print(f"\n[OK] Models saved to {PROJECT_ROOT / 'models'}")
    print("\nPIPELINE COMPLETE.")