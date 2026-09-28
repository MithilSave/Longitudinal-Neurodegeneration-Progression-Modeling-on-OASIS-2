"""Cross-model benchmark runner.

Ported verbatim from OASIS2_Colab.py (lines 2007-2159): `evaluate_dl_model`,
`run_benchmark`.

`evaluate_dl_model` has three branches keyed on `model_name`:

- 'TADM': `TemporalDiffusionModel` has no `forward()` (see
  `oasis2.models.tadm` module docstring) so predictions are drawn via
  `model.sample(...)`, then clinical heads are applied explicitly via
  `model.mmse_head(...)` / `model.cdr_head(...)` on the sampled state.
- 'ND-VAE': `PhysicsConstrainedNDVAE` is called directly as `model(...)`,
  returning `(recon, mmse_pred, cdr_logits, ...)`.
- default (ST-GNN-ODE): `STGraphNeuralODE` is called directly as
  `model(...)`, returning `(recon, mmse_pred, cdr_logits)`.

This branching is preserved faithfully rather than unified into a common
interface, since the underlying models genuinely have different call
surfaces (see Task 8 / `oasis2.models.tadm`).

As with the other `models/` modules, the original script relies on a
shared top-of-script `DEVICE` global; this module defines its own
module-scope `DEVICE = get_device()` to reproduce that behaviour.
"""
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from oasis2.config import get_device, get_outputs_dir
from oasis2.logging_utils import setup_logging

logger = setup_logging("benchmark")
DEVICE = get_device()


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
