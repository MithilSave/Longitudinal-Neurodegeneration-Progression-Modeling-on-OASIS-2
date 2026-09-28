"""Digital twin forward-prediction layer: autoregressive trajectory rollout,
Monte-Carlo uncertainty bands, and counterfactual scenarios.

Ported from OASIS2_Complete_Colab_Pipeline.ipynb cell 190 ("Phase 6: Trajectory
Simulator & Counterfactuals" / "Phase 7: Advanced Validation Metrics"), which
previously only ran inline in the notebook against notebook-global variables
(m2, nf_in, t_in, ...) and was never callable as a library function — meaning
the project's stated core deliverable (predict which regions are affected
next, per implementation.md §1) could not run outside the notebook at all.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from oasis2.logging_utils import setup_logging

logger = setup_logging("digital_twin")


def simulate_rollout(model: nn.Module, nf_init: torch.Tensor, cov: torch.Tensor, adj: torch.Tensor, steps: int = 3) -> torch.Tensor:
    """Autoregressive forward rollout: repeatedly predicts the next visit and
    feeds it back in as input. `nf_init` is (B, 1, N, D) — a single baseline visit.
    Returns (B, steps, N, D)."""
    model.eval()
    B, _, N, D = nf_init.shape
    current_nf = nf_init.clone()
    trajectories = []

    with torch.no_grad():
        for _step in range(steps):
            K = current_nf.shape[1]
            dummy_next = torch.zeros_like(current_nf[:, 0:1])
            nf_padded = torch.cat([current_nf, dummy_next], dim=1)
            t_padded = torch.arange(K + 1, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(nf_init.device)
            msk_padded = torch.ones((B, K + 1), dtype=torch.bool).to(nf_init.device)
            msk_padded[:, -1] = False

            recon, _, _, _, _ = model(nf_padded, t_padded, cov, adj, msk_padded)
            next_state = recon[:, -1:, :, :]
            trajectories.append(next_state)
            current_nf = torch.cat([current_nf, next_state], dim=1)

    return torch.cat(trajectories, dim=1)


def generate_confidence_bands(
    model: nn.Module, nf, times, cov, adj, mask, n_samples: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte-Carlo sample the VAE's stochastic reparameterization n_samples times
    and return (mean, 5th percentile, 95th percentile) prediction arrays.

    The model must be put into train() mode to re-sample the VAE's latent
    noise on every forward pass (PhysicsConstrainedNDVAE.reparameterize always
    samples eps ~ N(0,1), but train() vs eval() affects GRU/dropout behavior
    the same way it does for any other torch module) — this function always
    restores the model's original mode before returning, including when
    sampling raises, so it can't leak train-mode into unrelated code that
    reuses the same model instance afterward.
    """
    was_training = model.training
    model.train()
    samples = []
    try:
        with torch.no_grad():
            for _ in range(n_samples):
                recon, _, _, _, _ = model(nf, times, cov, adj, mask)
                samples.append(recon.cpu().numpy())
    finally:
        model.train(was_training)

    samples = np.array(samples)
    mean_pred = np.mean(samples, axis=0)
    lower_bound = np.percentile(samples, 5, axis=0)
    upper_bound = np.percentile(samples, 95, axis=0)
    return mean_pred, lower_bound, upper_bound


def apply_counterfactual(
    model: nn.Module, nf, times, cov, adj, mask, covariate_idx: int = 0, delta: float = -0.1
) -> float:
    """Perturb one covariate (e.g. age, scaled by 100 in the source data — see
    implementation.md) by `delta` and return the average change in predicted
    regional volumes vs. the unperturbed mean prediction."""
    mean_pred, _, _ = generate_confidence_bands(model, nf, times, cov, adj, mask, n_samples=10)

    was_training = model.training
    model.eval()
    cov_cf = cov.clone()
    cov_cf[:, covariate_idx] += delta
    try:
        with torch.no_grad():
            recon_cf, _, _, _, _ = model(nf, times, cov_cf, adj, mask)
    finally:
        model.train(was_training)

    diff = (recon_cf.cpu().numpy() - mean_pred).mean()
    return float(diff)
