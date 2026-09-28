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


def _deterministic_decode(model: nn.Module, nf: torch.Tensor, times: torch.Tensor, cov: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Decode using the posterior mean (mu) instead of a stochastic reparameterized
    sample, for a stable point-estimate prediction. Mirrors PhysicsConstrainedNDVAE.forward's
    per-timestep decode loop but skips reparameterize's random sampling, so calling this
    twice with identical inputs always returns identical output.

    `PhysicsConstrainedNDVAE.encode`/`decode` don't take an `adj` argument at all — the
    model's encoder is a GRU over flattened node features and its decoder is an MLP over
    (z, t, covariates); there is no graph convolution over `adj` in either. `adj` is only
    consumed by `physics_loss`, which this point-estimate path doesn't invoke, so `adj` is
    correctly omitted here.
    """
    mu, _logvar = model.encode(nf, cov, mask)
    B, K = times.shape
    recon_list = []
    for ki in range(K):
        x_hat, _m_hat, _c_hat = model.decode(mu, times[:, ki], cov)
        recon_list.append(x_hat)
    return torch.stack(recon_list, dim=1)


def simulate_rollout(model: nn.Module, nf_init: torch.Tensor, cov: torch.Tensor, adj: torch.Tensor, steps: int = 3) -> torch.Tensor:
    """Autoregressive forward rollout: repeatedly predicts the next visit and
    feeds it back in as input. `nf_init` is (B, 1, N, D) — a single baseline visit.
    Returns (B, steps, N, D).

    This is a deterministic point-estimate rollout: each step decodes from the
    posterior mean (mu) rather than a stochastic reparameterized sample, so calling
    this function twice with identical inputs always returns identical output — a
    caller building a report or dashboard from `.eval()` inference can rely on that.
    It intentionally does NOT report uncertainty; `generate_confidence_bands` is the
    sanctioned way to get the uncertainty band around a point estimate like this one.
    The two functions have a clear division of labor: `simulate_rollout` produces one
    stable trajectory, `generate_confidence_bands` produces the uncertainty around a
    given input.

    `adj` is accepted (and passed through unused) for interface stability: callers
    already pass it, and a future model swapped in here might need it even though
    `PhysicsConstrainedNDVAE`'s encode/decode do not use it.
    """
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

            recon = _deterministic_decode(model, nf_padded, t_padded, cov, msk_padded)
            next_state = recon[:, -1:, :, :]
            trajectories.append(next_state)
            current_nf = torch.cat([current_nf, next_state], dim=1)

    return torch.cat(trajectories, dim=1)


def generate_confidence_bands(
    model: nn.Module, nf, times, cov, adj, mask, n_samples: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte-Carlo sample the VAE's stochastic reparameterization n_samples times
    and return (mean, 5th percentile, 95th percentile) prediction arrays.

    The model is switched to train() mode as a defensive measure in case the
    underlying model has dropout/BatchNorm layers whose behavior should vary
    across MC samples. For the current PhysicsConstrainedNDVAE this has no
    effect: its only source of stochasticity is reparameterize()'s unconditional
    `eps ~ N(0,1)` sample, which fires regardless of `self.training`, and the
    model has no dropout or BatchNorm. The mode is still saved/restored via
    try/finally, so this function never leaks train-mode into the caller's
    model instance regardless of whether train() actually changes anything.
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
    regional volumes vs. the unperturbed baseline prediction.

    Both the baseline and counterfactual predictions are computed via the same
    deterministic posterior-mean decode path (`_deterministic_decode`), not via
    independent stochastic samples. Diffing two independently-sampled stochastic
    reconstructions (as an earlier version of this function did, comparing a
    10-sample MC mean from `generate_confidence_bands` against one fresh stochastic
    sample) buries a real counterfactual effect in sampling noise of comparable
    magnitude. Routing both sides through the same deterministic decode makes the
    reparameterization noise cancel out entirely, so `delta=0.0` yields a diff that
    is (up to floating point) exactly 0, and a real `delta` yields a diff that
    reflects the actual effect of the covariate perturbation.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            baseline_pred = _deterministic_decode(model, nf, times, cov, mask)

            cov_cf = cov.clone()
            cov_cf[:, covariate_idx] += delta
            recon_cf = _deterministic_decode(model, nf, times, cov_cf, mask)
    finally:
        model.train(was_training)

    diff = (recon_cf.cpu().numpy() - baseline_pred.cpu().numpy()).mean()
    return float(diff)
