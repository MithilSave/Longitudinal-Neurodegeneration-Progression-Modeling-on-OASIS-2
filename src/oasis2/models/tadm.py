"""Model 3: TADM — Temporally-Aware Trajectory Diffusion Model.

Ported verbatim from OASIS2_Colab.py (lines 1725-2000): `cosine_beta_schedule`,
`DiffusionConditioner`, `DenoisingNet`, `TemporalDiffusionModel`, `train_tadm`.

NOTE: unlike Model 1 (ST-GNN-ODE) and Model 2 (ND-VAE), `TemporalDiffusionModel`
does NOT define a `forward()` method in the source script. It is a DDPM whose
public surface is `training_loss(...)` (used during training) and
`sample(...)` (used at inference, e.g. by `evaluate_dl_model`'s `TADM` branch
at OASIS2_Colab.py line ~2033). Calling the model directly as
`model(...)` would raise `NotImplementedError` from `nn.Module.forward`, in
both the original script and this port. This module preserves that behaviour
exactly rather than inventing a `forward()` the original never had.

NOTE: as with `st_gnn_ode.py` and `nd_vae.py`, the original script relies on a
shared top-of-script `DEVICE` global that `train_tadm` calls `.to(DEVICE)`
with. This module defines its own module-scope `DEVICE = get_device()` (via
`oasis2.config.get_device`) to reproduce that behaviour without the global
script context.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from oasis2.config import get_device
from oasis2.logging_utils import setup_logging
from oasis2.models.layers import SinusoidalTimeEncoding, GraphConvLayer

logger = setup_logging("tadm")
DEVICE = get_device()


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule for DDPM."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.999)


class DiffusionConditioner(nn.Module):
    """Conditioning module: baseline GCN + time encoding + covariate MLP."""
    def __init__(self, n_regions, n_node_feat, n_cov, cond_dim=32):
        super().__init__()
        # Baseline encoder via graph conv
        self.gcn = GraphConvLayer(n_node_feat, cond_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEncoding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )
        # Covariate MLP
        self.cov_mlp = nn.Sequential(
            nn.Linear(n_cov, cond_dim),
            nn.GELU(),
        )
        self.out_proj = nn.Linear(3 * cond_dim, cond_dim)

    def forward(self, x0, delta_t, covariates, adj):
        """
        x0: (B, N, D) baseline node features
        delta_t: (B,) elapsed time
        covariates: (B, C)
        adj: (1, N, N)
        """
        # Graph encoding of baseline
        g = self.gcn(x0, adj)  # (B, N, cond_dim)
        g = g.permute(0, 2, 1)  # (B, cond_dim, N)
        g = self.pool(g).squeeze(-1)  # (B, cond_dim)

        # Time encoding
        t_emb = self.time_mlp[0](delta_t)  # sinusoidal
        t_emb = self.time_mlp[1](t_emb)
        t_emb = self.time_mlp[2](t_emb)

        # Covariate encoding
        c_emb = self.cov_mlp(covariates)

        # Combine
        combined = torch.cat([g, t_emb, c_emb], dim=-1)
        return self.out_proj(combined)  # (B, cond_dim)


class DenoisingNet(nn.Module):
    """Simple MLP denoiser with FiLM conditioning."""
    def __init__(self, data_dim, cond_dim=32, hidden_dim=64):
        super().__init__()
        # Diffusion step embedding
        self.step_emb = nn.Embedding(200, cond_dim)

        # FiLM conditioning
        self.film_gamma = nn.Linear(2 * cond_dim, hidden_dim)
        self.film_beta = nn.Linear(2 * cond_dim, hidden_dim)

        # Denoiser backbone
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x_noisy, diffusion_step, cond):
        """
        x_noisy: (B, data_dim) — noised target state
        diffusion_step: (B,) long — diffusion timestep index
        cond: (B, cond_dim) — conditioning vector
        """
        step_emb = self.step_emb(diffusion_step)  # (B, cond_dim)
        cond_full = torch.cat([cond, step_emb], dim=-1)  # (B, 2*cond_dim)

        h = self.net(x_noisy)  # (B, hidden)

        # FiLM modulation
        gamma = self.film_gamma(cond_full)
        beta = self.film_beta(cond_full)
        h = gamma * h + beta

        return self.out(h)  # (B, data_dim) — predicted noise


class TemporalDiffusionModel(nn.Module):
    """
    Model 3: TADM — Temporally-Aware Trajectory Diffusion Model.

    DDPM conditioned on baseline state, elapsed time, and covariates.
    Generates future neurodegeneration states.
    """
    def __init__(self, n_regions, n_node_feat=1, n_cov=6, T=200, cond_dim=32):
        super().__init__()
        self.n_regions = n_regions
        self.data_dim = n_regions * n_node_feat
        self.T = T

        # Noise schedule
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        self.conditioner = DiffusionConditioner(n_regions, n_node_feat, n_cov, cond_dim)
        self.denoiser = DenoisingNet(self.data_dim, cond_dim, hidden_dim=64)

        # Clinical prediction heads (from denoised output)
        self.mmse_head = nn.Sequential(
            nn.Linear(self.data_dim, 16), nn.GELU(), nn.Linear(16, 1),
        )
        self.cdr_head = nn.Sequential(
            nn.Linear(self.data_dim, 16), nn.GELU(), nn.Linear(16, 4),
        )

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise at step t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def training_loss(self, node_features, times, covariates, adj, mask):
        """
        Compute DDPM training loss over all valid (baseline → future) pairs.
        """
        B, K, N, D = node_features.shape
        losses = []

        for ki in range(1, K):
            # Check which samples have valid data at this visit
            valid = mask[:, ki] & mask[:, 0]
            if not valid.any():
                continue

            x0_base = node_features[valid, 0]  # (B', N, D) baseline
            x_target = node_features[valid, ki].reshape(-1, self.data_dim)  # (B', data_dim) future
            dt = times[valid, ki] - times[valid, 0]  # (B',) elapsed time
            cov = covariates[valid]

            # Conditioning
            cond = self.conditioner(x0_base, dt, cov, adj)  # (B', cond_dim)

            # Random diffusion timestep
            B_valid = x_target.shape[0]
            t_diff = torch.randint(0, self.T, (B_valid,), device=x_target.device)

            # Forward diffusion
            x_noisy, noise = self.q_sample(x_target, t_diff)

            # Predict noise
            noise_pred = self.denoiser(x_noisy, t_diff, cond)

            losses.append(F.mse_loss(noise_pred, noise))

        if not losses:
            return torch.tensor(0.0, device=node_features.device)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample(self, x0_base, delta_t, covariates, adj, n_steps=10):
        """
        DDIM-style fast sampling from noise → predicted future state.
        """
        B = x0_base.shape[0]
        cond = self.conditioner(x0_base, delta_t, covariates, adj)

        # Start from pure noise
        x = torch.randn(B, self.data_dim, device=x0_base.device)

        # DDIM sampling with n_steps
        step_indices = torch.linspace(self.T - 1, 0, n_steps, dtype=torch.long, device=x.device)

        for i, t_idx in enumerate(step_indices):
            t_batch = t_idx.expand(B).long()
            noise_pred = self.denoiser(x, t_batch, cond)

            alpha_t = self.alphas_cumprod[t_idx]
            if i < len(step_indices) - 1:
                alpha_prev = self.alphas_cumprod[step_indices[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=x.device)

            # DDIM update
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * noise_pred

        return x.view(B, self.n_regions, -1)  # (B, N, D)


def train_tadm(train_loader, val_loader, n_regions, n_epochs=200, lr=1e-3, patience=15):
    """Train TADM diffusion model."""
    batch0 = next(iter(train_loader))
    n_feat = batch0['node_features'].shape[-1]
    n_cov = batch0['covariates'].shape[-1]

    model = TemporalDiffusionModel(
        n_regions=n_regions, n_node_feat=n_feat, n_cov=n_cov, T=200, cond_dim=32,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            nf = batch['node_features'].to(DEVICE)
            t = batch['times'].to(DEVICE)
            cov = batch['covariates'].to(DEVICE)
            adj = batch['adj'].to(DEVICE)
            msk = batch['mask'].to(DEVICE)

            loss = model.training_loss(nf, t, cov, adj, msk)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                nf = batch['node_features'].to(DEVICE)
                t = batch['times'].to(DEVICE)
                cov = batch['covariates'].to(DEVICE)
                adj = batch['adj'].to(DEVICE)
                msk = batch['mask'].to(DEVICE)
                val_loss += model.training_loss(nf, t, cov, adj, msk).item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)

        if epoch % 20 == 0:
            logger.info(f"[TADM] Epoch {epoch}: train={avg_train:.4f}, val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"[TADM] Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    logger.info(f"[TADM] Training complete. Best val loss: {best_val_loss:.4f}")
    return model
