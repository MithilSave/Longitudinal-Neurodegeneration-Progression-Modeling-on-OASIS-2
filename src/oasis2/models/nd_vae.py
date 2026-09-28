"""Model 2: Physics-Constrained ND-VAE.

Ported verbatim from OASIS2_Colab.py (lines 1481-1718): `PhysicsConstrainedNDVAE`
(with `encode`, `reparameterize`, `decode`, `forward`, `physics_loss` methods),
`train_nd_vae`.

NOTE: `forward()` returns a 5-tuple `(recon, mmse_pred, cdr_logits, mu, logvar)`.
The later digital-twin trajectory simulator (Task 12) unpacks exactly this
5-tuple when calling the model, so the signature must not change.

NOTE: as with `st_gnn_ode.py`, the original script relies on a shared
top-of-script `DEVICE` global that `train_nd_vae` calls `.to(DEVICE)` with.
This module defines its own module-scope `DEVICE = get_device()` (via
`oasis2.config.get_device`) to reproduce that behaviour without the global
script context.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from oasis2.config import get_device
from oasis2.logging_utils import setup_logging

logger = setup_logging("nd_vae")
DEVICE = get_device()


class PhysicsConstrainedNDVAE(nn.Module):
    """
    Model 2: ND-VAE with biophysical regularization.

    Encoder maps longitudinal trajectories to latent z.
    Decoder generates trajectory conditioned on (z, t, c).
    Physics loss penalizes deviation from network diffusion dynamics.
    """
    def __init__(self, n_regions, n_node_feat=1, n_cov=6, latent_dim=16, hidden_dim=32):
        super().__init__()
        self.n_regions = n_regions
        self.latent_dim = latent_dim

        # Encoder: GRU over visits → latent
        self.enc_proj = nn.Linear(n_regions * n_node_feat + n_cov, hidden_dim)
        self.enc_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: (z, t, c) → X(t)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + 1 + n_cov, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, n_regions * n_node_feat),
        )

        # Clinical decoders
        self.dec_mmse = nn.Sequential(
            nn.Linear(latent_dim + 1 + n_cov, 16), nn.ELU(), nn.Linear(16, 1),
        )
        self.dec_cdr = nn.Sequential(
            nn.Linear(latent_dim + 1 + n_cov, 16), nn.ELU(), nn.Linear(16, 4),
        )

        # Learnable diffusion rate
        self.beta_diff = nn.Parameter(torch.tensor(0.1))

    def encode(self, node_features, covariates, mask):
        """Encode longitudinal trajectory to latent distribution."""
        B, K, N, D = node_features.shape
        # Flatten node features per visit
        x_flat = node_features.view(B, K, N * D)  # (B, K, N*D)
        cov_exp = covariates.unsqueeze(1).expand(-1, K, -1)
        inp = torch.cat([x_flat, cov_exp], dim=-1)
        inp = F.elu(self.enc_proj(inp))

        # Pack for GRU (handle variable lengths via mask)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            inp, lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.enc_gru(packed)
        h = h.squeeze(0)  # (B, hidden)

        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, t, covariates):
        """
        Decode latent z at time t with covariates.
        t: (B,) scalar time
        """
        inp = torch.cat([z, t.unsqueeze(-1), covariates], dim=-1)
        x_hat = self.dec(inp).view(-1, self.n_regions, 1)  # (B, N, D)
        mmse_hat = self.dec_mmse(inp).squeeze(-1)  # (B,)
        cdr_logits = self.dec_cdr(inp)  # (B, 4)
        return x_hat, mmse_hat, cdr_logits

    def forward(self, node_features, times, covariates, adj, mask):
        B, K, N, D = node_features.shape

        mu, logvar = self.encode(node_features, covariates, mask)
        z = self.reparameterize(mu, logvar)

        # Decode at each time point
        recon_list, mmse_list, cdr_list = [], [], []
        for ki in range(K):
            t_k = times[:, ki]
            x_hat, m_hat, c_hat = self.decode(z, t_k, covariates)
            recon_list.append(x_hat)
            mmse_list.append(m_hat)
            cdr_list.append(c_hat)

        recon = torch.stack(recon_list, dim=1)  # (B, K, N, D)
        mmse_pred = torch.stack(mmse_list, dim=1)  # (B, K)
        cdr_logits = torch.stack(cdr_list, dim=1)  # (B, K, 4)

        return recon, mmse_pred, cdr_logits, mu, logvar

    def physics_loss(self, recon, times, adj, mask):
        """
        Biophysical regularizer: ||dX/dt + β * L_hat * X||²_F
        Approximated via finite differences.
        """
        B, K, N, D = recon.shape
        if K < 2:
            return torch.tensor(0.0, device=recon.device)

        # Compute Laplacian L = I - A_hat (adj is already normalized)
        if adj.dim() == 3:
            adj_2d = adj[0]
        else:
            adj_2d = adj
        L = torch.eye(N, device=recon.device) - adj_2d

        total = torch.tensor(0.0, device=recon.device)
        count = 0
        for ki in range(K - 1):
            dt = (times[:, ki + 1] - times[:, ki]).clamp(min=0.01)  # (B,)
            dX = recon[:, ki + 1] - recon[:, ki]  # (B, N, D)
            dXdt = dX / dt.unsqueeze(-1).unsqueeze(-1)

            # L * X
            LX = torch.matmul(L.unsqueeze(0), recon[:, ki])  # (B, N, D)

            # Physics residual
            residual = dXdt + self.beta_diff * LX
            both_valid = mask[:, ki] & mask[:, ki + 1]
            if both_valid.any():
                total = total + residual[both_valid].pow(2).mean()
                count += 1

        return total / max(count, 1)


def train_nd_vae(train_loader, val_loader, n_regions, n_epochs=100, lr=1e-3, patience=10):
    """Train ND-VAE with β-annealing and physics loss."""
    batch0 = next(iter(train_loader))
    n_feat = batch0['node_features'].shape[-1]
    n_cov = batch0['covariates'].shape[-1]

    model = PhysicsConstrainedNDVAE(
        n_regions=n_regions, n_node_feat=n_feat, n_cov=n_cov,
        latent_dim=16, hidden_dim=32,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        # β-annealing: linearly increase KL weight over first 20 epochs
        beta_kl = min(1.0, epoch / 20.0) * 0.1
        lambda_phys = 0.01

        train_loss = 0.0
        for batch in train_loader:
            nf = batch['node_features'].to(DEVICE)
            t = batch['times'].to(DEVICE)
            cov = batch['covariates'].to(DEVICE)
            adj = batch['adj'].to(DEVICE)
            msk = batch['mask'].to(DEVICE)
            mmse_true = batch['mmse'].to(DEVICE)
            cdr_true = batch['cdr'].to(DEVICE)

            recon, mmse_pred, cdr_logits, mu, logvar = model(nf, t, cov, adj, msk)

            # Reconstruction L1
            loss_recon = F.l1_loss(recon[msk], nf[msk])

            # KL divergence
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Physics loss
            loss_phys = model.physics_loss(recon, t, adj, msk)

            # MMSE + CDR
            loss_mmse = F.mse_loss(mmse_pred[msk], mmse_true[msk])
            valid_cdr = msk & (cdr_true >= 0)
            if valid_cdr.any():
                loss_cdr = F.cross_entropy(cdr_logits[valid_cdr], cdr_true[valid_cdr])
            else:
                loss_cdr = torch.tensor(0.0, device=DEVICE)

            loss = loss_recon + beta_kl * kl + lambda_phys * loss_phys + 0.5 * loss_mmse + 0.3 * loss_cdr

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
                mmse_true = batch['mmse'].to(DEVICE)
                cdr_true = batch['cdr'].to(DEVICE)

                recon, mmse_pred, cdr_logits, mu, logvar = model(nf, t, cov, adj, msk)
                loss_recon = F.l1_loss(recon[msk], nf[msk])
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss_mmse = F.mse_loss(mmse_pred[msk], mmse_true[msk])
                valid_cdr = msk & (cdr_true >= 0)
                if valid_cdr.any():
                    loss_cdr = F.cross_entropy(cdr_logits[valid_cdr], cdr_true[valid_cdr])
                else:
                    loss_cdr = torch.tensor(0.0, device=DEVICE)
                val_loss += (loss_recon + 0.1 * kl + 0.5 * loss_mmse + 0.3 * loss_cdr).item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)

        if epoch % 10 == 0:
            logger.info(f"[ND-VAE] Epoch {epoch}: train={avg_train:.4f}, val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"[ND-VAE] Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    logger.info(f"[ND-VAE] Training complete. Best val loss: {best_val_loss:.4f}")
    return model
