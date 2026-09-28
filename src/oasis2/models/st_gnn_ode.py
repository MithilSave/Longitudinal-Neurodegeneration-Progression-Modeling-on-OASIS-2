"""Model 1: Continuous-Time Spatio-Temporal Graph Neural ODE (ST-GNN-ODE).

Ported verbatim from OASIS2_Colab.py (lines 1261-1474): `ODEFunc`,
`STGraphNeuralODE`, `train_st_gnn_ode`.

NOTE: the original script relies on a shared top-of-script `DEVICE` global
that `train_st_gnn_ode` calls `.to(DEVICE)` / `.to(DEVICE)` with. This module
defines its own module-scope `DEVICE = get_device()` (via
`oasis2.config.get_device`) to reproduce that behaviour without the global
script context.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from oasis2.config import get_device
from oasis2.logging_utils import setup_logging
from oasis2.models.layers import SinusoidalTimeEncoding, GraphConvLayer, GATLayer

logger = setup_logging("st_gnn_ode")
DEVICE = get_device()


class ODEFunc(nn.Module):
    """Neural vector field f_θ(h(t), A, t) for the Graph Neural ODE."""
    def __init__(self, hidden_dim, n_cov, time_dim=16):
        super().__init__()
        self.gat = GATLayer(hidden_dim, hidden_dim)
        self.time_enc = SinusoidalTimeEncoding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + n_cov + time_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.adj = None
        self.cov = None

    def set_context(self, adj, cov):
        self.adj = adj
        self.cov = cov

    def forward(self, t, h):
        # h: (B, N, D)
        B, N, D = h.shape
        # Graph attention
        g_out = self.gat(h, self.adj)

        # Time encoding
        t_enc = self.time_enc(t.expand(B))  # (B, time_dim)
        t_enc = t_enc.unsqueeze(1).expand(-1, N, -1)  # (B, N, time_dim)

        # Covariate
        cov_exp = self.cov.unsqueeze(1).expand(-1, N, -1)  # (B, N, C)

        # Concatenate and MLP
        combined = torch.cat([g_out, cov_exp, t_enc], dim=-1)
        return self.mlp(combined)


class STGraphNeuralODE(nn.Module):
    """
    Model 1: Continuous-Time Spatio-Temporal Graph Neural ODE.

    Encodes baseline state, integrates ODE forward to target times,
    decodes to predict future atrophy and clinical endpoints.
    """
    def __init__(self, n_node_feat=1, hidden_dim=32, n_cov=6, n_cdr_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_node_feat + n_cov, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ode_func = ODEFunc(hidden_dim, n_cov, time_dim=16)

        # Decoders
        self.decoder_recon = nn.Linear(hidden_dim, n_node_feat)
        self.decoder_mmse = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ELU(), nn.Linear(16, 1),
        )
        self.decoder_cdr = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ELU(), nn.Linear(16, n_cdr_classes),
        )

    def forward(self, node_features, times, covariates, adj, mask):
        """
        Args:
            node_features: (B, K, N, D)
            times: (B, K) — years from baseline
            covariates: (B, C)
            adj: (1, N, N)
            mask: (B, K)
        Returns:
            recon: (B, K, N, D), mmse_pred: (B, K), cdr_logits: (B, K, 4)
        """
        B, K, N, D = node_features.shape

        # Encode baseline (t=0)
        cov_exp = covariates.unsqueeze(1).expand(-1, N, -1)  # (B, N, C)
        x0 = torch.cat([node_features[:, 0], cov_exp], dim=-1)  # (B, N, D+C)
        h0 = self.encoder(x0)  # (B, N, hidden)

        # Set ODE context
        self.ode_func.set_context(adj, covariates)

        # Integrate ODE to each time point using Euler steps (fast)
        recon_list = []
        mmse_list = []
        cdr_list = []

        h = h0
        prev_t = torch.zeros(1, device=h.device)

        for ki in range(K):
            t_target = times[:, ki].mean()  # batch-averaged time
            dt = t_target - prev_t
            if dt > 0.001:
                # Euler integration with 5 steps
                n_steps = 5
                step = dt / n_steps
                for _ in range(n_steps):
                    h = h + step * self.ode_func(prev_t, h)
                    prev_t = prev_t + step
            prev_t = t_target

            # Decode
            recon_list.append(self.decoder_recon(h))
            # Global pool for clinical prediction
            h_global = h.mean(dim=1)  # (B, hidden)
            mmse_list.append(self.decoder_mmse(h_global).squeeze(-1))
            cdr_list.append(self.decoder_cdr(h_global))

        recon = torch.stack(recon_list, dim=1)  # (B, K, N, D)
        mmse_pred = torch.stack(mmse_list, dim=1)  # (B, K)
        cdr_logits = torch.stack(cdr_list, dim=1)  # (B, K, 4)

        return recon, mmse_pred, cdr_logits


def train_st_gnn_ode(train_loader, val_loader, n_epochs=50, lr=1e-3, patience=10):
    """Train ST-GNN-ODE model with early stopping."""
    # Get dimensions from first batch
    batch0 = next(iter(train_loader))
    n_feat = batch0['node_features'].shape[-1]
    n_cov = batch0['covariates'].shape[-1]

    model = STGraphNeuralODE(n_node_feat=n_feat, hidden_dim=32, n_cov=n_cov).to(DEVICE)
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
            mmse_true = batch['mmse'].to(DEVICE)
            cdr_true = batch['cdr'].to(DEVICE)

            recon, mmse_pred, cdr_logits = model(nf, t, cov, adj, msk)

            # Reconstruction loss (MSE on future visits)
            loss_recon = F.mse_loss(recon[msk], nf[msk])

            # MMSE loss
            loss_mmse = F.mse_loss(mmse_pred[msk], mmse_true[msk])

            # CDR loss (cross-entropy, skip masked)
            valid_cdr = msk & (cdr_true >= 0)
            if valid_cdr.any():
                loss_cdr = F.cross_entropy(
                    cdr_logits[valid_cdr], cdr_true[valid_cdr]
                )
            else:
                loss_cdr = torch.tensor(0.0, device=DEVICE)

            loss = loss_recon + 0.5 * loss_mmse + 0.3 * loss_cdr

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

                recon, mmse_pred, cdr_logits = model(nf, t, cov, adj, msk)
                loss_recon = F.mse_loss(recon[msk], nf[msk])
                loss_mmse = F.mse_loss(mmse_pred[msk], mmse_true[msk])
                valid_cdr = msk & (cdr_true >= 0)
                if valid_cdr.any():
                    loss_cdr = F.cross_entropy(cdr_logits[valid_cdr], cdr_true[valid_cdr])
                else:
                    loss_cdr = torch.tensor(0.0, device=DEVICE)
                val_loss += (loss_recon + 0.5 * loss_mmse + 0.3 * loss_cdr).item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)

        if epoch % 10 == 0:
            logger.info(f"[ST-GNN-ODE] Epoch {epoch}: train={avg_train:.4f}, val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"[ST-GNN-ODE] Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    logger.info(f"[ST-GNN-ODE] Training complete. Best val loss: {best_val_loss:.4f}")
    return model
