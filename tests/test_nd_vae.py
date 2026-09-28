import torch

from oasis2.models.nd_vae import PhysicsConstrainedNDVAE, train_nd_vae


def test_forward_returns_five_tensors(synthetic_batch):
    b = synthetic_batch
    N = b["node_features"].shape[2]
    model = PhysicsConstrainedNDVAE(n_regions=N, n_node_feat=1, n_cov=b["covariates"].shape[-1], latent_dim=8, hidden_dim=16)
    recon, mmse_pred, cdr_logits, mu, logvar = model(
        b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"]
    )
    B, K, _, D = b["node_features"].shape
    assert recon.shape == (B, K, N, D)
    assert mu.shape == (B, 8) and logvar.shape == (B, 8)


def test_physics_loss_is_zero_for_single_visit(synthetic_batch):
    b = synthetic_batch
    N = b["node_features"].shape[2]
    model = PhysicsConstrainedNDVAE(n_regions=N, n_cov=b["covariates"].shape[-1], latent_dim=8, hidden_dim=16)
    recon = b["node_features"][:, :1]  # K=1
    loss = model.physics_loss(recon, b["times"][:, :1], b["adj"], b["mask"][:, :1])
    assert loss.item() == 0.0


def test_physics_loss_finite_and_nonnegative(synthetic_batch):
    b = synthetic_batch
    N = b["node_features"].shape[2]
    model = PhysicsConstrainedNDVAE(n_regions=N, n_cov=b["covariates"].shape[-1], latent_dim=8, hidden_dim=16)
    loss = model.physics_loss(b["node_features"], b["times"], b["adj"], b["mask"])
    assert loss.item() >= 0.0
    assert loss.item() == loss.item()  # not NaN


def test_train_nd_vae_runs(synthetic_batch):
    class ListLoader:
        def __init__(self, batch, n=3): self.batch, self.n = batch, n
        def __iter__(self): return iter([self.batch] * self.n)
        def __len__(self): return self.n

    loader = ListLoader(synthetic_batch)
    N = synthetic_batch["node_features"].shape[2]
    model = train_nd_vae(loader, loader, n_regions=N, n_epochs=2, lr=1e-3, patience=10)
    assert model is not None
