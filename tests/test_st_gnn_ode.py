import torch

from oasis2.models.st_gnn_ode import STGraphNeuralODE, train_st_gnn_ode


def test_forward_shapes(synthetic_batch):
    b = synthetic_batch
    n_feat = b["node_features"].shape[-1]
    n_cov = b["covariates"].shape[-1]
    model = STGraphNeuralODE(n_node_feat=n_feat, hidden_dim=16, n_cov=n_cov)
    recon, mmse_pred, cdr_logits = model(
        b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"]
    )
    B, K, N, D = b["node_features"].shape
    assert recon.shape == (B, K, N, D)
    assert mmse_pred.shape == (B, K)
    assert cdr_logits.shape == (B, K, 4)


def test_gradients_flow(synthetic_batch):
    b = synthetic_batch
    model = STGraphNeuralODE(n_node_feat=1, hidden_dim=16, n_cov=b["covariates"].shape[-1])
    recon, mmse_pred, _ = model(b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"])
    loss = recon.pow(2).mean() + mmse_pred.pow(2).mean()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g == g for g in grad_norms)  # no NaNs


def test_train_st_gnn_ode_runs_for_a_few_epochs(synthetic_batch):
    from torch.utils.data import DataLoader, TensorDataset

    class ListLoader:
        """Minimal stand-in that yields the same synthetic batch dict repeatedly."""
        def __init__(self, batch, n_batches=3):
            self.batch, self.n = batch, n_batches
        def __iter__(self):
            return iter([self.batch] * self.n)
        def __len__(self):
            return self.n

    loader = ListLoader(synthetic_batch)
    model = train_st_gnn_ode(loader, loader, n_epochs=2, lr=1e-3, patience=10)
    assert model is not None
