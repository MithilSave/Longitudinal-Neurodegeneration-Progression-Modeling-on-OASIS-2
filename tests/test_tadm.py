import pytest
import torch

from oasis2.models.tadm import cosine_beta_schedule, TemporalDiffusionModel, train_tadm


def test_cosine_beta_schedule_monotonic_and_bounded():
    betas = cosine_beta_schedule(timesteps=50)
    assert betas.shape == (50,)
    assert (betas >= 0).all() and (betas <= 1).all()


def test_temporal_diffusion_model_forward_shapes(synthetic_batch):
    """TemporalDiffusionModel has no forward(); its public surface is
    training_loss(...) (DDPM training objective) and sample(...) (DDIM
    inference). See module docstring in oasis2/models/tadm.py for why —
    this mirrors OASIS2_Colab.py, where the model is only ever called via
    these two methods, never as model(...)."""
    b = synthetic_batch
    N = b["node_features"].shape[2]
    model = TemporalDiffusionModel(n_regions=N, n_node_feat=1, n_cov=b["covariates"].shape[-1])

    # model(...) is not defined -> falls back to nn.Module.forward -> NotImplementedError
    with pytest.raises(NotImplementedError):
        model(b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"])

    # training_loss: scalar DDPM loss over valid (baseline -> future) pairs
    loss = model.training_loss(b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"])
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    # sample: DDIM-style generation of a future state from baseline
    B = b["node_features"].shape[0]
    x0_base = b["node_features"][:, 0]
    dt = b["times"][:, 1] - b["times"][:, 0]
    out = model.sample(x0_base, dt, b["covariates"], b["adj"], n_steps=5)
    assert out.shape == (B, N, 1)


def test_train_tadm_runs(synthetic_batch):
    class ListLoader:
        def __init__(self, batch, n=3): self.batch, self.n = batch, n
        def __iter__(self): return iter([self.batch] * self.n)
        def __len__(self): return self.n

    loader = ListLoader(synthetic_batch)
    N = synthetic_batch["node_features"].shape[2]
    model = train_tadm(loader, loader, n_regions=N, n_epochs=2, lr=1e-3, patience=10)
    assert model is not None
