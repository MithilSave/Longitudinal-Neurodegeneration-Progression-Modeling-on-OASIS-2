import numpy as np
import torch

from oasis2.digital_twin import simulate_rollout, generate_confidence_bands, apply_counterfactual
from oasis2.models.nd_vae import PhysicsConstrainedNDVAE


def _make_model(synthetic_batch):
    N = synthetic_batch["node_features"].shape[2]
    return PhysicsConstrainedNDVAE(n_regions=N, n_cov=synthetic_batch["covariates"].shape[-1], latent_dim=8, hidden_dim=16)


def test_simulate_rollout_produces_requested_steps(synthetic_batch):
    b = synthetic_batch
    model = _make_model(b)
    nf_init = b["node_features"][0:1, 0:1]  # single subject, single visit
    rollout = simulate_rollout(model, nf_init, b["covariates"][0:1], b["adj"], steps=3)
    assert rollout.shape[1] == 3  # 3 predicted future steps
    assert rollout.shape[0] == 1


def test_generate_confidence_bands_upper_bounds_lower(synthetic_batch):
    b = synthetic_batch
    model = _make_model(b)
    mean_pred, lower, upper = generate_confidence_bands(
        model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"], n_samples=5
    )
    assert (upper >= lower).all()
    assert mean_pred.shape == lower.shape == upper.shape


def test_generate_confidence_bands_restores_model_mode(synthetic_batch):
    """Regression test: the notebook version left the model in .train() mode."""
    b = synthetic_batch
    model = _make_model(b)
    model.eval()
    assert not model.training
    generate_confidence_bands(model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"], n_samples=3)
    assert not model.training, "model was left in train() mode after generate_confidence_bands"


def test_generate_confidence_bands_restores_mode_even_on_exception(synthetic_batch, monkeypatch):
    b = synthetic_batch
    model = _make_model(b)
    model.eval()

    original_forward = model.forward
    calls = {"n": 0}

    def flaky_forward(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated failure mid-sampling")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", flaky_forward)
    try:
        generate_confidence_bands(model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"], n_samples=5)
    except RuntimeError:
        pass
    assert not model.training, "model mode must be restored even if sampling raises"


def test_apply_counterfactual_changes_age_covariate(synthetic_batch):
    b = synthetic_batch
    model = _make_model(b)
    diff = apply_counterfactual(
        model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"],
        covariate_idx=0, delta=-0.1,
    )
    assert isinstance(diff, float)
