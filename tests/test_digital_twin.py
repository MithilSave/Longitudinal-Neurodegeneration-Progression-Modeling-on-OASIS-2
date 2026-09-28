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


def test_simulate_rollout_is_deterministic(synthetic_batch):
    """Regression test: simulate_rollout previously called the model directly,
    which samples eps ~ N(0,1) unconditionally in reparameterize() even under
    .eval() + no_grad(), making repeated calls with identical inputs diverge by
    up to ~0.47 on a ~[0,1]-scaled feature. It must now be a deterministic
    point-estimate rollout (decodes from the posterior mean)."""
    b = synthetic_batch
    model = _make_model(b)
    nf_init = b["node_features"][0:1, 0:1]

    r1 = simulate_rollout(model, nf_init, b["covariates"][0:1], b["adj"], steps=3)
    r2 = simulate_rollout(model, nf_init, b["covariates"][0:1], b["adj"], steps=3)

    assert torch.allclose(r1, r2), f"simulate_rollout is non-deterministic: max diff {(r1 - r2).abs().max().item()}"


def test_apply_counterfactual_signal_exceeds_noise_floor(synthetic_batch):
    """Regression test: apply_counterfactual previously diffed a 10-sample MC
    mean (generate_confidence_bands) against one independently-stochastic sample,
    so delta=0.0 (no real change) and delta=-0.1 (real change) produced diffs of
    comparable magnitude (~0.016 vs ~0.015 in the reviewer's measurement) — the
    "effect" was indistinguishable from sampling noise. Both sides must now go
    through the same deterministic decode so delta=0.0 collapses to (near) zero
    and a real delta produces a clean, larger signal."""
    b = synthetic_batch
    model = _make_model(b)

    diff_zero = apply_counterfactual(
        model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"],
        covariate_idx=0, delta=0.0,
    )
    diff_real = apply_counterfactual(
        model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"],
        covariate_idx=0, delta=-0.1,
    )

    assert abs(diff_zero) < 1e-5, f"delta=0.0 should produce ~zero diff, got {diff_zero}"
    assert diff_real != 0.0
    assert abs(diff_real) > abs(diff_zero) + 1e-4, (
        f"real counterfactual effect ({diff_real}) is not distinguishable from the "
        f"delta=0.0 noise floor ({diff_zero})"
    )
