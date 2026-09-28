import numpy as np
import torch

from oasis2.interpretability import NodePredictor, SingleInputWrapper, TemplateRenderer, compute_node_attribution
from oasis2.models.st_gnn_ode import STGraphNeuralODE


def test_node_predictor_returns_scalar_per_sample(synthetic_batch):
    b = synthetic_batch
    model = STGraphNeuralODE(n_node_feat=1, hidden_dim=16, n_cov=b["covariates"].shape[-1])
    wrapper = NodePredictor(model, target_node_idx=0)
    out = wrapper(b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"])
    assert out.shape == (b["node_features"].shape[0],)


def test_single_input_wrapper_handles_captum_batch_expansion(synthetic_batch):
    b = synthetic_batch
    model = STGraphNeuralODE(n_node_feat=1, hidden_dim=16, n_cov=b["covariates"].shape[-1])
    node_wrapper = NodePredictor(model, target_node_idx=0)
    si = SingleInputWrapper(node_wrapper, b["times"], b["covariates"], b["adj"], b["mask"])
    # Captum's IntegratedGradients internally repeats the input batch n_steps times
    expanded_nf = b["node_features"].repeat(5, 1, 1, 1)
    out = si(expanded_nf)
    assert out.shape == (b["node_features"].shape[0] * 5,)


def test_template_renderer_names_the_top_node():
    renderer = TemplateRenderer()
    attr = np.array([0.1, 0.9, -0.2, 0.05])
    report = renderer.render(attr)
    assert "Node 1" in report  # index of the max value


def test_compute_node_attribution_end_to_end(synthetic_batch):
    """Exercises the full Captum IntegratedGradients path on a tiny model."""
    b = synthetic_batch
    model = STGraphNeuralODE(n_node_feat=1, hidden_dim=8, n_cov=b["covariates"].shape[-1])
    attr_sum, report, delta = compute_node_attribution(
        model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"], target_node_idx=0
    )
    assert attr_sum.shape == (b["node_features"].shape[2],)  # (N,)
    assert isinstance(report, str)
    assert "[CLINICIAN REPORT]" in report
    assert isinstance(delta, float)


def test_compute_node_attribution_sample_idx_selects_patient(synthetic_batch):
    """A non-default sample_idx should pull attributions/delta for that sample,
    not silently fall back to sample 0."""
    b = synthetic_batch
    model = STGraphNeuralODE(n_node_feat=1, hidden_dim=8, n_cov=b["covariates"].shape[-1])
    attr_sample0, _report0, _delta0 = compute_node_attribution(
        model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"],
        target_node_idx=0, sample_idx=0,
    )
    attr_sample2, _report2, _delta2 = compute_node_attribution(
        model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"],
        target_node_idx=0, sample_idx=2,
    )
    assert attr_sample0.shape == attr_sample2.shape == (b["node_features"].shape[2],)
    # Different input samples should (almost certainly) produce different attributions.
    assert not np.allclose(attr_sample0, attr_sample2)


def test_compute_node_attribution_convergence_delta_is_small(synthetic_batch):
    """Captum's convergence delta measures how well sum(attributions) approximates
    f(input) - f(baseline) (the IG completeness axiom). A degenerate/wrong
    computation, or too few IG steps, would show up as a large delta."""
    b = synthetic_batch
    model = STGraphNeuralODE(n_node_feat=1, hidden_dim=8, n_cov=b["covariates"].shape[-1])
    attr_sum, _report, delta = compute_node_attribution(
        model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"], target_node_idx=0
    )
    # Sanity check the attribution itself isn't degenerate before trusting delta.
    assert not np.all(attr_sum == 0)
    assert not np.any(np.isnan(attr_sum))
    # Empirically ~1e-3 to 1e-2 in magnitude for this synthetic fixture; 0.1 gives
    # headroom while still catching a badly broken IG computation.
    assert abs(delta) < 0.1
