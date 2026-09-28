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
    import numpy as np
    renderer = TemplateRenderer()
    attr = np.array([0.1, 0.9, -0.2, 0.05])
    report = renderer.render(attr)
    assert "Node 1" in report  # index of the max value


def test_compute_node_attribution_end_to_end(synthetic_batch):
    """Exercises the full Captum IntegratedGradients path on a tiny model."""
    b = synthetic_batch
    model = STGraphNeuralODE(n_node_feat=1, hidden_dim=8, n_cov=b["covariates"].shape[-1])
    attr_sum, report = compute_node_attribution(
        model, b["node_features"], b["times"], b["covariates"], b["adj"], b["mask"], target_node_idx=0
    )
    assert attr_sum.shape == (b["node_features"].shape[2],)  # (N,)
    assert isinstance(report, str)
    assert "[CLINICIAN REPORT]" in report
