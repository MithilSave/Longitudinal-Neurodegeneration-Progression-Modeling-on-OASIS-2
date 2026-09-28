"""Interpretability layer for the digital twin's regional predictions.

Wraps a trained trajectory model (e.g. STGraphNeuralODE) so Captum's
IntegratedGradients can attribute a single node's predicted future value back
to the baseline input features. Ported from OASIS2_Complete_Colab_Pipeline.ipynb
cell 188 ("Phase 4/5: Interpretability Layer"), which previously only ran
inline inside the notebook against notebook-global variables (m1, test_ld,
nf_in, ...) and was never callable as a library function.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from oasis2.logging_utils import setup_logging

logger = setup_logging("interpretability")


class NodePredictor(nn.Module):
    """Wraps a trajectory model to output a single node's next-visit prediction."""

    def __init__(self, model: nn.Module, target_node_idx: int = 0):
        super().__init__()
        self.model = model
        self.target_node_idx = target_node_idx

    def forward(self, nf, times, cov, adj, mask):
        recon, _, _ = self.model(nf, times, cov, adj, mask)
        if recon.shape[1] > 1:
            return recon[:, 1, self.target_node_idx, 0]
        return recon[:, 0, self.target_node_idx, 0]


class SingleInputWrapper(nn.Module):
    """Exposes a model that takes (nf, times, cov, adj, mask) as a single-input
    nf -> scalar function, repeating the fixed context tensors to match
    Captum's internal batch expansion (IntegratedGradients replicates the
    input along dim 0 by `n_steps`)."""

    def __init__(self, model: nn.Module, t, cov, adj, msk):
        super().__init__()
        self.model = model
        self.t, self.cov, self.adj, self.msk = t, cov, adj, msk

    def forward(self, nf):
        B_in = self.t.shape[0]
        B_nf = nf.shape[0]
        if B_nf > B_in:
            M = B_nf // B_in
            t = self.t.repeat(M, 1)
            cov = self.cov.repeat(M, 1)
            msk = self.msk.repeat(M, 1)
            return self.model(nf, t, cov, self.adj, msk)
        return self.model(nf, self.t, self.cov, self.adj, self.msk)


class TemplateRenderer:
    """Turns a per-region attribution array into a one-line clinician-facing report."""

    def render(self, attr_array: np.ndarray) -> str:
        top_node = int(np.argmax(attr_array))
        top_val = attr_array[top_node]
        return (
            f"[CLINICIAN REPORT] The model's prediction of future atrophy is most "
            f"strongly driven by the baseline state of Node {top_node} "
            f"(Attribution Score: {top_val:.4f})."
        )


def compute_node_attribution(
    model: nn.Module, nf, times, cov, adj, mask, target_node_idx: int = 0, sample_idx: int = 0
) -> Tuple[np.ndarray, str, float]:
    """Run Integrated Gradients on `target_node_idx`'s prediction and render a report.

    `nf` (and the other batch tensors) may hold a full batch of B samples, but
    Integrated Gradients is only interpreted here for a single patient: the whole
    batch is run through IG (Captum needs it for its internal step expansion), but
    only `sample_idx`'s attribution is extracted and returned. This is a deliberate
    tradeoff of the current approach — computing the full batch is wasted work when
    only one sample's attribution is needed — not a bug to fix here.

    Returns (attr_sum, report, delta):
      - attr_sum has shape (N,), the baseline-visit attribution for every region
        for sample `sample_idx`.
      - report is a one-line clinician-facing summary string.
      - delta is Captum's convergence delta for `sample_idx`: how well
        sum(attributions) approximates f(input) - f(baseline) (the completeness
        axiom for Integrated Gradients). Values close to 0 indicate the IG
        approximation (governed by `n_steps`) is trustworthy; large |delta|
        signals the attribution may be unreliable.
    """
    from captum.attr import IntegratedGradients

    node_wrapper = NodePredictor(model, target_node_idx=target_node_idx)
    single_input = SingleInputWrapper(node_wrapper, times, cov, adj, mask)
    ig = IntegratedGradients(single_input)

    baseline = torch.zeros_like(nf)
    attributions, delta = ig.attribute(nf, baseline, return_convergence_delta=True)
    attr_sum = attributions[sample_idx, 0, :, 0].cpu().detach().numpy()
    delta_val = float(delta[sample_idx].cpu().detach().item())

    renderer = TemplateRenderer()
    report = renderer.render(attr_sum)
    logger.info(report)
    logger.info("Convergence delta for sample %d: %.6f", sample_idx, delta_val)
    return attr_sum, report, delta_val
