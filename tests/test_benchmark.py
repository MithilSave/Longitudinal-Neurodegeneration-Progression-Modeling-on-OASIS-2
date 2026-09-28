"""Tests for oasis2.models.benchmark (cross-model benchmark runner)."""
from oasis2.models.benchmark import evaluate_dl_model, run_benchmark
from oasis2.models.st_gnn_ode import STGraphNeuralODE


class ListLoader:
    """Minimal DataLoader-like wrapper that replays a fixed batch `n` times."""

    def __init__(self, batch, n=2):
        self.batch, self.n = batch, n

    def __iter__(self):
        return iter([self.batch] * self.n)

    def __len__(self):
        return self.n


def test_evaluate_dl_model_returns_metrics_dict(synthetic_batch):
    b = synthetic_batch
    N = b["node_features"].shape[2]
    model = STGraphNeuralODE(n_node_feat=1, hidden_dim=16, n_cov=b["covariates"].shape[-1])

    metrics = evaluate_dl_model(model, ListLoader(b), "ST-GNN-ODE")
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_evaluate_dl_model_metrics_have_expected_keys(synthetic_batch):
    b = synthetic_batch
    model = STGraphNeuralODE(n_node_feat=1, hidden_dim=16, n_cov=b["covariates"].shape[-1])

    metrics = evaluate_dl_model(model, ListLoader(b), "ST-GNN-ODE")
    expected = {
        "atrophy_rmse", "atrophy_mae", "atrophy_pearson_r",
        "mmse_mae", "cdr_accuracy", "cdr_f1_macro",
    }
    assert set(metrics.keys()) <= expected
    assert "atrophy_rmse" in metrics
    assert "mmse_mae" in metrics


def test_run_benchmark_produces_dataframe(tmp_path, synthetic_batch):
    b = synthetic_batch
    model = STGraphNeuralODE(n_node_feat=1, hidden_dim=16, n_cov=b["covariates"].shape[-1])

    ndm_results = {"validation": {"rmse": 0.1, "mae": 0.05, "pearson_r": 0.9}}
    df = run_benchmark(
        ndm_results=ndm_results,
        st_gnn_ode_model=model,
        nd_vae_model=None,
        tadm_model=None,
        test_loader=ListLoader(b),
        n_regions=b["node_features"].shape[2],
        out_dir=tmp_path,
    )

    assert "NDM (Baseline)" in df.index
    assert "ST-GNN-ODE" in df.index
    assert (tmp_path / "benchmark_results.csv").exists()
