import torch

from oasis2.data.datasets import LongitudinalGraphDataset, collate_longitudinal, build_dl_datasets


def test_build_dl_datasets_returns_three_loaders_or_none(synthetic_demo_df, synthetic_regional_volumes, synthetic_connectome):
    from oasis2 import ndm
    w_scores = ndm.compute_w_scores(synthetic_regional_volumes, synthetic_demo_df)
    train_loader, val_loader, test_loader = build_dl_datasets(
        demo_df=synthetic_demo_df,
        w_scores=w_scores,
        adj_matrix=synthetic_connectome,
        region_names=[f"region_{i}" for i in range(synthetic_connectome.shape[0])],
        seed=42,
    )
    assert train_loader is not None
    batch = next(iter(train_loader))
    assert batch["node_features"].dim() == 4  # (B, K, N, D)
    assert batch["adj"].shape[-2:] == synthetic_connectome.shape


def test_collate_longitudinal_pads_to_max_k(synthetic_connectome):
    N = synthetic_connectome.shape[0]
    adj = torch.tensor(synthetic_connectome, dtype=torch.float32).unsqueeze(0)
    items = [
        {
            "node_features": torch.rand(k, N, 1),
            "times": torch.arange(k, dtype=torch.float32),
            "covariates": torch.rand(6),
            "mmse": torch.rand(k) * 30,
            "cdr": torch.randint(0, 4, (k,)),
            "adj": adj,
        }
        for k in (2, 4)
    ]
    batch = collate_longitudinal(items)
    assert batch["node_features"].shape[1] == 4  # padded to max K
    assert batch["mask"].shape == (2, 4)
    assert batch["mask"][0].tolist() == [True, True, False, False]
