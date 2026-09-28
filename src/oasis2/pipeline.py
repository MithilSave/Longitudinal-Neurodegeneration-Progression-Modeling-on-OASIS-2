"""End-to-end pipeline orchestration: phases 0/1 (data), 3 (NDM), 4 (deficits),
DL model training + benchmark, and the digital twin layer (trajectory rollout,
confidence bands, interpretability).

This replaces the linear script that previously lived at the tail of
OASIS2_Colab.py (script-level globals like demo_df, outputs_dir, beta_df) with
an explicit run_pipeline() function. It is also the first place the digital
twin (oasis2.digital_twin) and interpretability (oasis2.interpretability)
modules are actually wired into the rest of the pipeline — previously they
only existed as standalone notebook cells that were never called from the
main driver.

Output layout: phase 3 and phase 4 artifacts (w_scores.csv, fitted_betas.csv,
predicted_atrophy_5yr.csv, deficit_scores.csv, ...) are written under
`outputs_dir/phase3/` and `outputs_dir/phase4/` respectively, matching the
original notebook's directory layout. This is a deliberate choice: `run_phase3`
/ `run_phase4` themselves default to a *global* `outputs/phase3` /
`outputs/phase4` (via `config.get_outputs_dir`) when no `out_dir` is given, but
`run_pipeline` always passes `out_dir` explicitly as a subdirectory of its own
`outputs_dir` argument so a pipeline run's outputs stay together under one
directory instead of leaking into the global default location.
"""
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from oasis2.config import get_outputs_dir
from oasis2.connectome import load_template_connectome
from oasis2.data.datasets import build_dl_datasets
from oasis2.deficits import run_phase4
from oasis2.digital_twin import generate_confidence_bands, simulate_rollout
from oasis2.interpretability import compute_node_attribution
from oasis2.logging_utils import setup_logging
from oasis2.models.benchmark import run_benchmark
from oasis2.models.nd_vae import train_nd_vae
from oasis2.models.st_gnn_ode import train_st_gnn_ode
from oasis2.models.tadm import train_tadm
from oasis2.ndm import run_phase3

logger = setup_logging("pipeline")


def run_pipeline(
    demo_df: pd.DataFrame,
    regional_volumes: pd.DataFrame,
    outputs_dir: Optional[Path] = None,
    dl_epochs: int = 50,
) -> Dict:
    """Run the full OASIS-2 pipeline: NDM (phase 3), deficit mapping (phase 4),
    DL model training + benchmark, and the digital twin layer (rollout,
    confidence bands, node attribution).

    Returns a dict with keys: "phase3", "phase4", "benchmark", "digital_twin",
    "models".
    """
    if outputs_dir is None:
        outputs_dir = get_outputs_dir()
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running Phase 3 (NDM)...")
    p3_results = run_phase3(regional_volumes=regional_volumes, demo_df=demo_df, out_dir=outputs_dir / "phase3")
    w_scores = p3_results["w_scores"]

    logger.info("Running Phase 4 (deficit mapping)...")
    p4_results = run_phase4(ndm_results=p3_results, demo_df=demo_df, out_dir=outputs_dir / "phase4")

    adj_matrix, region_names = load_template_connectome(None)
    n_regions = adj_matrix.shape[0]
    train_loader, val_loader, test_loader = build_dl_datasets(
        demo_df=demo_df, w_scores=w_scores, adj_matrix=adj_matrix, region_names=region_names, seed=42,
    )

    st_gnn_ode_model = nd_vae_model = tadm_model = None
    benchmark_df = pd.DataFrame()
    digital_twin_result = {"rollout": None, "confidence_bands": None, "attribution": None}

    if train_loader is not None:
        logger.info("Training ST-GNN-ODE...")
        st_gnn_ode_model = train_st_gnn_ode(train_loader, val_loader, n_epochs=dl_epochs, lr=1e-3, patience=10)
        logger.info("Training ND-VAE...")
        nd_vae_model = train_nd_vae(train_loader, val_loader, n_regions, n_epochs=dl_epochs, lr=1e-3, patience=10)
        logger.info("Training TADM...")
        tadm_model = train_tadm(train_loader, val_loader, n_regions, n_epochs=dl_epochs, lr=1e-3, patience=15)

        logger.info("Running cross-model benchmark...")
        benchmark_df = run_benchmark(
            ndm_results=p3_results, st_gnn_ode_model=st_gnn_ode_model, nd_vae_model=nd_vae_model,
            tadm_model=tadm_model, test_loader=test_loader, n_regions=n_regions,
            out_dir=outputs_dir / "benchmark",
        )

        batch = next(iter(test_loader))
        if nd_vae_model is not None:
            logger.info("Simulating digital-twin rollout and confidence bands...")
            rollout = simulate_rollout(
                nd_vae_model, batch["node_features"][0:1, 0:1], batch["covariates"][0:1], batch["adj"], steps=3,
            )
            bands = generate_confidence_bands(
                nd_vae_model, batch["node_features"], batch["times"], batch["covariates"], batch["adj"], batch["mask"],
            )
            digital_twin_result["rollout"] = rollout
            digital_twin_result["confidence_bands"] = bands
        if st_gnn_ode_model is not None:
            logger.info("Computing node attribution (interpretability)...")
            attr_sum, report, delta = compute_node_attribution(
                st_gnn_ode_model, batch["node_features"], batch["times"], batch["covariates"],
                batch["adj"], batch["mask"], target_node_idx=0,
            )
            digital_twin_result["attribution"] = {"attr_sum": attr_sum, "report": report, "delta": delta}
    else:
        logger.warning(
            "No subjects with >= 2 visits found; skipping DL model training, benchmark, "
            "and the digital twin layer."
        )

    return {
        "phase3": p3_results,
        "phase4": p4_results,
        "benchmark": benchmark_df,
        "digital_twin": digital_twin_result,
        "models": {"st_gnn_ode": st_gnn_ode_model, "nd_vae": nd_vae_model, "tadm": tadm_model},
    }
