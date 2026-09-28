# Notebook-to-.py Pipeline Conversion & Digital Twin Fixes — Design

## 1. Background

The canonical pipeline lives in [OASIS2_Complete_Colab_Pipeline.ipynb](../../../OASIS2_Complete_Colab_Pipeline.ipynb) (192 cells, 149 code cells). A prior pass produced [OASIS2_Colab.py](../../../OASIS2_Colab.py), a 3549-line flat script with 68 top-level functions/classes extracted from the notebook, plus a `diff.txt` from that comparison.

Auditing the notebook's 73 top-level defs against the .py file shows exactly 5 are missing:

- `NodePredictor`, `SingleInputWrapper`, `TemplateRenderer` — Captum interpretability layer (notebook Phase 4/5)
- `simulate_rollout`, `generate_confidence_bands` — autoregressive trajectory rollout + Monte-Carlo confidence bands (notebook Phase 6)

These are the actual "digital twin" forward-prediction and explainability code (per [implementation.md](../../../implementation.md), the project's stated goal is predicting which regions will be affected next, with an interpretable explanation) — so today, none of that runs outside the notebook. This is the primary driver of "issues with the digital twin model."

No raw MRI scans are present locally (only demographics xlsx + template connectome — see repo `data/`), so Phase 1 (SynthSeg) cannot be exercised end-to-end. All DL-model and digital-twin testing will use synthetic tensors (small `N_REGIONS`, few synthetic subjects/visits) matching the shapes the real data pipeline produces.

## 2. Goals

1. Convert the full notebook pipeline into a proper multi-module Python package (`src/oasis2/`), completing the 5 missing pieces and verifying the other 68 against the notebook.
2. Make the package importable and testable outside Colab (isolate Colab-only bootstrap code).
3. Find and fix bugs in the digital twin model (ST-GNN-ODE, ND-VAE, TADM, NDM baseline, trajectory simulator, interpretability layer) using synthetic-data tests.
4. Leave both notebooks untouched — they remain Colab/local-runnable historical artifacts; the package becomes the canonical, tested implementation.

## 3. Non-goals

- Running the pipeline against real MRI data (not available locally).
- Changing the modeling approach/architecture described in `model1_ST_GNN_ODE.md` / `model2_ND_VAE.md` / `model3_TADM.md` — fixes stay within the existing design intent unless a bug makes that impossible.
- Rewriting the notebooks to import from the new package (out of scope; can be a follow-up).

## 4. Package layout

```
src/oasis2/
  __init__.py
  config.py              # PROJECT_ROOT, get_data_dir/get_outputs_dir/get_models_dir, device/env detection
  logging_utils.py        # setup_logging
  io_utils.py             # save/load checkpoint & json, z_score
  colab_setup.py          # Drive mount, pip installs — only imported/run under Colab
  data/
    ingest.py             # demographics loader, OASIS-2 dir walk, ANALYZE->NIfTI conversion, subject index
    synthseg.py            # SynthSeg wrapper, regional volume extraction, spot-check
    datasets.py            # CDR_MAP, LongitudinalGraphDataset, collate_longitudinal, build_dl_datasets
  connectome.py            # build_synthetic_connectome, load_template_connectome, compute_laplacian, eigen_decompose_laplacian
  ndm.py                   # ndm_propagate, compute_w_scores, fit_beta(s), predict_future_atrophy, validate_ndm, run_phase3
  models/
    layers.py              # SinusoidalTimeEncoding, GraphConvLayer, GATLayer
    st_gnn_ode.py           # ODEFunc, STGraphNeuralODE, train_st_gnn_ode
    nd_vae.py               # PhysicsConstrainedNDVAE, train_nd_vae
    tadm.py                 # cosine_beta_schedule, DiffusionConditioner, DenoisingNet, TemporalDiffusionModel, train_tadm
    benchmark.py            # evaluate_dl_model, run_benchmark
  deficits.py               # get_domain_weights, compute_deficit_scores(_all), validate_deficit_predictions, compute_longitudinal_deltas, run_phase4
  interpretability.py       # NEW: NodePredictor, SingleInputWrapper, TemplateRenderer
  digital_twin.py           # NEW: simulate_rollout, generate_confidence_bands, counterfactual helper
  visualization.py          # plotting functions currently inline at the tail of the notebook/script
  pipeline.py               # orchestrates phases 0-7 end to end
  cli.py                    # argparse entry point (`python -m oasis2 --phase ...`)
tests/
  test_connectome.py, test_ndm.py, test_datasets.py,
  test_st_gnn_ode.py, test_nd_vae.py, test_tadm.py,
  test_digital_twin.py, test_interpretability.py, test_benchmark.py
```

Module boundaries follow the notebook's own phase markers, so each file maps to a reviewable, independently testable unit.

## 5. Conversion approach

For each of the 68 already-ported functions: diff against the notebook source, move into its target module, fix any import/scoping issues from de-flattening (e.g. `DEVICE`, `N_REGIONS`, `PROJECT_ROOT` currently rely on module-level globals defined earlier in the same file — these become explicit `config` imports or function parameters).

For the 5 missing functions/classes: port from the notebook cells (already reviewed — see below), adapting global references (`m1`, `m2`, `test_ld`, `nf_in`, etc., which only exist because the notebook runs top-to-bottom) into explicit parameters.

## 6. Known issues found so far (to fix during implementation)

- `simulate_rollout` and `generate_confidence_bands` are entirely unported — the pipeline currently cannot produce a future-trajectory prediction or uncertainty band programmatically, only inside the notebook's own execution order.
- `generate_confidence_bands` calls `model.train()` to get stochastic VAE sampling but never calls `.eval()` afterward in the notebook version — needs a `try/finally` or context manager so it doesn't leave the model in train mode for subsequent calls.
- Full correctness audit of `run_phase3`, `run_phase4`, and the three training loops (`train_st_gnn_ode`, `train_nd_vae`, `train_tadm`) against the notebook has not been done yet — this happens during implementation via synthetic-data tests (shape checks, gradient flow, loss finiteness, and math sanity checks like NDM's closed-form `expm(-βLt)` vs. the ND-VAE physics-loss finite-difference residual sign).

## 7. Testing strategy

Synthetic fixtures: small connectome (`N_REGIONS=8`), a handful of synthetic subjects with 2-4 visits, random-but-plausible volumes/covariates. Tests per module cover: correct output shapes, no NaN/Inf, gradients flow (`loss.backward()` doesn't error and produces nonzero grads), and targeted math checks (e.g., NDM propagation matches direct matrix exponential at t=0 → identity; VAE reconstruction loss decreases over a few optimizer steps on a fixed synthetic batch). No test depends on real MRI data or SynthSeg.

## 8. Risks

- Notebook globals leaking into logic (e.g., code relying on cell-execution order) may hide additional bugs only surfaced once functions are called in isolation — expected, and is exactly what the test suite is meant to catch.
- Captum is an extra dependency (`pip install captum`) — will be added to a requirements file rather than a runtime `subprocess.run(['pip','install',...])` call.
