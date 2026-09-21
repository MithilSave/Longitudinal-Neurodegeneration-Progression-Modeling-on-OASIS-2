# Model 1: Continuous-Time Spatio-Temporal Graph Neural ODE (ST-GNN-ODE)

> **File:** `src/models/st_gnn_ode.py`
> **Checkpoint:** `models/st_gnn_ode.pt`
> **Config:** `configs/st_gnn_ode.yaml`

---

## 1. What This Model Does

The **ST-GNN-ODE** (Spatio-Temporal Graph Neural Ordinary Differential Equation) models the **continuous-time, non-linear propagation of neurodegeneration** across the brain's structural connectome.

### Core Problem It Solves

The legacy Network Diffusion Model (NDM, Raj et al. 2012) models pathology spread using a linear diffusion equation:

`
dx/dt = -beta * L * x
`

This has two critical limitations:

1. **It is linear** — real neurodegeneration involves non-linear, region-specific dynamics.
2. **It assumes fixed time steps** — OASIS-2 subjects have *irregular* inter-visit intervals (months to years apart), which linear models cannot natively handle.

**ST-GNN-ODE replaces this** with a learned neural vector field that operates continuously in time on the brain's graph topology, natively handling any delta_t between visits.

---

## 2. Architecture Overview

`
Input: X(t0) — baseline brain state (N regions x D features)
       c      — static covariates (age, sex, education, SES, eTIV, ASF)
       L_hat  — normalized graph Laplacian (N x N connectome)
       {t1, t2, ..., t_{K-1}} — irregular observation timestamps

            +-----------------------------------------------+
            |                  ENCODER                      |
            |   GATConv layers + Linear MLP                 |
            |   X(t0), c  -->  h(t0) in R^{NxH}            |
            +--------------------+---------------------------+
                                 |
            +--------------------v---------------------------+
            |        NEURAL ODE  (torchdiffeq)              |
            |   dh(t)/dt = f_theta(h(t), L_hat, t)         |
            |                                               |
            |   f_theta = GATConv(h(t), edges)             |
            |           + MLP(h(t) || c || sin(wt))        |
            |                                               |
            |   Solved via RK4 / dopri5 integrator         |
            |   from t0 --> t1 --> ... --> t_{K-1}         |
            +--------------------+---------------------------+
                                 |
            +--------------------v---------------------------+
            |                 DECODER                       |
            |   Recon:    h(tk) --> X_hat(tk)               |
            |   Clinical: h(tk) --> y_hat(tk)               |
            |             (MMSE score, CDR stage)           |
            +-----------------------------------------------+
`

---

## 3. Mathematical Formulation

### Step 1 — Encoder

The encoder maps the baseline brain state and static covariates into a latent node-feature matrix:

`
h(t0) = Encoder(X(t0), c)
`

- **Input:** X(t0) in R^{NxD} — regional atrophy/volume at baseline, c in R^C — static covariates
- **Output:** h(t0) in R^{NxH} — hidden node embeddings
- **Implementation:** 2-layer Graph Attention Network (GATConv) + linear projection

### Step 2 — Neural ODE Dynamics Function

The core of the model defines how the latent brain state evolves continuously over time:

`
dh(t)/dt = f_theta(h(t), L_hat, t)
         = GATConv(h(t), E)              [spatial:  WHERE pathology spreads]
         + MLP(h(t) || c || sin(wt))     [temporal: WHEN, over elapsed time ]
`

- **GATConv term** — captures *where* pathology propagates across connectome edges with learned attention weights
- **MLP term** — captures *when* — sin(wt) is a sinusoidal time encoding making the model aware of elapsed time
- **L_hat** — normalized graph Laplacian encoding the brain's structural connectivity

### Step 3 — ODE Integration

The ODE is numerically integrated from baseline to each target visit timestamp:

`
h(tk) = h(t0) + integral[t0 -> tk] f_theta(h(tau), L_hat, tau) d_tau
`

- **Solver:** Dormand-Prince (dopri5) adaptive step-size integrator, or fixed-step RK4
- **Library:** torchdiffeq with adjoint sensitivity method (memory-efficient backpropagation)
- **Handles irregular delta_t natively** — integrates to exactly tk, no interpolation or resampling needed

### Step 4 — Decoder

Two separate decoders operate on the latent state at each timestamp:

`
X_hat(tk)  = Decoder_recon   (h(tk))  -->  future regional volumes (per ROI)
y_hat(tk)  = Decoder_clinical (h(tk)) -->  MMSE score + CDR stage
`

---

## 4. Loss Function

The total training loss combines three terms summed over all future visits:

`
L_ODE = Sum_{k=1}^{K-1} [
    || X_hat(tk) - X(tk) ||^2                         [atrophy reconstruction MSE]
  + lambda_MMSE * (MMSE_hat_k - MMSE_k)^2             [cognitive score regression ]
  + lambda_CDR  * CrossEntropy(CDR_hat_k, CDR_k)      [dementia staging           ]
]
`

| Term | Type | Purpose |
|------|------|---------|
| Atrophy MSE | Regression | Predicts future regional brain volumes accurately |
| MMSE MSE | Regression | Projects cognitive test scores (0-30 scale) |
| CDR Cross-Entropy | Classification | Stages dementia severity (CDR: 0, 0.5, 1, 2) |

---

## 5. Step-by-Step Inference Process

`
STEP 1 -- LOAD subject data
  |-- X(t0) : regional volumes at baseline visit (normalized by eTIV)
  |-- c     : [age, sex, EDUC, SES, eTIV, ASF] -- static covariates
  |-- L_hat : precomputed normalized graph Laplacian from template connectome
  +-- {tk}  : elapsed years from baseline for each future visit

STEP 2 -- ENCODE baseline state
  +-- h(t0) = Encoder(X(t0), c)   shape: [N, H]

STEP 3 -- INTEGRATE ODE forward in time
  +-- For each tk in {t1, ..., t_{K-1}}:
        h(tk) = h(t0) + integral f_theta(h(tau), L_hat, tau) d_tau
        Solved by dopri5 adaptive integrator

STEP 4 -- DECODE predictions at each visit
  |-- X_hat(tk)  = Decoder_recon(h(tk))    future regional volumes
  +-- y_hat(tk)  = Decoder_clinical(h(tk)) MMSE + CDR predictions

STEP 5 -- EVALUATE (test set only)
  |-- Compute RMSE / MAE on regional atrophy
  |-- Compute Pearson r for spatial pattern correspondence
  |-- Compute Macro F1 / AUC on CDR staging
  +-- Compute MAE on MMSE projection
`

---

## 6. Key Design Choices & Rationale

| Decision | Rationale |
|----------|-----------|
| **Graph Attention (GAT) over GCN** | Learns which connectome edges matter most — not all connections are equally important for pathology propagation |
| **Neural ODE over discrete RNN** | Handles irregular visit intervals natively; no artificial discretization of continuous biological processes |
| **Adjoint sensitivity method** | Memory-efficient backpropagation through the ODE solver — enables training on long subject trajectories |
| **Sinusoidal time encoding** | Distinguishes 6-month from 3-year follow-ups without hand-engineering time features |
| **Dual decoder** | Jointly learns regional structure and clinical outcome — structural changes constrain and inform cognitive predictions |

---

## 7. Dependencies

`
torch >= 2.2.0
torch-geometric >= 2.5.0      # GATConv, graph utilities
torchdiffeq >= 0.2.3          # odeint, adjoint sensitivity method
numpy >= 1.24.0
`

---

## 8. Inputs & Outputs Summary

| | Shape | Description |
|--|-------|-------------|
| **Input** X(t0) | [N, D] | Baseline regional features (N = ROIs, D = feature dims) |
| **Input** c | [C] | Static subject covariates |
| **Input** L_hat | [N, N] | Normalized graph Laplacian |
| **Input** {tk} | [K-1] | Future visit timestamps (years from baseline) |
| **Output** X_hat(tk) | [K-1, N, D] | Predicted future regional states |
| **Output** MMSE | [K-1] | Predicted cognitive scores |
| **Output** CDR | [K-1, 4] | Predicted dementia stage logits |

---

## 9. References

- Chen, R. T. Q., et al. (2018). *Neural Ordinary Differential Equations.* NeurIPS.
- Velickovic, P., et al. (2018). *Graph Attention Networks.* ICLR.
- Raj, A., Kuceyeski, A., & Weiner, M. (2012). *A network diffusion model of disease progression in dementia.* Neuron, 73(6), 1204-1215.
- Marcus, D. S., et al. (2010). *OASIS-2: Longitudinal MRI data in nondemented and demented older adults.*
