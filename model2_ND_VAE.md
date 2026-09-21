# Model 2: Physics-Constrained Network Diffusion VAE (ND-VAE)

> **File:** `src/models/nd_vae.py`
> **Checkpoint:** `models/nd_vae.pt`
> **Config:** `configs/nd_vae.yaml`

---

## 1. What This Model Does

The **ND-VAE** (Network Diffusion Variational Autoencoder) is a **deep generative model** that bridges biophysical first principles with modern deep representation learning. It learns a probabilistic latent space over longitudinal neurodegeneration trajectories, while simultaneously enforcing that the learned dynamics remain **physically plausible** via a connectome diffusion regularizer.

### Core Problem It Solves

Pure deep learning models (e.g., vanilla VAEs or LSTMs) can produce predictions that violate known brain physics — such as pathology spreading to anatomically disconnected regions. Conversely, the classic NDM is too rigid: it forces all dynamics to follow a single linear diffusion equation.

**ND-VAE combines both worlds:**
- A flexible **encoder** that maps noisy, longitudinal MRI biomarker trajectories to a compact probabilistic latent code
- A **physics-informed decoder** that generates future atrophy trajectories while being penalized whenever it deviates from connectome-governed diffusion dynamics

---

## 2. Architecture Overview

`
Longitudinal data: X(t0), X(t1), ..., X(t_{K-1})
Static covariates: c
Visit timestamps:  t0, t1, ..., t_{K-1}

+--------------------------------------------------+
|               ENCODER  (q_phi)                  |
|  Temporal GRU / Transformer over visits         |
|  [X(t0:K-1), t, c]  -->  mu_phi, log_sigma_phi  |
+---------------------+----------------------------+
                       |   Reparameterization trick
                       |   z ~ N(mu_phi, sigma^2_phi)
                       v
+--------------------------------------------------+
|           TRAJECTORY DECODER  (g_psi)           |
|   For any query time t:                         |
|   X_hat(t) = g_psi(z, t, c)                    |
|   (MLP conditioned on latent code + timestamp) |
+---------------------+----------------------------+
                       |
         +-------------+-------------+
         |                           |
+--------v--------+       +----------v---------+
|  RECONSTRUCTION |       | PHYSICS REGULARIZER|
|   LOSS (MAE)    |       | dX_hat/dt must obey|
|  X_hat(tk) vs   |       | connectome diffusion|
|  X(tk)          |       | equation (auto-diff)|
+-----------------+       +--------------------+
`

---

## 3. Mathematical Formulation

### Step 1 — Probabilistic Encoder

The encoder defines a variational posterior over a latent code z given the full longitudinal trajectory:

`
q_phi(z | X_{0:K-1}, t, c) = N(mu_phi, sigma^2_phi)
`

- **Input:** All K visit brain states X(t0)...X(t_{K-1}), timestamps t, static covariates c
- **Output:** Gaussian distribution parameters (mu_phi, sigma^2_phi) in latent space R^Z
- **Implementation:** Temporal GRU (or Transformer) that aggregates all visits, then projects to mean and log-variance

### Step 2 — Reparameterization & Sampling

The latent code is sampled using the reparameterization trick (enables gradient flow):

`
epsilon ~ N(0, I)
z = mu_phi + sigma_phi * epsilon
`

This allows the model to be trained end-to-end with backpropagation even though z is stochastic.

### Step 3 — Generative Trajectory Decoder

Given the latent code, the decoder generates brain states at any queried timestamp t:

`
X_hat(t) = g_psi(z, t, c)
`

- **Input:** z (latent code), t (scalar timestamp in years), c (static covariates)
- **Output:** X_hat(t) in R^{NxD} — predicted regional brain state at time t
- **Implementation:** MLP with sinusoidal time encoding; z and c are concatenated as conditioning

### Step 4 — Biophysical Regularizer (Key Innovation)

This is the critical physics-informed component. It penalizes the decoder whenever its generated trajectory deviates from the connectome's diffusion dynamics:

`
R_physics(X_hat, t) = || dX_hat(t)/dt  +  beta_diff * L_hat * X_hat(t) ||_F^2
`

Where:
- `dX_hat(t)/dt` — temporal derivative computed via **automatic differentiation** (torch.autograd) with respect to scalar input t
- `beta_diff` — a learned (or fixed) diffusion coefficient
- `L_hat` — normalized graph Laplacian from the structural connectome
- `|| . ||_F^2` — Frobenius norm (sum of squared elements)

**Interpretation:** If X_hat(t) perfectly obeys connectome diffusion physics, R_physics = 0. Any deviation from the diffusion equation is penalized, pushing the model toward physically realistic trajectories while still allowing learned non-linearities.

---

## 4. Loss Function

The full training objective has three components:

`
L_ND-VAE = E_{q_phi} [ Sum_{k=0}^{K-1} || X_hat(tk) - X(tk) ||_1 ]    [Reconstruction: MAE over all visits]
         + beta_KL * D_KL( q_phi(z|.) || N(0,I) )                       [KL divergence: regularize latent space]
         + lambda_phys * integral[0 -> t_max] R_physics(X_hat, tau) dtau [Physics: enforce diffusion consistency]
`

| Term | Type | Purpose |
|------|------|---------|
| Reconstruction MAE | Regression | Accurately predicts future regional volumes at each visit |
| KL Divergence | Regularization | Keeps the latent space smooth and interpretable (prevents posterior collapse) |
| Physics Integral | Physics constraint | Prevents predictions that violate connectome-governed diffusion dynamics |

### Hyperparameter Roles

`
beta_KL     : controls how tightly z must match N(0,I)
              (high beta_KL = smooth latent space; low = more expressive encoder)
lambda_phys : controls how strictly physics is enforced
              (high = closer to classic NDM; low = more freely learned dynamics)
`

---

## 5. Step-by-Step Training Process

`
STEP 1 -- LOAD batch of subjects
  |-- All visit brain states: {X(t0), X(t1), ..., X(t_{K-1})}
  |-- Timestamps: {t0, t1, ..., t_{K-1}} (years from baseline)
  |-- Static covariates: c
  +-- Graph Laplacian: L_hat

STEP 2 -- ENCODE trajectory --> latent distribution
  +-- (mu_phi, log_sigma_phi) = Encoder(X_{0:K-1}, t, c)

STEP 3 -- SAMPLE latent code z
  +-- z = mu_phi + sigma_phi * epsilon  (epsilon ~ N(0,I))

STEP 4 -- DECODE at all observed timestamps
  +-- For each tk: X_hat(tk) = g_psi(z, tk, c)

STEP 5 -- COMPUTE physics regularizer
  |-- For each tk: compute dX_hat/dt via autograd (d/d(tk))
  +-- R_physics(tk) = || dX_hat(tk)/dt + beta_diff * L_hat * X_hat(tk) ||_F^2

STEP 6 -- COMPUTE total loss
  +-- L = MAE(X_hat, X) + beta_KL * KL(q||p) + lambda_phys * mean(R_physics)

STEP 7 -- BACKPROPAGATE and UPDATE parameters
  +-- Gradients flow through: Decoder, physics regularizer, Encoder

STEP 8 -- INFERENCE (new subject, future visit)
  +-- Encode observed visits --> z
  +-- Query decoder at any future t*: X_hat(t*) = g_psi(z, t*, c)
  +-- Uncertainty estimate: run encoder multiple times (dropout) or sample z ~ q_phi
`

---

## 6. Why a VAE? Probabilistic Interpretation

Unlike Model 1 (which produces a single trajectory), ND-VAE produces a **distribution over trajectories**. This has practical advantages:

| Capability | Description |
|-----------|-------------|
| **Uncertainty quantification** | Sample multiple z vectors from q_phi to produce trajectory confidence bands |
| **Generative sampling** | Sample z ~ N(0,I) to synthesize plausible neurodegeneration progressions |
| **Latent space interpolation** | Interpolate between two subjects' z codes to explore intermediate phenotypes |
| **Anomaly detection** | High reconstruction error + large KL divergence flags atypical subjects |

---

## 7. Key Design Choices & Rationale

| Decision | Rationale |
|----------|-----------|
| **VAE over deterministic AE** | Enables probabilistic uncertainty quantification over future trajectories |
| **MAE (L1) over MSE (L2)** | More robust to outlier visits (neurodegeneration data has noisy measurements) |
| **Autograd-based physics loss** | Computes exact temporal derivatives of the decoder — no finite-difference approximation |
| **Integrated physics penalty** | Enforces physical consistency across the continuous time domain, not just at observed visits |
| **KL annealing (optional)** | Gradually increasing beta_KL during training prevents posterior collapse in early epochs |

---

## 8. Dependencies

`
torch >= 2.2.0
torch-geometric >= 2.5.0      # Graph Laplacian utilities
numpy >= 1.24.0
scipy >= 1.11.0               # Optional: matrix operations
`

---

## 9. Inputs & Outputs Summary

| | Shape | Description |
|--|-------|-------------|
| **Input** X_{0:K-1} | [K, N, D] | All visit brain states |
| **Input** t | [K] | Visit timestamps (years from baseline) |
| **Input** c | [C] | Static subject covariates |
| **Input** L_hat | [N, N] | Normalized graph Laplacian |
| **Output** mu_phi | [Z] | Latent mean (encodes disease trajectory shape) |
| **Output** sigma_phi | [Z] | Latent std (encodes trajectory uncertainty) |
| **Output** X_hat(t*) | [N, D] | Predicted brain state at any queried time t* |
| **Output** R_physics | scalar | Physics consistency violation score |

---

## 10. References

- Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-Informed Neural Networks.* Journal of Computational Physics.
- Raj, A., et al. (2015). *Network diffusion model of progression predicts longitudinal patterns of atrophy and metabolism in Alzheimer's disease.* Cell Reports, 10(3), 359-369.
- Marcus, D. S., et al. (2010). *OASIS-2: Longitudinal MRI data in nondemented and demented older adults.*
