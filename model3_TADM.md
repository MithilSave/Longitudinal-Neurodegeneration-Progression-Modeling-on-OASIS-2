# Model 3: Temporally-Aware Trajectory Diffusion Model (TADM)

> **File:** `src/models/tadm_diffusion.py`
> **Checkpoint:** `models/tadm.pt`
> **Config:** `configs/tadm_diffusion.yaml`

---

## 1. What This Model Does

The **TADM** (Temporally-Aware Trajectory Diffusion Model) is a **score-based conditional diffusion model** that learns the full probability distribution of future brain states. Given a subject's baseline anatomy and clinical covariates, TADM generates probabilistic predictions of neurodegeneration at any future point in time — including realistic uncertainty estimates, not just single-point predictions.

### Core Idea: Denoising Diffusion Probabilistic Models (DDPM) Applied to Brains

TADM adapts the DDPM framework (originally used for image generation) to forecast neurodegeneration:

- **Standard DDPM:** learns to generate realistic images by learning to *reverse* a noise-adding process
- **TADM:** learns to generate realistic *future brain states* by learning to reverse noise conditioned on (baseline state, elapsed time, clinical covariates)

This gives TADM the ability to model **complex, multi-modal distributions** of disease trajectories — capturing the fact that different subjects can follow very different progression paths even from similar baselines.

---

## 2. Architecture Overview

`
TRAINING (Forward + Reverse Process)
======================================

Target state: X_k = X(t_k)       [brain state at future visit k]
Condition:    X(t0)               [baseline brain state]
              c                   [static covariates]
              delta_t_k = t_k - t0 [elapsed time in years]


FORWARD PROCESS (fixed, adds noise):
  X_k^(0) = X_k (clean)
  X_k^(1) --> X_k^(2) --> ... --> X_k^(T) ~ N(0, I)
  Each step adds Gaussian noise scaled by noise schedule alpha_bar_tau


CONDITIONING MODULE:
  +--------------------------------------------------+
  |          CONDITIONING ENCODER (e_cond)           |
  |  e_time = MLP_time(Sinusoidal(delta_t_k))        |
  |  e_graph = GCN(X(t0), L_hat)                    |
  |  e_cov   = MLP_cov(c)                            |
  |  e_cond  = e_time + e_graph + e_cov              |
  +--------------------------------------------------+


DENOISING BACKBONE (learnable):
  +--------------------------------------------------+
  |      SCORE NETWORK: epsilon_theta                |
  |  Input:  X_k^(tau), tau, e_cond                 |
  |  Output: predicted noise epsilon_hat             |
  |  Architecture: MLP / GNN with AdaGN/FiLM        |
  |                modulation by e_cond              |
  +--------------------------------------------------+


INFERENCE (Reverse Process, DDIM sampling):
  Start: X_k^(T) ~ N(0, I)
  Denoise: X_k^(T) --> X_k^(T-1) --> ... --> X_k^(0) = X_hat(t_k)
  Condition on e_cond at every denoising step
`

---

## 3. Mathematical Formulation

### Step 1 — Forward (Noising) Process

For a target future brain state X_k at elapsed time delta_t_k, a fixed Markov chain adds Gaussian noise over T diffusion steps:

`
q(X_k^(tau) | X_k) = N(sqrt(alpha_bar_tau) * X_k,  (1 - alpha_bar_tau) * I)
`

Where:
- `tau in {1, ..., T}` — discrete diffusion step (T typically = 1000)
- `alpha_bar_tau = prod_{s=1}^{tau} (1 - beta_s)` — cumulative noise schedule
- At tau=T, X_k^(T) approaches pure Gaussian noise N(0, I)

This forward process does NOT require training — it is a fixed mathematical recipe for corrupting data.

### Step 2 — Temporal Conditioning Mechanism

The conditioning embedding encodes everything the denoising network knows about the baseline state and elapsed time:

`
e_time  = MLP_time( Sinusoidal(delta_t_k) )    [temporal context]
e_graph = GCN( X(t0), L_hat )                  [baseline brain graph state]
e_cov   = MLP_cov( c )                         [static clinical covariates]

e_cond  = e_time + e_graph + e_cov             [combined conditioning signal]
`

- `Sinusoidal(delta_t_k)` — encodes elapsed time as a set of sine/cosine frequencies (same principle as positional encodings in Transformers)
- `GCN(X(t0), L_hat)` — graph convolutional network extracts spatial context from the baseline brain state on the connectome
- **AdaGN / FiLM modulation:** e_cond is injected into every layer of the denoising backbone via feature-wise linear modulation, so the model can adapt its denoising strategy based on the patient's baseline

### Step 3 — Denoising (Reverse) Process

The learnable score network epsilon_theta predicts the noise that was added at step tau, conditioned on the current noisy state and e_cond:

`
epsilon_hat = epsilon_theta( X_k^(tau), tau, e_cond )
`

This allows the clean signal to be estimated:

`
X_k^(0)_hat = ( X_k^(tau) - sqrt(1 - alpha_bar_tau) * epsilon_hat ) / sqrt(alpha_bar_tau)
`

### Step 4 — DDIM Accelerated Sampling (Inference)

At inference time, the full T-step reverse process is replaced by DDIM (Denoising Diffusion Implicit Models) which uses ~50 steps instead of 1000, with the same quality:

`
X_k^(tau-1) = sqrt(alpha_bar_{tau-1}) * X_k^(0)_hat
            + sqrt(1 - alpha_bar_{tau-1}) * epsilon_theta(X_k^(tau), tau, e_cond)
`

---

## 4. Loss Function

TADM is trained with a simple score-matching objective — predict the noise that was added:

`
L_TADM = E_{X_0, delta_t, X_k, tau, epsilon} [
    || epsilon  -  epsilon_theta( X_k^(tau), tau, e_cond ) ||_2^2
]
`

Where:
- `epsilon ~ N(0, I)` — the actual noise that was added during the forward process
- `epsilon_theta(...)` — the network's predicted noise
- The expectation is over random subjects, visit pairs, diffusion steps, and noise samples

**Why this works:** If the network perfectly predicts the noise at every diffusion step, the reverse process produces samples from the true posterior distribution of future brain states.

| Loss Component | Role |
|---------------|------|
| Noise MSE | Trains the denoising backbone to perfectly reverse the forward process |
| Conditioning via e_cond | Forces predictions to be consistent with baseline anatomy and elapsed time |
| No explicit physics term | Unlike ND-VAE, physics is implicitly learned from data patterns |

---

## 5. Step-by-Step Training Process

`
STEP 1 -- LOAD training batch
  |-- X(t0)     : baseline brain state [N, D]
  |-- X_k       : target future brain state at visit k [N, D]
  |-- delta_t_k : elapsed time from baseline to visit k (years)
  |-- c         : static covariates [C]
  +-- L_hat     : normalized graph Laplacian [N, N]

STEP 2 -- SAMPLE diffusion step
  +-- tau ~ Uniform{1, ..., T}   (random diffusion step)

STEP 3 -- ADD NOISE to target (forward process)
  |-- epsilon ~ N(0, I)           (random noise)
  +-- X_k^(tau) = sqrt(alpha_bar_tau) * X_k + sqrt(1 - alpha_bar_tau) * epsilon

STEP 4 -- BUILD conditioning embedding
  |-- e_time  = MLP_time(Sinusoidal(delta_t_k))
  |-- e_graph = GCN(X(t0), L_hat)
  |-- e_cov   = MLP_cov(c)
  +-- e_cond  = e_time + e_graph + e_cov

STEP 5 -- PREDICT noise with denoising backbone
  +-- epsilon_hat = epsilon_theta(X_k^(tau), tau, e_cond)

STEP 6 -- COMPUTE loss & UPDATE
  +-- L = || epsilon - epsilon_hat ||_2^2
  +-- Backpropagate through epsilon_theta, MLP_time, GCN, MLP_cov
`

---

## 6. Step-by-Step Inference (Generating Future Brain States)

`
STEP 1 -- ENCODE baseline condition
  |-- e_time  = MLP_time(Sinusoidal(delta_t*))   [delta_t* = desired future time]
  |-- e_graph = GCN(X(t0), L_hat)
  |-- e_cov   = MLP_cov(c)
  +-- e_cond  = e_time + e_graph + e_cov

STEP 2 -- INITIALIZE with pure noise
  +-- X^(T) ~ N(0, I)   [shape: N x D]

STEP 3 -- ITERATIVE DDIM DENOISING (50 steps)
  For tau = T, T-1, ..., 1:
    |-- epsilon_hat = epsilon_theta(X^(tau), tau, e_cond)
    |-- X^(0)_hat  = (X^(tau) - sqrt(1-alpha_bar_tau) * epsilon_hat) / sqrt(alpha_bar_tau)
    +-- X^(tau-1)  = sqrt(alpha_bar_{tau-1}) * X^(0)_hat
                   + sqrt(1 - alpha_bar_{tau-1}) * epsilon_hat

STEP 4 -- OUTPUT predicted future brain state
  +-- X_hat(t*) = X^(0)   [predicted regional volumes at time delta_t* from baseline]

STEP 5 -- UNCERTAINTY ESTIMATION
  +-- Repeat STEP 2-4 multiple times with different initial noise samples
  +-- Compute mean and std across samples --> confidence bands on regional atrophy
`

---

## 7. Key Advantages Over Models 1 & 2

| Feature | ST-GNN-ODE (M1) | ND-VAE (M2) | TADM (M3) |
|---------|----------------|-------------|-----------|
| Trajectory type | Single continuous ODE path | Probabilistic VAE | Full distribution via diffusion |
| Uncertainty | No (deterministic) | Approximate (Gaussian latent) | Rich (arbitrary distribution) |
| Multi-modal trajectories | Cannot model | Limited by Gaussian assumption | Yes — can model multiple distinct progression modes |
| Physics constraint | No | Yes (explicit regularizer) | Implicit (learned from data) |
| Sampling speed (inference) | Fast (ODE solve) | Fast (single forward pass) | Slower (DDIM, 50 steps) |
| Data requirements | Moderate | Moderate | Higher (benefits from more subjects) |

---

## 8. Key Design Choices & Rationale

| Decision | Rationale |
|----------|-----------|
| **DDPM framework** | State-of-the-art generative modeling; handles non-Gaussian, multi-modal distributions better than VAEs |
| **Sinusoidal temporal encoding** | Standard best practice for continuous time; distinguishes visit intervals without discretization |
| **GCN baseline encoder** | Injects spatial prior (which brain regions are already affected) into every denoising step |
| **AdaGN / FiLM modulation** | Conditioning at every layer of the backbone — stronger than only conditioning at input |
| **DDIM sampling** | Reduces inference from 1000 to 50 steps (20x speedup) with negligible quality loss |
| **Subject-level train/test split** | Prevents longitudinal data leakage — same discipline as Models 1 & 2 |

---

## 9. Dependencies

`
torch >= 2.2.0
torch-geometric >= 2.5.0      # GCN baseline encoder
numpy >= 1.24.0
`

No torchdiffeq needed — TADM uses its own discrete diffusion schedule.

---

## 10. Inputs & Outputs Summary

| | Shape | Description |
|--|-------|-------------|
| **Input** X(t0) | [N, D] | Baseline regional brain state |
| **Input** c | [C] | Static subject covariates |
| **Input** L_hat | [N, N] | Normalized graph Laplacian |
| **Input** delta_t* | scalar | Desired elapsed time for prediction (years) |
| **Output** X_hat(t*) | [N, D] | Predicted future brain state (mean sample) |
| **Output** Uncertainty | [N, D] | Std across multiple samples (confidence interval) |
| **Internal** e_cond | [E] | Conditioning embedding (time + graph + covariates) |
| **Internal** tau | scalar | Diffusion step (1 to T during training) |

---

## 11. References

- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
- Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models (DDIM).* ICLR.
- Dhariwal, P., & Nichol, A. (2021). *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS. [AdaGN]
- Perez, E., et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI.
- Marcus, D. S., et al. (2010). *OASIS-2: Longitudinal MRI data in nondemented and demented older adults.*
