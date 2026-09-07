# Master Model Implementation Plan: Post-2022 Deep Learning Progression Models

## 1\. Executive Summary & Objective

This specification outlines the architectural transition from the legacy **Network Diffusion Model (NDM)** (Raj et al., 2012\) to **three post-2022 Deep Learning architectures** for longitudinal neurodegeneration modeling on the **OASIS-2** cohort and structural connectome data:

1. **Model 1 (Continuous-Time Graph Neural ODE / ST-GNN-ODE)**: Models continuous-time non-linear pathology propagation across connectome topologies, natively handling irregular inter-visit intervals ($\\Delta t$).  
2. **Model 2 (Physics-Constrained Network Diffusion VAE \- ND-VAE)**: A deep generative variational framework that regularizes latent space dynamics using physical graph diffusion priors.  
3. **Model 3 (Temporally-Aware Latent Progression Diffusion Model \- TADM)**: A score-based conditional diffusion model that forecasts future neurodegeneration states conditioned on baseline anatomy and continuous elapsed time.

---

## 2\. Target Directory & Module Structure

Antigravity should construct and organize the repository under the following modular layout:

longitudinal-neurodegeneration-progression-modeling-on-oasis-2/

├── configs/

│   ├── config.yaml                     \# Global configurations (paths, seed, device)

│   ├── st\_gnn\_ode.yaml                 \# Hyperparameters for Model 1

│   ├── nd\_vae.yaml                     \# Hyperparameters for Model 2

│   └── tadm\_diffusion.yaml             \# Hyperparameters for Model 3

├── data/

│   ├── raw/

│   │   └── oasis\_longitudinal\_demographics.xlsx

│   └── processed/

│       ├── template\_connectome.npy     \# Normalized adjacency matrix (N x N)

│       └── longitudinal\_dataset.pt     \# Preprocessed PyTorch Geometric/Temporal tensors

├── src/

│   ├── \_\_init\_\_.py

│   ├── dataset/

│   │   ├── \_\_init\_\_.py

│   │   ├── oasis\_loader.py             \# OASIS-2 longitudinal tabular & ROI parser

│   │   └── graph\_builder.py            \# Connectome Laplacian & PyG graph construction

│   ├── models/

│   │   ├── \_\_init\_\_.py

│   │   ├── baseline\_ndm.py             \# Legacy 2012 analytical benchmark

│   │   ├── st\_gnn\_ode.py               \# Model 1: Continuous-Time Graph Neural ODE

│   │   ├── nd\_vae.py                   \# Model 2: Physics-Constrained VAE

│   │   └── tadm\_diffusion.py           \# Model 3: Temporally-Aware Diffusion Model

│   ├── training/

│   │   ├── \_\_init\_\_.py

│   │   ├── train\_st\_gnn\_ode.py

│   │   ├── train\_nd\_vae.py

│   │   ├── train\_tadm.py

│   │   └── loss\_functions.py           \# Physics losses, score matching, MSE/L1

│   └── evaluation/

│       ├── \_\_init\_\_.py

│       ├── metrics.py                  \# RMSE, MAE, Pearson r, CDR F1/AUC

│       └── benchmark\_runner.py         \# Comparative benchmark across all 4 models

├── scripts/

│   ├── prepare\_data.py                 \# Standalone ETL pipeline

│   └── run\_experiments.py              \# Orchestration entrypoint

├── requirements.txt                    \# Pinned environment dependencies

└── README.md

---

## 3\. Data Pipeline & Representation Specification

### 3.1 Input Formats & Processing

* **Connectome Matrix (`template_connectome.npy`)**:  
  * Shape: $(N, N)$ where $N$ is the number of parcellated regions of interest (ROIs).  
  * Compute normalized symmetric graph Laplacian: $$\\tilde{A} \= A \+ I\_N, \\quad \\tilde{D}*{ii} \= \\sum\_j \\tilde{A}*{ij}, \\quad \\hat{L} \= I\_N \- \\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2}$$  
* **OASIS-2 Demographics & Volumetrics (`oasis_longitudinal_demographics.xlsx`)**:  
  * Mandatory Columns: `Subject ID`, `MRI ID`, `Visit`, `MR Delay` (days), `M/F`, `Age`, `EDUC`, `SES`, `MMSE`, `CDR`, `eTIV`, `nWBV`, `ASF`.  
  * Time Vector ($t$): Calculated as normalized years from baseline: $t \= \\frac{\\text{MR Delay}}{365.25}$.  
  * Subject-level Split: 70% Train, 15% Validation, 15% Test partitioned strictly by `Subject ID` to eliminate longitudinal subject leakage.

### 3.2 Tensor Data Contract

For subject $i$ with $K\_i$ visits:

* **Node Feature Matrix**: $\\mathbf{X}\_i \\in \\mathbb{R}^{K\_i \\times N \\times D}$ ($D$: node features, e.g., regional atrophy, normalized volume, cortical thickness).  
* **Static Covariate Vector**: $\\mathbf{c}\_i \\in \\mathbb{R}^{C}$ (Normalized Age, One-hot Sex, Education, SES, eTIV, ASF).  
* **Observation Timestamps**: $\\mathbf{t}*i \= \[t*{i,0}, t\_{i,1}, \\dots, t\_{i,K\_i-1}\] \\in \\mathbb{R}^{K\_i}$, where $t\_{i,0} \= 0$.  
* **Target Clinical Endpoints**: $\\mathbf{y}\_i \\in \\mathbb{R}^{K\_i \\times 2}$ (Continuous MMSE score, Ordinal CDR $\\in {0, 0.5, 1, 2}$).

---

## 4\. Model Architectures & Technical Specifications

### 4.1 Model 1: Continuous-Time Spatio-Temporal Graph Neural ODE (ST-GNN-ODE)

* **Rationale**: Replaces the linear diffusion equation ($\\frac{d\\mathbf{x}}{dt} \= \-\\beta L \\mathbf{x}$) with a non-linear continuous-time neural vector field operating on graph topologies, natively handling arbitrary irregular observation times $t\_k$.  
* **Mathematical Formulation**: $$\\mathbf{h}(t\_0) \= \\text{Encoder}(\\mathbf{X}(t\_0), \\mathbf{c})$$ $$\\frac{d\\mathbf{h}(t)}{dt} \= f\_\\theta(\\mathbf{h}(t), \\hat{L}, t) \= \\text{GATConv}\\Big(\\mathbf{h}(t), \\mathcal{E}\\Big) \+ \\text{MLP}\\Big(\\mathbf{h}(t) \\mathbin{\\Vert} \\mathbf{c} \\mathbin{\\Vert} \\sin(\\omega t)\\Big)$$ $$\\mathbf{h}(t\_k) \= \\mathbf{h}(t\_0) \+ \\int\_{t\_0}^{t\_k} f\_\\theta(\\mathbf{h}(\\tau), \\hat{L}, \\tau) , d\\tau \\quad (\\text{solved via RK4 or dopri5 via } \\texttt{torchdiffeq})$$ $$\\hat{\\mathbf{X}}(t\_k) \= \\text{Decoder}*{\\text{recon}}(\\mathbf{h}(t\_k)), \\quad \\hat{\\mathbf{y}}(t\_k) \= \\text{Decoder}*{\\text{clinical}}(\\mathbf{h}(t\_k))$$  
* **Loss Function**: $$\\mathcal{L}*{\\text{ODE}} \= \\sum*{k=1}^{K-1} \\Big( |\\hat{\\mathbf{X}}(t\_k) \- \\mathbf{X}(t\_k)|*2^2 \+ \\lambda*{\\text{MMSE}} (\\hat{\\text{MMSE}}\_k \- \\text{MMSE}*k)^2 \+ \\lambda*{\\text{CDR}} \\text{CrossEntropy}(\\hat{\\text{CDR}}\_k, \\text{CDR}\_k) \\Big)$$

---

### 4.2 Model 2: Physics-Constrained Network Diffusion VAE (ND-VAE)

* **Rationale**: Bridges biophysical first principles and deep representation learning. The encoder maps longitudinal biomarker trajectories to a probabilistic latent space, while the decoder is regularized by an explicit network diffusion penalty.  
* **Mathematical Formulation**:  
  * **Encoder**: $$q\_\\phi(\\mathbf{z} \\mid \\mathbf{X}*{0:K-1}, \\mathbf{t}, \\mathbf{c}) \\sim \\mathcal{N}(\\boldsymbol{\\mu}*\\phi, \\boldsymbol{\\sigma}^2\_\\phi)$$  
  * **Generative Trajectory Decoder**: $$\\hat{\\mathbf{X}}(t) \= g\_\\psi(\\mathbf{z}, t, \\mathbf{c})$$  
  * **Biophysical Regularizer**: Penalizes deviations from connectome diffusion dynamics: $$\\mathcal{R}*{\\text{physics}}(\\hat{\\mathbf{X}}, t) \= \\left| \\frac{\\partial \\hat{\\mathbf{X}}(t)}{\\partial t} \+ \\beta*{\\text{diff}} \\hat{L} \\hat{\\mathbf{X}}(t) \\right|\_F^2$$ where $\\frac{\\partial \\hat{\\mathbf{X}}(t)}{\\partial t}$ is computed via automated differentiation with respect to input scalar $t$.  
* **Loss Function**: $$\\mathcal{L}*{\\text{ND-VAE}} \= \\mathbb{E}*{q\_\\phi}\\left\[ \\sum\_{k=0}^{K-1} |\\hat{\\mathbf{X}}(t\_k) \- \\mathbf{X}(t\_k)|*1 \\right\] \+ \\beta*{\\text{KL}} D\_{\\text{KL}}\\Big(q\_\\phi(\\mathbf{z} \\mid \\cdot) ,||, \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\Big) \+ \\lambda\_{\\text{phys}} \\int\_0^{t\_{\\max}} \\mathcal{R}\_{\\text{physics}}(\\hat{\\mathbf{X}}, \\tau) , d\\tau$$

---

### 4.3 Model 3: Temporally-Aware Trajectory Diffusion Model (TADM)

* **Rationale**: Leverages denoising diffusion probabilistic models (DDPM) conditioned on baseline patient state ($\\mathbf{X}(0), \\mathbf{c}$) and target continuous elapsed duration ($\\Delta t$) to model the probability distribution of future neurodegeneration states.  
* **Mathematical Formulation**:  
  * **Forward Process**: For target state $\\mathbf{X}\_k \= \\mathbf{X}(t\_k)$ at continuous interval $\\Delta t\_k \= t\_k \- t\_0$: $$q(\\mathbf{X}\_k^{(\\tau)} \\mid \\mathbf{X}*k) \= \\mathcal{N}\\Big(\\sqrt{\\bar{\\alpha}*\\tau} \\mathbf{X}*k, , (1 \- \\bar{\\alpha}*\\tau) \\mathbf{I}\\Big) \\quad \\text{for diffusion step } \\tau \\in {1, \\dots, T}$$  
  * **Conditioning Mechanism**: $$\\mathbf{e}*{\\text{cond}} \= \\text{MLP}*{\\text{time}}(\\text{Sinusoidal}(\\Delta t\_k)) \+ \\text{GCN}(\\mathbf{X}(0), \\hat{L}) \+ \\text{MLP}\_{\\text{cov}}(\\mathbf{c})$$  
  * **Denoising Backbone**: Parameterized network $\\boldsymbol{\\epsilon}\_\\theta(\\mathbf{X}*k^{(\\tau)}, \\tau, \\mathbf{e}*{\\text{cond}})$ using Adaptive Group Normalization (AdaGN) / FiLM modulation.  
* **Loss Function**: $$\\mathcal{L}*{\\text{TADM}} \= \\mathbb{E}*{\\mathbf{X}\_0, \\Delta t, \\mathbf{X}*k, \\tau, \\boldsymbol{\\epsilon}}\\left\[ \\left| \\boldsymbol{\\epsilon} \- \\boldsymbol{\\epsilon}*\\theta\\Big(\\mathbf{X}*k^{(\\tau)}, , \\tau, , \\mathbf{e}*{\\text{cond}}\\Big) \\right|\_2^2 \\right\]$$

---

## 5\. Benchmarking & Evaluation Suite

To substantiate improvement over the legacy 2012 Network Diffusion Model, the test harness must evaluate all models across the exact same test splits:

| Metric | Target Variable | Objective |
| :---- | :---- | :---- |
| **RMSE & MAE** | Future Regional Atrophy ($\\hat{\\mathbf{X}}(t\_k)$) | Minimize prediction error at follow-up visits |
| **Pearson Correlation ($r$)** | Regional Atrophy Patterns | Measure spatial correspondence across connectome ROIs |
| **Macro F1 & Multiclass AUC** | Clinical Dementia Rating (CDR: 0, 0.5, 1, 2\) | Staging classification accuracy |
| **MAE** | MMSE Score (0–30) | Measure cognitive score projection error |
| **Log-Likelihood / CRPS** | Probabilistic Spread | Evaluate predictive uncertainty |

---

## 6\. Phased Implementation Roadmap for Antigravity

### Phase 1: Environment & Dependency Setup

* Create `requirements.txt`:  
    
  torch\>=2.2.0  
    
  torch-geometric\>=2.5.0  
    
  torchdiffeq\>=0.2.3  
    
  numpy\>=1.24.0  
    
  pandas\>=2.0.0  
    
  openpyxl\>=3.1.0  
    
  scipy\>=1.11.0  
    
  scikit-learn\>=1.3.0  
    
  pyyaml\>=6.0.1  
    
  tqdm\>=4.66.0

### Phase 2: Data Preprocessing & Pipeline Construction

* Implement `src/dataset/oasis_loader.py` to parse demographics, sort visits by `Subject ID` and `Visit`, compute elapsed years $\\Delta t$, and build tabular tensors.  
* Implement `src/dataset/graph_builder.py` to load `template_connectome.npy`, verify symmetry, compute normalized Laplacian $\\hat{L}$, and generate PyTorch Geometric `Data` structures.  
* Implement subject-level stratified splitting (`prepare_data.py`).

### Phase 3: Baseline NDM Verification

* Re-implement/refactor `src/models/baseline_ndm.py` solving $\\mathbf{x}(t) \= \\exp(-\\beta \\hat{L} t) \\mathbf{x}(0)$ via matrix exponential (`scipy.linalg.expm` / `torch.linalg.matrix_exp`). Record baseline test metrics.

### Phase 4: Implementation of the 3 Deep Learning Models

* **Model 1**: Code `src/models/st_gnn_ode.py` using `torchdiffeq.odeint`. Include adjoint sensitivity method option for memory efficiency.  
* **Model 2**: Code `src/models/nd_vae.py` with automated differentiation for the biophysical loss penalty $\\mathcal{R}\_{\\text{physics}}$.  
* **Model 3**: Code `src/models/tadm_diffusion.py` with sinusoidal temporal conditioning and DDIM sampling support.

### Phase 5: Training Engines & Loss Functions

* Write modular trainers (`train_st_gnn_ode.py`, `train_nd_vae.py`, `train_tadm.py`) with checkpointing, early stopping on validation loss, and tensorboard/wandb logging.

### Phase 6: Automated Evaluation & Comparative Table Generation

* Build `src/evaluation/benchmark_runner.py` to run inference across test subjects, compute all metrics, and output a markdown comparison table against the 2012 NDM baseline.

