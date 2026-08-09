# Longitudinal Neurodegeneration Progression Modeling on OASIS-2

A computational pipeline for modeling the longitudinal progression of neurodegeneration using the [OASIS-2](https://www.oasis-brains.org/) dataset. This project adapts established neuroimaging techniques — structural segmentation, network diffusion modeling, and cognitive deficit prediction — into an end-to-end workflow that runs on **Google Colab** or **locally**.

## Overview

The pipeline takes longitudinal T1-weighted MRI scans from 150 subjects (aged 60–96, tracked for dementia via CDR) and progresses through seven phases:

| Phase | Description | Method |
|-------|-------------|--------|
| **0** | Data Audit | Validate scans, build subject index, log exclusions |
| **1** | Preprocessing & Segmentation | SynthSeg whole-brain parcellation → regional volumes |
| **2a** | Classification (Primary) | SVM / Random Forest on volumetric + clinical features |
| **2b** | Classification (Secondary) | Transfer-learning CNN (ResNet-18 / VGG16), exploratory |
| **3** | Network Diffusion Model | NDM on structural connectome — models pathology spread |
| **4** | Cognitive Deficit Prediction | Region → cognitive-domain mapping, validated against MMSE/CDR |
| **5** | Clinical Risk Profiling | Demographic + imaging feature risk stratification |
| **6** | Progression vs. Fluctuation | Trajectory-based classification from longitudinal CDR |

> **Design rationale:** This is a structural analog of a glioblastoma computational pipeline (multi-modal segmentation → reaction-diffusion PDE → symptom mapping → radiogenomics), adapted to what OASIS-2's T1-only longitudinal data actually supports. See [implementation.md](implementation.md) for the full mapping and mathematical details.

## Data Setup

### Demographics (included in repo)

The longitudinal demographics spreadsheet is included at:
```
data/raw/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx
```

### Raw MRI Scans (download separately — ~24.7 GB)

The raw MRI data is too large for GitHub. Download it and place it in the project:

1. **Download** the OASIS-2 raw MRI data from: `<YOUR_GOOGLE_DRIVE_LINK_HERE>`
   - Alternatively, download directly from [oasis-brains.org](https://www.oasis-brains.org/#data)
2. **Extract** into `data/raw/` so the structure looks like:
   ```
   data/raw/OAS2_RAW_PART1/
   ├── OAS2_0001_MR1/RAW/
   │   ├── mpr-1.nifti.img
   │   ├── mpr-1.nifti.hdr
   │   ├── mpr-2.nifti.img
   │   └── ...
   ├── OAS2_0001_MR2/RAW/
   └── ...
   ```

### Template Connectome (included in repo)

A structural connectome template (`data/processed/template_connectome.npy`) is included for the Network Diffusion Model (Phase 3). This follows the approach of [Raj et al. (2015)](https://doi.org/10.1016/j.celrep.2014.12.034), who used a healthy reference connectome for ADNI predictions under the same constraint (no diffusion MRI available).

## Notebooks

| Notebook | Description |
|----------|-------------|
| [OASIS2_Complete_Colab_Pipeline_FIXED (1).ipynb](<OASIS2_Complete_Colab_Pipeline_FIXED (1).ipynb>) | Full pipeline for **Google Colab** — self-contained, handles data download via Kaggle API |
| [OASIS2_Local_Pipeline.ipynb](OASIS2_Local_Pipeline.ipynb) | Full pipeline for **local execution** — uses data from `data/raw/` |

### Running on Google Colab

1. Upload the Colab notebook to [Google Colab](https://colab.research.google.com/)
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Run all cells — the notebook will prompt for Kaggle credentials to download the data
4. For best results, mount Google Drive to persist outputs across sessions

### Running Locally

1. Clone this repo and [set up the data](#data-setup)
2. Install dependencies:
   ```bash
   pip install nibabel nilearn numpy scipy pandas scikit-learn matplotlib seaborn networkx SimpleITK statsmodels torch
   ```
3. Open `OASIS2_Local_Pipeline.ipynb` in Jupyter and run all cells

## Repository Structure

```
├── .gitignore
├── README.md
├── implementation.md                              # Full mathematical and architectural documentation
├── OASIS2_Complete_Colab_Pipeline_FIXED (1).ipynb  # Colab notebook (self-contained)
├── OASIS2_Local_Pipeline.ipynb                    # Local notebook
├── data/
│   ├── raw/
│   │   ├── oasis_longitudinal_demographics-*.xlsx # Demographics spreadsheet (tracked)
│   │   └── OAS2_RAW_PART1/                        # Raw MRI scans (not tracked — see Data Setup)
│   └── processed/
│       └── template_connectome.npy                # Structural connectome for NDM
├── models/                                        # Saved model checkpoints (not tracked)
└── outputs/                                       # Figures, metrics, reports (not tracked)
```

## Key Methodology Notes

- **Subject-level splitting only** — Yagis et al. (2021) measured slice-level cross-validation inflating OASIS-specific accuracy by ~30 percentage points. All evaluation uses `GroupKFold` on subject ID.
- **Age-normalized atrophy scores** — Regional volumes are residualized against age-expected trajectories from healthy controls (W-scores), not raw z-scores, to avoid conflating normal aging with disease.
- **Realistic accuracy expectations** — 60–75% subject-level classification accuracy is consistent with published, leakage-free results at this sample size (~150 subjects). Anything much higher warrants a leakage check.
- **NDM uses a template connectome** — Standard practice per Raj et al. (2012, 2015) when patient-specific DTI is unavailable.

## References

- Raj, A., Kuceyeski, A., & Weiner, M. (2012). A network diffusion model of disease progression in dementia. *Neuron*, 73(6), 1204–1215.
- Raj, A., et al. (2015). Network diffusion model of progression predicts longitudinal patterns of atrophy and metabolism in Alzheimer's disease. *Cell Reports*, 10(3), 359–369.
- Wen, J., et al. (2020). Convolutional neural networks for classification of Alzheimer's disease: Overview and reproducible evaluation. *Medical Image Analysis*, 63, 101694.
- Yagis, E., et al. (2021). Effect of data leakage in brain MRI classification using 2D convolutional neural networks. *Scientific Reports*, 11, 22544.
- Billot, B., et al. (2023). SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining. *Medical Image Analysis*.
- Marcus, D. S., et al. (2010). Open Access Series of Imaging Studies (OASIS-2): Longitudinal MRI data in nondemented and demented older adults.

## License

This project uses the [OASIS-2 dataset](https://www.oasis-brains.org/), which is available under its own data use agreement.
