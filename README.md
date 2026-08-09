# OASIS-2 Longitudinal Neurodegeneration Pipeline (Colab Version)

This repository contains the complete pipeline for modeling longitudinal neurodegeneration using the OASIS-2 dataset, consolidated entirely into a single **Google Colab Notebook**.

## Getting Started

### 1. Download the Notebook
You can find the standalone Colab notebook here:

otebooks/OASIS2_Complete_Colab_Pipeline.ipynb

### 2. Upload to Google Colab
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **File -> Upload notebook**.
3. Select OASIS2_Complete_Colab_Pipeline.ipynb.

### 3. Run the Pipeline
The notebook is completely self-contained. It contains all the necessary source code for the models, data preprocessing, and evaluation.
- Go to **Runtime -> Run all** to execute the entire pipeline end-to-end.
- It will guide you through uploading your Kaggle kaggle.json credentials so it can securely download the OASIS-2 MRI scans directly to the Colab environment.

*(Note: For the best performance, especially during the CNN and Network Diffusion Model phases, enable a GPU in Colab by going to Runtime -> Change runtime type -> T4 GPU).*

## Repository Structure

`	ext
├── data/
│   └── raw/                                  # Place the kaggle OASIS-2 MRI files here if doing manual inspection
├── notebooks/
│   └── OASIS2_Complete_Colab_Pipeline.ipynb  # 🌟 The Main Pipeline!
├── README.md
└── implementation.md                         # Detailed mathematical and architecture document
`
