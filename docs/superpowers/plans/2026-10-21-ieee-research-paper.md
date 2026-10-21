# IEEE Research Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an IEEE-format (IEEEtran conference, two-column) research paper, compiled to PDF, describing the OASIS-2 digital-twin pipeline (NDM baseline, ST-GNN-ODE, ND-VAE, TADM, cognitive-deficit mapping, 3D simulator) with the two simulator screenshots analysed in the Results section.

**Architecture:** LaTeX master (`paper/main.tex`) + BibTeX (`paper/references.bib`). Figures come from three sources: the user's two screenshots, TikZ diagrams written inline, and two matplotlib figures computed from files already committed in the repo (demographics xlsx, template connectome). Every number in the paper traces to code, the demographics file, a screenshot, or a verified citation. Nothing is invented.

**Tech Stack:** MiKTeX (`latexmk`, `pdflatex`, `bibtex`; `IEEEtran`, `tikz`, `booktabs`, `cite`, `amsmath` all confirmed installed), Python 3.11 with numpy/pandas/scipy/matplotlib.

**User decisions (from Q&A):** Results = screenshots + qualitative only (no benchmark metrics exist; the saved Colab run stopped at a W-score error). Format = LaTeX IEEEtran → PDF. Scope = all three models + NDM baseline + simulator. References = repo refs + web-verified additions. Authors = multiple; names not yet supplied, so the author block is one clearly marked macro to fill in. Simulator source is unavailable, so it is described only from what is visible.

## Honesty constraints (these shape the text)

1. `data/processed/template_connectome.npy` is the **synthetic** connectome from `build_synthetic_connectome` (56×56, symmetric, density 0.208, exponential weights thresholded at the 80th percentile, +0.5 medial-temporal boost). The README implies it follows Raj 2015's healthy reference connectome. The paper must call it a surrogate template and list this as a limitation.
2. Implemented models are simplified relative to `model*.md`. ST-GNN-ODE uses a single-head GAT, hidden size 32, and 5 fixed Euler steps per visit (no `torchdiffeq`/dopri5). ND-VAE's physics loss uses finite differences, not autograd. TADM uses T=200, an MLP denoiser (hidden 64), and 10 DDIM steps. The paper describes the **implemented** versions.
3. Node features are a single channel (W-score). `build_dl_datasets` fills later visits with the baseline W-score times exp(−0.1·t). This is a modeling simplification and goes in Limitations.
4. No benchmark table is reported. The paper says quantitative validation is future work.
5. Screenshot 2 shows all 11 domains at an identical 94%. The paper describes this as saturation and does not assert a cause. Screenshot 1's β=9.101 sits near the fit upper bound of 10, which is also noted.
6. Do not name the simulator's implementation, framework or data pipeline.

## File structure

- Create: `paper/main.tex` — document, all sections
- Create: `paper/references.bib` — verified references only
- Create: `paper/figures/sim_cn_baseline.jpg` — screenshot 1 (copy of `images/1.jpg`)
- Create: `paper/figures/sim_demented_year3p3.jpg` — screenshot 2 (copy of `images/2.jpg`)
- Create: `paper/figures/fig_cohort.pdf`, `paper/figures/fig_ndm_spectrum.pdf` — generated
- Create: `paper/scripts/make_figures.py` — regenerates the two matplotlib figures from repo data
- Create: `paper/figures/pipeline.tex`, `paper/figures/models.tex` — TikZ diagrams `\input` by main.tex
- Modify: `.gitignore` — ignore LaTeX build artifacts (not `main.pdf`)

---

### Task 1: Scaffold and compile a skeleton

**Files:** Create `paper/main.tex`, `paper/references.bib`; modify `.gitignore`

- [ ] **Step 1:** Create `paper/main.tex` with `\documentclass[conference]{IEEEtran}`, packages `cite, amsmath, amssymb, graphicx, booktabs, tikz, url, microtype, textcomp, xcolor`, title, a single `\author{...}` block holding placeholder names, abstract, IEEE keywords, empty section headings, and `\bibliographystyle{IEEEtran}\bibliography{references}`.
- [ ] **Step 2:** Create `paper/references.bib` with one entry (`marcus2010oasis`) so BibTeX runs.
- [ ] **Step 3:** Compile.

Run: `cd paper && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: `main.pdf` produced, 0 errors.

- [ ] **Step 4:** Append to `.gitignore`: `paper/*.aux`, `paper/*.bbl`, `paper/*.blg`, `paper/*.fdb_latexmk`, `paper/*.fls`, `paper/*.log`, `paper/*.out`.
- [ ] **Step 5:** Commit: `git add paper .gitignore && git commit -m "docs(paper): scaffold IEEEtran paper"`

### Task 2: Figures from repo data and screenshots

**Files:** Create `paper/scripts/make_figures.py`, `paper/figures/*`

- [ ] **Step 1:** Copy the screenshots from `C:\Users\krish\AppData\Local\Temp\claude\...\images\{1,2}.jpg` into `paper/figures/` under the names above.
- [ ] **Step 2:** Write `make_figures.py`.
  - `fig_cohort.pdf`: three panels from the demographics xlsx. (a) Histogram of visits per subject. (b) CDR counts across all 373 sessions. (c) Baseline age histogram by Group (Nondemented / Demented / Converted).
  - `fig_ndm_spectrum.pdf`: two panels from the template connectome. (a) Laplacian eigenvalue spectrum (L = D − A, same as `compute_laplacian`). (b) Mode-wise decay exp(−βλ_i t) for β = 7.233 and β = 9.101 at t = 3.3 years, plotted against λ_i, with β values taken from the screenshots.
  - Use grayscale-safe styles and a single-column width of 3.4 in (or double-column 7.0 in).
- [ ] **Step 3:** Run: `python -X utf8 paper/scripts/make_figures.py`. Expected: both PDFs written.
- [ ] **Step 4:** Commit.

### Task 3: Verified references

**Files:** Modify `paper/references.bib`. All entries below were verified by search or are from the user's docs and are standard.

- [ ] **Step 1:** Add entries for:
  - Raj 2012 (Neuron 73(6):1204–1215)
  - Raj 2015 (Cell Reports 10(3):359–369)
  - Marcus 2010 (J Cogn Neurosci 22(12):2677–2684, doi 10.1162/jocn.2009.21407)
  - Billot 2023 (Med Image Anal 86:102789, doi 10.1016/j.media.2023.102789)
  - Yagis 2021 (Sci Rep 11:22544, doi 10.1038/s41598-021-01681-w)
  - Wen 2020 (Med Image Anal 63:101694)
  - Desikan 2006 (NeuroImage 31(3):968–980, doi 10.1016/j.neuroimage.2006.01.021)
  - Chen 2018 (Neural ODE, NeurIPS)
  - Rubanova 2019 (Latent ODEs, NeurIPS 32, pp. 5320–5330)
  - Velickovic 2018 (GAT, ICLR)
  - Kingma 2014 (VAE, ICLR)
  - Raissi 2019 (PINN, J Comput Phys 378:686–707)
  - Ho 2020 (DDPM, NeurIPS)
  - Song 2021 (DDIM, ICLR)
  - Nichol 2021 (Improved DDPM, PMLR 139:8162–8171)
  - Perez 2018 (FiLM, AAAI)
  - Sundararajan 2017 (Integrated Gradients, PMLR 70:3319–3328)
  - Wang 2024 (Virtual brain twins, Natl Sci Rev 11(5):nwae079, doi 10.1093/nsr/nwae079)
- [ ] **Step 2:** Compile; confirm no "undefined citation" warnings once the sections cite them (re-check in Task 6).

### Task 4: TikZ diagrams

**Files:** Create `paper/figures/pipeline.tex`, `paper/figures/models.tex`

- [ ] **Step 1:** `pipeline.tex`: a left-to-right block diagram of T1 MRI → segmentation (SynthSeg / FSL fallback) → regional volumes → W-score → NDM fit (β) → DL models (ST-GNN-ODE / ND-VAE / TADM) → deficit mapping → 3D twin simulator. Fits one column or `figure*`.
- [ ] **Step 2:** `models.tex`: a three-row schematic (`figure*`). ST-GNN-ODE: encoder → ODE(GAT + MLP(sin t)) → decoders. ND-VAE: GRU encoder → z → decoder, with a physics-loss arrow. TADM: baseline GCN + time + covariates → e_cond → FiLM denoiser.
- [ ] **Step 3:** `\input` both from `main.tex`, compile, and confirm they render without overfull boxes.
- [ ] **Step 4:** Commit.

### Task 5: Write the paper sections in `main.tex`

Write the text from the facts below. Target 6–7 pages.

- [ ] **Step 1 — Title, abstract, keywords.**
  - Working title: *Connectome-Constrained Digital Twins of Longitudinal Neurodegeneration on OASIS-2: Network Diffusion, Graph Neural ODE and Generative Forecasting with an Interactive 3D Simulator*.
  - Abstract (~200 words): problem; the 56-region pipeline; four forecasters; the cognitive-domain deficit mapping; the simulator; qualitative results; an honest note on limitations.
  - Keywords: digital twin, Alzheimer's disease, network diffusion, neural ODE, graph neural network, longitudinal MRI, OASIS-2.
- [ ] **Step 2 — I. Introduction.**
  - Motivation: which regions are affected now, and which next.
  - Gaps: linear NDM, irregular visits, interpretability.
  - Contributions (4 bullets).
  - Paper roadmap.
- [ ] **Step 3 — II. Related Work.**
  - NDM [Raj 2012, 2015]
  - Neural ODEs for irregular time [Chen 2018, Rubanova 2019]
  - GATs [Velickovic]
  - Physics-informed learning [Raissi]
  - VAE [Kingma]
  - Diffusion models [Ho, Song, Nichol, Perez]
  - Virtual brain twins [Wang 2024]
  - Leakage-aware evaluation [Yagis, Wen]
- [ ] **Step 4 — III. Data and Preprocessing.**
  - OASIS-2 [Marcus 2010]: 150 subjects, 373 sessions, ages 60–96, 94/43/9/4 subjects with 2/3/4/5 visits, baseline groups 72/64/14, 88 F / 62 M, baseline age 75.4±7.5, MMSE 27.6±3.0, CDR counts 206/123/41/3, max follow-up 2639 days. (Table I, Fig. 2.)
  - Parcellation: 56 regions (34 left cortical, 10 subcortical, 12 right cortical; Desikan–Killiany naming [Desikan 2006]); SynthSeg [Billot] or FSL fallback.
  - W-score: age regression fit on CDR=0 visits, residual z-score, sign flipped so atrophy is positive.
  - Template connectome (synthetic surrogate, stated explicitly) and Laplacian L = D − A.
- [ ] **Step 5 — IV. Methods.**
  - A. NDM baseline: x(t) = exp(−βLt)x(0) via eigendecomposition; per-subject β by bounded scalar optimisation on [0.001, 10], fit on visit 1→2 and validated on visit 3.
  - B. ST-GNN-ODE: equations as implemented, single-head masked GAT, sinusoidal time encoding, 5 Euler steps, MSE + 0.5·MMSE (as in code; check the exact weights when writing) + CDR cross-entropy.
  - C. ND-VAE: GRU encoder, z∈R^16, MLP decoder, L1 reconstruction + β_KL·KL (annealed to 0.1) + λ_phys=0.01 · finite-difference ‖dX/dt + β L X‖² + clinical heads.
  - D. TADM: cosine schedule T=200, GCN/time/covariate conditioning, FiLM MLP denoiser, ε-prediction MSE, DDIM sampling with 10 steps, repeated sampling for uncertainty.
  - E. Training protocol: subject-level 70/15/15 split (seed 42), batch 8, AdamW lr 1e-3, wd 1e-4, cosine LR, early stopping (patience 10/10/15), epoch caps 50/100/200. Metrics defined: atrophy RMSE/MAE/Pearson r, MMSE MAE, CDR accuracy/macro-F1.
  - F. Cognitive-domain deficit mapping: 11 domains, region→domain weight map, weighted mean then min–max normalisation.
  - G. Digital-twin simulator: described from the screenshots only, covering the panels and the propagation label exp(−βLt)x0.
- [ ] **Step 6 — V. Results (two simulator case studies).**
  - Fig. 5 / screenshot 1: OAS2_0001, Nondemented, 87/M, CDR 0.0, β=9.101, Year 0.0. Active region left insula, +0.82σ, associated with emotional regulation and interoception. Domain bars: object recognition 73%, emotional regulation 71%, behavioral control 66%, attention and semantic memory 61%, episodic memory 55%, working memory 48%, learning and visuospatial 43%, executive 40%, language 35%. Explain: relative within-subject ranking, since scores are min–max normalised, so a high percentage in a CDR-0 subject is not a clinical risk. Explain the colour scale, the 56 nodes and the tracts.
  - Fig. 6 / screenshot 2: OAS2_0002, Demented, 75/M, CDR 0.5, β=7.233, Year 3.3. Active region Left-Hippocampus, +0.94σ, subcortical nucleus, associated with episodic memory and learning. All 11 domains read 94% (saturation). Integrated-Gradients explainability text names Left Hippocampus and Entorhinal Cortex as baseline drivers.
  - Comparative discussion: baseline vs. year-3.3, healthy vs. demented twin, effect of larger vs. smaller β. Use Fig. (NDM spectrum) to show faster high-frequency mode decay at larger β.
  - Table: side-by-side summary of the two cases. Everything in the table comes from the screenshots.
  - Paragraph: what we do *not* claim, namely no predictive-accuracy claim from the screenshots.
- [ ] **Step 7 — VI. Discussion and Limitations; VII. Conclusion and Future Work.**
  - Limitations: the five honesty constraints above, plus subject count (small N), single scanner, no multimodal data, simulator saturation, β near its bound, evaluation scripts that compare against the same W-score features.
  - Future work: real DTI connectome (Budapest / HCP), quantitative benchmark with leakage-free subject-level CV, calibration of the uncertainty bands, user study of the simulator.
- [ ] **Step 8:** Compile after each section group. Commit.

### Task 6: Verify, polish, ship

- [ ] **Step 1:** `cd paper && latexmk -pdf -interaction=nonstopmode main.tex`, then grep `main.log` for `Warning` and `Overfull`. Fix undefined references/citations and any overfull box over 5 pt.
- [ ] **Step 2:** Convert PDF pages to PNG and read each one to confirm layout: two columns, figure placement, no caption overlap, screenshots legible. Command: `pdftoppm -r 80 -png main.pdf pg` if available, otherwise a Python fitz/pdf2image fallback.
- [ ] **Step 3:** Check page count (target 6–8) and that every figure/table is referenced in text.
- [ ] **Step 4:** Fact-check pass: every number in the PDF against the source (code, xlsx, screenshots).
- [ ] **Step 5:** Commit `paper/main.pdf` along with the sources. Message: `docs(paper): add IEEE paper source and compiled PDF`.
- [ ] **Step 6:** Report to the user: the PDF path, the author block to fill in, and the honesty caveats (synthetic connectome, no benchmark numbers).
