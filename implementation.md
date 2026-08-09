# Implementation Plan: Longitudinal Neurodegeneration Progression Modeling on OASIS-2

## 0. Read this first (context for whoever/whatever builds this)

The source document (`Brain_Tumor_Growth_Prediction_Model.md`) is a literature review of a **glioblastoma** computational pipeline: multi-modal MRI (T1/T1c/T2/FLAIR) segmentation → DTI-based reaction-diffusion tumor growth PDE → eloquent-cortex symptom mapping → radiogenomics → pseudoprogression classification.

The actual dataset available is **OASIS-2** (longitudinal T1-weighted-only MRI, 150 subjects, 60–96 yrs, tracked for dementia via CDR). This plan is written to run on **Google Colab** or **Google Antigravity** — the two environments it'll actually be executed in — rather than assuming a fixed local CPU machine (see §1 for what differs between them). OASIS-2 has no contrast-enhanced sequences, no tumor labels, no DTI, no genomic data, and no treatment/radiotherapy records — so the tumor pipeline cannot be built as literally described. Every stage below is a **direct structural analog**, not a reinterpretation of tumor code applied to the wrong data:

| Original (tumor) stage | This implementation | Why it's a legitimate analog |
|---|---|---|
| Multi-modal 3D MRI segmentation (nnU-Net / Swin UNETR) | Single-modality (T1) whole-brain parcellation (SynthSeg) + volumetric-feature classification, with transfer-learning CNN as a secondary track | Same "deep learning does the spatial/categorical labeling" role, scoped to what T1-only data and this sample size actually support (see §4) |
| Fisher-Kolmogorov reaction-diffusion PDE + DTI anisotropy tensor | **Network Diffusion Model (NDM)** on a structural connectome graph | NDM (Raj et al.) is the literature's own established analog: pathology spreading along brain connectivity, same math family (diffusion operator on a Laplacian) as reaction-diffusion on a tissue tensor |
| Inverse problem / PINN parameter estimation | Least-squares fit of the NDM diffusivity constant β per subject from their real 2+ visits | Same goal (estimate patient-specific spread-rate parameter from sparse longitudinal imaging), scaled to what's tractable on CPU |
| Cellular pathology (glioma microenvironment) | Literature/discussion section only — not code | The original treats this narratively too; no dataset makes this computational for either disease |
| Neuroanatomical mapping → aphasia/motor/visual deficits | Region→cognitive-domain mapping → memory/language/executive/visuospatial deficit prediction | Same mechanism: overlay predicted regional damage onto a function atlas |
| Radiogenomics (IDH/MGMT) | Clinical/demographic risk stratification (age, education, SES, eTIV, nWBV) | No genotype in OASIS-2; this is explicitly a **proxy**, flagged as such, not a genomic model |
| Pseudoprogression vs. true progression | **Sustained progression vs. fluctuation** classification, using per-visit CDR trajectory shape rather than a single end-state label | This is the one stage where OASIS-2 has a *genuine* ground-truth analog already built into the dataset — real longitudinal clinical scores, not synthesized labels |

If you (Claude Code, or Krish reading this later) hit a step where the data doesn't support what's written, stop and flag it rather than quietly faking a result — same principle as above.

**Priority order if time runs short:** Phases 0–4 are the core deliverable (segmentation → trained model → growth simulation → symptom prediction). Phases 5–6 are real but explicitly lower-confidence extensions (small samples, proxy features) — build them, but report their limitations honestly rather than overselling accuracy.

### Revision notes (research + implementation re-check against literature)

This plan was checked against the primary sources it leans on, and against known failure modes for this exact dataset family. Concrete corrections made:

- **SynthSeg runtime was a placeholder before; it's now a sourced number.** Confirmed ~1–2 min/scan on CPU (vs. ~6–15s on GPU), and the tool has a `--vol` flag that writes regional volumes straight to CSV — this removes a manual voxel-counting step that Phase 1 originally implied.
- **The template-connectome choice in Phase 3 is not a workaround — it's what the field itself does.** Raj, Kuceyeski & Weiner (2012), the paper that introduced the Network Diffusion Model, built its own connectome from only 14 healthy-subject tractography scans. The follow-up validation on ADNI (Raj et al., 2015, *Cell Reports*) explicitly used a healthy reference connectome instead of patient-specific ones *because ADNI had no DTI data* — the identical constraint OASIS-2 has here — and reported that connectome variability had only minor influence on predictions. Phase 3 below cites this directly instead of presenting the template connectome as a lesser fallback.
- **Phase 2 no longer trains a CNN from scratch.** Published, dataset-matched evidence changes the recommended approach here in two ways: (1) Wen et al. (2020, *Medical Image Analysis*) found linear SVMs on volumetric features competitive with CNNs for this exact task, and (2) Yagis et al. (2021, *Scientific Reports*) — who specifically used transfer learning rather than from-scratch training, citing insufficient subjects to do otherwise — reported honest subject-level-split accuracy on OASIS of only ~66% (VGG16, transfer learning), against a fabricated ~97% when the split leaked at the slice level. Phase 2 is restructured below into a primary classical-ML track and a secondary, transfer-learning-based CNN track, with explicit expected-accuracy ranges instead of an implied "train and hope" approach.
- **Data leakage risk is now quantified, not just mentioned.** Yagis et al. (2021) measured slice-level cross-validation inflating reported accuracy by **~30 percentage points specifically on OASIS data** (from ~66% honest to ~97% leaked, in their VGG16 experiment). Subject-level splitting is now stated as a hard requirement with this number attached, so the reason isn't just "trust me."
- **Phase 6 now uses visit-level CDR trajectories, not the pre-computed Group label alone.** The original framing risked being a shallower analog than intended. True-progression-vs-pseudoprogression is fundamentally about *distinguishing a real trend from a look-alike fluctuation* — OASIS-2's per-visit CDR scores support that distinction directly (monotonic decline vs. fluctuation), which is a tighter structural match than the single end-state `Converted`/`Nondemented` label.
- **Phase 3's atrophy scoring is now age-normalized**, not a raw z-score against the nondemented group — regressing out normal age-related volume loss before scoring avoids conflating healthy aging with disease-driven atrophy.

---

## 1. Environment Setup

The two target environments differ enough that they need separate setup paths, but the pipeline code itself should be written once and run unchanged on either — detect GPU availability at runtime (`torch.cuda.is_available()`) rather than hardcoding a CPU or GPU assumption anywhere.

### 1a. Google Colab

Colab's free tier gives GPU access (typically a T4), which matters a lot here: SynthSeg drops from ~1–2 min/scan on CPU to ~6–15s/scan on GPU (see §3), and Phase 2b's transfer-learning CNN trains much faster. It does **not** change the sample-size argument in Phase 2 (see the note there) — GPU speeds iteration, it doesn't make training a CNN from scratch on 150 subjects a good idea.

Colab-specific things that will break the pipeline if ignored:

- **Enable the GPU runtime explicitly**: Runtime → Change runtime type → T4 GPU. It's off by default.
- **The filesystem is ephemeral** — everything outside a mounted Drive is deleted when the runtime recycles (idle timeout ~90 min, hard cap ~12 hr on free tier). Mount Drive first thing in every session and write all persistent outputs there, not to `/content`:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  # work under /content/drive/MyDrive/<project>/ for anything that needs to survive a disconnect
  ```
- **Download OASIS-2 into Drive once, not every session.** It's several GB; re-downloading each session wastes time you don't get back if the runtime disconnects mid-download. After the first successful download, point all later sessions at the Drive copy.
- **Checkpoint aggressively, phase by phase.** Given the idle-timeout risk, each phase should write its output (e.g. `regional_volumes.csv`, model checkpoints, fitted β values) to Drive as soon as it's done, and each notebook should be able to resume from the last checkpoint rather than rerunning everything from scratch after a disconnect. Don't let Phase 1's SynthSeg batch run be a single multi-hour cell with no intermediate saves.
- **Packages need reinstalling each session** (`!pip install ...`) — Colab's base image already has numpy/pandas/scikit-learn/matplotlib/torch (CUDA-enabled) preinstalled, so you only need to add what's missing:
  ```python
  !pip install nibabel nilearn SimpleITK networkx statsmodels
  !pip install synthseg  # standalone package — see the FreeSurfer-license note below
  ```
- **Avoid the full FreeSurfer install for SynthSeg on Colab.** Running `mri_synthseg` through a full FreeSurfer install requires a registered license file, which is unnecessary friction to redo every ephemeral session. Use the standalone SynthSeg package (PyPI/GitHub) instead — it's the same pretrained model without the license-file requirement.

### 1b. Google Antigravity

Antigravity is a local, agent-driven IDE (VS Code-based) — it executes on whatever machine it's installed on, so it doesn't itself provide cloud GPU the way Colab does. Setup is the standard local Python flow, and it matters that every command below is copy-pasteable exactly as written, since an agent will be the one running them:

```bash
python3 -m venv venv
source venv/bin/activate
pip install nibabel nilearn numpy scipy pandas scikit-learn matplotlib seaborn networkx SimpleITK statsmodels
pip install synthseg

# install torch matching whatever hardware this machine actually has:
# GPU present:
pip install torch
# CPU-only machine:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

- Check `nvidia-smi` (or just `torch.cuda.is_available()` in Python) before installing torch, and install the matching build rather than assuming either way.
- Unlike Colab, the filesystem here is persistent — no Drive-mounting or checkpoint-against-disconnect concerns, but it's still good practice to write intermediate outputs (regional volumes, fitted parameters) to disk after each phase rather than keeping everything in memory across a long agent session.
- Since this is agent-executed, keep each phase as a distinct, independently-runnable script under `src/` (per the repo structure in §10) with clear inputs/outputs, rather than one long notebook — that gives the agent (and Krish's partner) a natural place to resume if a phase needs rerunning.

### 1c. Common to both

- **Detect and use GPU when present, fall back cleanly when not.** In every model-training or SynthSeg call, check `torch.cuda.is_available()` and set the device accordingly rather than hardcoding CPU or CUDA.
- **SynthSeg**: the standalone pip-installable package (not the full FreeSurfer suite) is the recommended path in both environments — it's a pretrained CNN, inference only, no training required, and sidesteps the FreeSurfer license-file step. Confirmed runtime: ~1–2 min/scan on CPU, ~6–15s/scan on GPU (see §3).
- Keep `requirements.txt` in the repo so both environments (and both people working on this) install the identical dependency set.

---

## 2. Phase 0 — Data Audit

Before writing any pipeline code, inspect what was actually downloaded — OASIS-2 is distributed in more than one form (raw `.img/.hdr` Analyze volumes via XNAT/oasis-brains.org, vs. Kaggle mirrors that sometimes ship only the demographic CSV plus a subset of processed slices).

1. Walk the dataset directory and log: number of subjects, number of sessions per subject, which derivative folders exist per session (raw `mpr-*`, gain-field-corrected `t88_gfc`, any existing `FSL_SEG` tissue masks).
2. Load `oasis_longitudinal.csv` (or equivalent) and confirm the columns: `Subject ID, MRI ID, Group, Visit, MR Delay, M/F, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF`.
3. Confirm the `Group` label values: `Nondemented`, `Demented`, `Converted` — the `Converted` subjects are the ground truth for Phase 6.
4. Produce a small data-quality report: missing SES/MMSE values (OASIS-2 has known missingness here), subjects with only 1 visit (unusable for any longitudinal step — exclude), scan resolution/orientation consistency.

**Acceptance criteria:** a clean `subjects_index.csv` mapping subject → visit → file paths → demographics → group label, with exclusions logged and justified.

---

## 3. Phase 1 — Preprocessing & Structural Segmentation

Replaces the multi-modal tumor segmentation stage.

1. **Format conversion.** If raw scans are Analyze (`.img/.hdr`), load with `nibabel` (native support) and re-save as NIfTI for downstream tooling.
2. **Use existing derivatives where available.** OASIS's standard release layout (documented for OASIS-1's public distribution; verify OASIS-2 follows the same convention in Phase 0) is `RAW/`, `PROCESSED/SUBJ_111/` (motion-corrected, co-registered average, 1mm isotropic), `PROCESSED/T88_111/` (gain-field-corrected, atlas-registered to Talairach 88 space, plus a brain-masked version), and `FSL_SEG/` (an existing 3-tissue GM/WM/CSF mask from FSL). If `FSL_SEG` is present, use it directly for tissue-level volumes rather than recomputing; SynthSeg (step 3) is still needed for finer subcortical/cortical regions (hippocampus, amygdala, ventricles) that `FSL_SEG` doesn't provide.
3. **Whole-brain parcellation via SynthSeg.** Run `mri_synthseg` (standalone pip package, no full FreeSurfer install required) on the T1 volume for every session:
   ```bash
   mri_synthseg --i t88_scan.nii.gz --o synthseg_out.nii.gz --vol volumes.csv --parc --threads 4
   ```
   - `--vol volumes.csv` writes per-region volumes directly to CSV — do not manually integrate voxel counts from the label map, the tool already does this.
   - `--parc` adds cortical parcellation on top of the ~37-structure subcortical/ventricular segmentation (hippocampus, amygdala, ventricles, thalamus, plus cortical regions) — needed for Phase 4's region→function mapping.
   - `--threads 4` (or however many cores are free) — relevant on CPU; irrelevant when running on Colab's GPU, where SynthSeg picks up CUDA automatically if the standalone package detects it.
   - **Confirmed runtime: ~1–2 min/scan on CPU, ~6–15s/scan on GPU.** On Colab's GPU, all ~373 OASIS-2 sessions take roughly 40–90 minutes total — batch it but no overnight run needed. On CPU (e.g., Antigravity on a non-GPU machine), the same batch is more like 6–12 hours — still batch it, but plan around running it unattended, and checkpoint per-subject so a partial run isn't wasted if it's interrupted.
   - Process the whole session folder or an input text file listing all scans in one call, not one `mri_synthseg` invocation per scan — the tool's own documentation notes per-call startup overhead is otherwise wasted repeatedly.
4. **Extract regional volumes.** Already handled by the `--vol` output — load it directly rather than re-deriving from the label map. Normalize by eTIV (already provided in the CSV) to control for head-size differences.

**Output:** `regional_volumes.csv` — one row per (subject, visit), one column per brain region, plus normalized (eTIV-adjusted) versions.

**Acceptance criteria:** spot-check 3–5 subjects' SynthSeg output visually (overlay `synthseg_out.nii.gz` on the T1 slice) before trusting the volumes. If `--parc`'s cortical labels don't obviously correspond to a standard atlas naming convention you recognize, check the tool's label lookup table before assuming a match in Phase 3/4 — don't guess at the correspondence.

---

## 4. Phase 2 — Classification: Volumetric ML (primary) + Transfer-Learning CNN (secondary)

**This phase is restructured from a straight "train a CNN" plan.** Published, dataset-matched evidence says a from-scratch CNN on ~150 subjects is the wrong default: Wen et al. (2020, *Medical Image Analysis*) found a plain linear SVM on volumetric features competitive with CNN architectures on this task, and dedicated transfer-learning work on OASIS itself (Yagis et al., 2021, *Scientific Reports*) used pretrained ImageNet backbones specifically *because* the subject count wasn't enough to train from scratch. Follow that precedent. **This holds regardless of whether you're running on Colab's GPU or Antigravity's CPU** — it's a data-size argument, not a compute-availability one. A GPU makes 4b faster to iterate on; it doesn't make training from scratch on 150 subjects a better idea.

### 4a. Primary track: classical ML on volumetric + clinical features (build this first)

- **Task:** binary classification, Nondemented vs. Demented, using CDR-derived labels (CDR > 0 = Demented).
- **Features:** Phase 1's per-region volumes (eTIV-normalized) + clinical/demographic columns (Age, Sex, Education, SES).
- **Models:** logistic regression, linear SVM, random forest/gradient boosting — all trivial to train on CPU, and this is the credible, literature-supported baseline, not a placeholder to beat.
- **This is the model you should trust and report as your primary result.**

### 4b. Secondary track: transfer-learning CNN (build only if 4a is solid and time remains)

- **Do not train from scratch.** Use a pretrained 2D backbone (e.g., ImageNet-pretrained ResNet-18 or VGG16 from `torchvision`) as a frozen or lightly fine-tuned feature extractor, following the same rationale as Yagis et al. (2021): too few subjects to learn good filters from scratch, but a pretrained backbone plus a small trained classification head is tractable on CPU.
- **Slice selection:** extract a fixed set of informative axial/coronal slices per volume (e.g., highest-entropy slices, or a fixed anatomical range around mid-brain) rather than all slices — reduces compute and avoids diluting signal with uninformative edge slices.
- **Hard requirement — subject-level split only.** Yagis et al. (2021) measured slice-level cross-validation inflating *OASIS-specific* reported accuracy by roughly 30 percentage points (their VGG16 experiment: ~66% honest subject-level split vs. ~97% leaked slice-level split — and a randomly-relabeled control still hit ~93–96% "accuracy" under a slice-level split purely from leakage). If a subject's slices ever appear in both train and val/test, the resulting number is not real. Use `GroupKFold` (or equivalent) keyed on subject ID, never a plain random split on pooled slices.
- **Realistic accuracy expectation:** roughly 60–75% subject-level accuracy is in line with published, leakage-free results at this sample size — treat anything much higher on your own held-out set as a sign to check for leakage before celebrating it, not as a target to hit.
- **Runtime budget:** on Colab's GPU, a training run should complete in minutes; on a CPU-only environment (Antigravity without a GPU), design for well under an hour (small batch size, few epochs, early stopping on validation loss). Either way, log wall-clock time so it's clear which environment produced which numbers.

**Output:** 4a's classical model as the primary reported result; 4b's model (if built) reported alongside it with its own honestly-measured accuracy, explicitly framed as exploratory. Either model's learned/derived features can optionally feed Phase 5/6.

---

## 5. Phase 3 — Network Diffusion Model (growth/spread simulation)

Replaces the Fisher-Kolmogorov reaction-diffusion PDE + DTI anisotropy tensor with the **Network Diffusion Model (NDM)**, Raj, Kuceyeski & Weiner (2012, *Neuron* 73(6):1204–1215) — the established, directly-analogous framework in the dementia literature (pathology spreading along the brain's connectivity graph, mathematically the same diffusion-operator family as reaction-diffusion on a tissue tensor).

### 5.1 Build the connectome graph

You need a brain connectivity graph `H` (the NDM's counterpart to the DTI-derived white-matter tensor in the original tumor model).

**Use a template/healthy-reference structural connectome — this is not a compromise, it's what the field itself does for exactly this reason.** The original Raj et al. (2012) paper built its own connectome from only 14 healthy-subject tractography scans, not from the patients being modeled. More directly on point: Raj et al. (2015, *Cell Reports*, "Network diffusion model of progression predicts longitudinal patterns of atrophy and metabolism in Alzheimer's Disease") explicitly used a **healthy reference connectome rather than individual patient connectomes** for their ADNI predictions, stating plainly that ADNI didn't contain diffusion MRI — the identical situation OASIS-2 is in here — and found that connectome variability had only minor influence on the model's predictions. Use a publicly available averaged structural connectome template (e.g., a Desikan-Killiany-parcellated healthy-population DTI-derived connectivity matrix such as the Budapest Reference Connectome) and cite this precedent directly in your report rather than presenting it as a limitation to apologize for.

- **Fully self-contained fallback** (only if sourcing a template connectome proves to be a real obstacle): approximate connectivity from atlas region centroids — a graph where edge weight is inversely proportional to Euclidean distance between region centroids (from `nilearn` atlas metadata). This is a much weaker substitute with no literature precedent behind it specifically — use it only as a last resort, and say so if you do.
- Either way: compute the **graph Laplacian** `L = D - A` from the weighted adjacency matrix `A` (D = degree matrix), and confirm the connectome's region labels can be mapped onto SynthSeg's output regions before going further — a silent region-name mismatch here will quietly corrupt everything downstream.

### 5.2 Define the diffusion state (age-normalized, not a raw z-score)

For each subject/visit, convert regional volumes into a **regional pathology score** `x`. Don't just z-score against the raw nondemented-subject distribution — that conflates ordinary age-related volume loss with disease-driven atrophy. Instead:

1. Within the stably-Nondemented subjects, fit a simple linear regression of each region's (eTIV-normalized) volume against age.
2. For every subject/visit, compute the **residual** from that age-expected volume, then z-score the residual.

This age-adjusted residual score (a standard approach in the structural-imaging literature, sometimes called a W-score) is what plays the role the tumor-cell-density field `c(x,t)` plays in the original equation — higher deviation = higher "pathology."

### 5.3 The model

Closed-form linear diffusion on the graph, per Raj et al. (2012):

```
x(t) = exp(-β · L · t) · x(0)
```

where `β` is a scalar diffusivity rate (the direct analog of the tumor model's proliferation/diffusion constants) and `t` is time in years since baseline visit. Implement the matrix exponential via eigen-decomposition of `L` rather than repeated dense `expm` calls — you'll be evaluating this at many different `t` values per subject during fitting and validation, and eigen-decomposition (compute once, then evaluate `exp(-β·λ_i·t)` per eigenvalue) is both faster and the numerically standard way this model is implemented in the literature.

### 5.4 Parameter fitting (the "inverse problem" stage)

For each subject with 2+ visits: fit `β` by minimizing the squared error between the model's predicted `x(t_2)` and the subject's actual observed `x` at their second visit, using `scipy.optimize`. This is the scaled-down, CPU-tractable version of the adjoint-method/PINN parameter estimation described in the source document — same goal (estimate a patient-specific spread-rate constant from sparse longitudinal data), much simpler solver.

### 5.5 Validation

- Hold out a subset of subjects; fit β on visit 1→2, then predict visit 3 (for subjects with 3+ visits) and compare to actual.
- Report correlation / MSE between predicted and actual regional atrophy patterns. For context (not a target to match — their cohort, connectome, and parcellation all differ from this setup), Raj et al. (2015) reported a Pearson correlation around R≈0.93 between predicted and measured end-of-study atrophy on ADNI; be explicit that this OASIS-2 implementation is a proof-of-concept validation on a much smaller, differently-processed dataset, not a replication.

**Output:** per-subject fitted β, predicted future regional atrophy trajectories extending past the last observed visit.

---

## 6. Phase 4 — Neuroanatomical Mapping & Cognitive Deficit Prediction

Replaces the eloquent-cortex/aphasia mapping section — same mechanism, different target regions and deficits.

1. Build a region → cognitive-domain lookup table grounded in standard neuropsychology, e.g.:
   - Hippocampus, entorhinal cortex → episodic memory
   - Frontal lobe (esp. dorsolateral prefrontal) → executive function
   - Temporal/perisylvian regions → language
   - Parietal lobe → visuospatial function
   - Amygdala → emotional/behavioral regulation
2. For a subject's predicted future atrophy vector (from Phase 3), compute a weighted deficit score per cognitive domain (sum of region atrophy × functional relevance weight for that region).
3. **Validate against real outcomes already in the dataset**: correlate predicted deficit severity against actual MMSE trajectory and CDR change for that subject. This is a genuinely checkable prediction, unlike the tumor document's symptom section which has no ground truth to validate against in the source data.

**Output:** per-subject predicted cognitive-domain deficit profile + a report of correlation against real MMSE/CDR outcomes.

---

## 7. Phase 5 — Clinical/Demographic Risk Profiling (proxy for radiogenomics)

Be explicit in code comments and the final report: **this is not a molecular/genomic model** — OASIS-2 has no genotype data (no APOE, unlike OASIS-3). It's a structured-clinical-feature risk stratifier, playing an analogous "combine non-imaging biological/risk info with imaging to refine the prediction" role.

- Features: Age, Sex, Education, SES, eTIV, nWBV, plus Phase 1 regional volumes and optionally Phase 2's learned feature vector.
- Target: risk of decline (e.g., predicted future CDR increase, or the Phase 3 fitted β as a continuous "aggressiveness" target).
- Model: gradient boosting or logistic regression — trivial on CPU, don't over-engineer this stage.
- Report feature importances — this is the direct analog of "which imaging/molecular signature drives aggressive vs. indolent behavior" in the original, just with clinical variables instead of IDH/MGMT status.

**Optional stretch, only if time allows:** OASIS-3 does include APOE genotype and could be added as a genuinely molecular extension of this stage — call this out as future work rather than attempting it under the current CPU/time budget.

---

## 8. Phase 6 — Sustained Progression vs. Fluctuation Classification (proxy for pseudoprogression vs. true progression)

**Refined from the original framing.** Using OASIS-2's final `Converted`/`Nondemented`/`Demented` group label alone treats this as an endpoint classification problem, which is a shallower analog than intended — TP vs. PsP is fundamentally about *whether a trajectory reflects real change or a look-alike fluctuation*, not about a single end-state label. OASIS-2's longitudinal CSV has **per-visit CDR scores**, which supports the closer analog directly:

- **Define the target from the CDR trajectory itself**, not the pre-computed Group column: a subject showing monotonic, non-decreasing CDR across visits that crosses into dementia territory is a "true/sustained progression" case; a subject whose CDR fluctuates (e.g., 0 → 0.5 → 0, or any non-monotonic pattern) without net sustained increase is the "fluctuation/look-alike" case. This is a tighter structural match to the original TP-vs-PsP problem (can two similar snapshots represent different underlying dynamics?) than the static Group label, and it's checkable directly from data already in the dataset.
- **Features:** longitudinal *change* (delta) in regional volumes between visits, delta MMSE, the Phase 3 fitted diffusivity β (rate of spread is plausibly more diagnostic than a single snapshot — this mirrors how the source document emphasizes DTI/ADC *dynamics* over static imaging for TP/PsP discrimination), plus demographics.
- **Class imbalance and small-n are severe here** (the sustained-progression group will be a small fraction of ~150 subjects). Use leave-one-subject-out or repeated stratified k-fold cross-validation, class-weighted loss, and report sensitivity/specificity with bootstrap confidence intervals rather than a single point accuracy. Avoid hyperparameter tuning against the same small sample — at this n, tuning is very likely to overfit the evaluation itself, not just the model. Do not claim clinical-grade performance from this sample size; report it as exploratory pattern-finding.
- **Output:** classifier + an honest limitations paragraph for the report (small-n, proxy nature, no independent replication cohort).

---

## 9. Phase 7 — Integration & Report Assembly

1. One end-to-end script/notebook that, given a subject ID, runs Phases 1→6 and outputs:
   - Regional volumes across visits
   - Predicted future atrophy map (Phase 3) rendered as a glass-brain or region-bar-chart via `nilearn` plotting
   - Predicted cognitive-domain deficit profile (Phase 4) vs. actual MMSE/CDR trend
   - Risk score (Phase 5) and sustained-progression risk probability (Phase 6)
2. Aggregate evaluation notebook: dataset-wide metrics for the Phase 2 classifiers (4a primary, 4b secondary), Phase 3 validation correlation, Phase 5 feature importances, Phase 6 CV metrics.
3. **Write a short "Methods Adaptation" section for the final report** explicitly documenting the pivot from the tumor-focused literature review to this OASIS-2 implementation and why (dataset/compute constraints) — this is good academic practice and mirrors the honesty of this plan itself.

---

## 10. Suggested Repository Structure

```
project/
├── data/
│   ├── raw/                  # original OASIS-2 download, untouched
│   └── processed/            # NIfTI conversions, SynthSeg outputs
├── notebooks/
│   ├── 00_data_audit.ipynb
│   ├── 01_segmentation.ipynb
│   ├── 02a_classical_ml.ipynb        # primary classification track
│   ├── 02b_transfer_cnn.ipynb        # secondary/exploratory track
│   ├── 03_network_diffusion.ipynb
│   ├── 04_symptom_mapping.ipynb
│   ├── 05_risk_profiling.ipynb
│   ├── 06_progression_classification.ipynb
│   └── 07_integration_report.ipynb
├── src/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── classical_model.py    # primary: volumetric + clinical features
│   ├── cnn_model.py           # secondary: transfer-learning CNN
│   ├── ndm.py                 # network diffusion model + β fitting
│   ├── symptom_mapping.py
│   ├── risk_model.py
│   ├── progression_model.py   # trajectory-based sustained vs. fluctuation classifier
│   └── utils.py
├── models/                   # saved checkpoints
├── outputs/                  # figures, per-subject reports, metrics CSVs
├── requirements.txt
└── implementation.md         # this file
```

---

## 11. Known Limitations (carry these into the final report, don't bury them)

- T1-only data means no true multi-modal segmentation and no DTI-derived anisotropy — the NDM's connectome is a template/population-average, not patient-specific white matter architecture (though this is literature-standard practice under the same constraint, see §5.1).
- No genomic data — Phase 5 is a clinical proxy, explicitly not radiogenomics.
- No treatment/radiotherapy records — irrelevant here since OASIS-2 isn't an oncology dataset, but worth stating why that section of the original literature review has no analog at all beyond Phase 6's trajectory-based framing.
- Phase 6's sustained-progression class will be small relative to the full cohort — too small for a robust classifier on its own; treat results as exploratory pattern-finding, not a validated diagnostic tool.
- Compute-driven metric ceiling in Phase 2 comes from sample size, not hardware — expect modest, literature-consistent (not state-of-the-art) classification metrics whether run on Colab's GPU or Antigravity's CPU (roughly 60–75% subject-level accuracy is realistic; treat anything much higher as a leakage red flag, see §4b).
- Colab's ephemeral runtime is a real failure mode if not handled: unsaved intermediate outputs (e.g., a partially-completed SynthSeg batch) are lost on disconnect. §1a's checkpoint-to-Drive guidance isn't optional housekeeping — skipping it risks losing hours of Phase 1 processing to a routine idle timeout.
- SynthSeg's `--parc` cortical labels need to be confirmed against the connectome template's region naming before Phase 3/4 — a silent mismatch here would quietly break both stages without an obvious error.
- Phase 0's directory-structure assumptions (RAW/PROCESSED/FSL_SEG/SUBJ_111/T88_111) are documented for OASIS-1's public release; confirm OASIS-2 matches before relying on it, rather than assuming.

---

## 12. References

- Raj, A., Kuceyeski, A., & Weiner, M. (2012). A network diffusion model of disease progression in dementia. *Neuron*, 73(6), 1204–1215.
- Raj, A., LoCastro, E., Kuceyeski, A., Tosun, D., Relkin, N., & Weiner, M. (2015). Network diffusion model of progression predicts longitudinal patterns of atrophy and metabolism in Alzheimer's disease. *Cell Reports*, 10(3), 359–369.
- Wen, J., Thibeau-Sutre, E., Diaz-Melo, M., et al. (2020). Convolutional neural networks for classification of Alzheimer's disease: Overview and reproducible evaluation. *Medical Image Analysis*, 63, 101694.
- Yagis, E., Workalemahu Atnafu, S., García Seco de Herrera, A., Marzi, C., Scheda, R., Giannelli, M., Tessa, C., Citi, L., & Diciotti, S. (2021). Effect of data leakage in brain MRI classification using 2D convolutional neural networks. *Scientific Reports*, 11, 22544.
- Billot, B., Greve, D. N., Puonti, O., et al. (2023). SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining. *Medical Image Analysis*.
- Marcus, D. S., Fotenos, A. F., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2010). Open Access Series of Imaging Studies (OASIS-2): Longitudinal MRI data in nondemented and demented older adults.
