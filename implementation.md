# AD-CELLTWIN (OASIS-2 Scope): An Interpretable Regional Digital Twin for Mapping and Predicting Alzheimer's Disease Progression

## 0. Scope note — mapping the digital-twin concept onto OASIS-2

The digital-twin architecture above assumes a multi-modal dataset: structural MRI, tau/amyloid PET, DTI, fMRI, CSF/blood biomarkers, genetics, and single-cell/spatial transcriptomics. **OASIS-2 provides none of the molecular or connectomic layers** — it is a longitudinal structural MRI (T1-weighted) dataset with demographics, CDR, MMSE, and derived volumetrics (eTIV, nWBV, ASF) across 2–5 visits per subject.

Rather than pretend OASIS-2 supports cellular-level or PET-based prediction, this plan does two honest things:

1. **Builds the full ambitious architecture as a target design** (Sections 2–13), so the system is forward-compatible with richer data (ADNI, PET, spatial transcriptomics) later.
2. **Defines exactly which layers are actually implementable on OASIS-2 today** (marked ✅ *implementable now*) versus which are **stubbed with synthetic/placeholder interfaces** (marked 🔶 *interface only — requires future data*) so the codebase is honest about what it's actually predicting.

The practical unit of analysis on OASIS-2 is **brain regions / tissue compartments derived from structural MRI**, not individual cells. We call this a **Regional Digital Twin** — Level 1–2 of the five-level hierarchy in Section 12 — with Levels 3–5 (cell type, cellular state, spatial neighborhood) built as architectural placeholders for when snRNA-seq/spatial transcriptomics or ADNI-style multimodal data is integrated.

---

## 1. Central research question

Can we construct a personalized, interpretable digital twin from a subject's longitudinal structural MRI (OASIS-2) that can:

1. Locate currently affected/atrophied brain regions ✅
2. Quantify regional vulnerability relative to the subject's own baseline and to the cohort ✅
3. Model the direction and rate of structural change across visits ✅
4. Predict which regions are likely to show significant volume/atrophy change at the next visit ✅
5. Estimate a plausible time horizon for that change, conditioned on visit spacing in the data ✅
6. Explain every prediction in terms of the actual input features (not a black box) ✅
7. Provide a defined extension interface for cellular/molecular layers once such data exists 🔶

---

## 2. System overview

```
                    PERSONALIZED REGIONAL DIGITAL TWIN (OASIS-2 scope)
                              │
             ┌────────────────┼────────────────┐
             ↓                ↓                ↓
        3D Anatomy       Region Graph      Molecular Layer
             │                │                │
          T1 MRI          structural       🔶 placeholder
       (preprocessed)     proxy graph      (snRNA-seq / spatial
             │            (co-atrophy         transcriptomics /
             │             correlation)        PET — future work)
             └────────────────┼────────────────┘
                              ↓
                    Structural Pathology Proxy
                     (regional atrophy, nWBV,
                      CDR/MMSE-linked decline)
                              │
                    ┌─────────┴─────────┐
                    ↓                   ↓
              Cross-sectional      Longitudinal
                 severity            trajectory
                    │                   │
                    └─────────┬─────────┘
                              ↓
                    Regional vulnerability score
                              ↓
                    Propagation graph (region→region)
                              ↓
                  FUTURE STATE PREDICTION + INTERPRETABLE REPORT
```

This keeps the same conceptual skeleton as the original digital-twin proposal (anatomy → network → pathology → vulnerability → propagation → prediction → explanation), but every arrow above is backed by a feature that actually exists in OASIS-2, plus one clearly marked extension point for the molecular layer.

---

## 3. What gets mapped, at what resolution

### Level 1 — Whole-brain regions ✅ *implementable now*

From skull-stripped, registered T1 volumes, extract volumetric/morphometric features for at least:

- Hippocampus (L/R)
- Entorhinal cortex (L/R) — proxy via medial temporal lobe ROI if full parcellation unavailable
- Temporal cortex (L/R)
- Parietal cortex (L/R)
- Frontal cortex (L/R)
- Posterior cingulate / precuneus
- Amygdala (L/R)
- Thalamus (L/R)
- Lateral ventricles (inverse proxy — enlargement signals atrophy)
- Whole-brain (nWBV, eTIV — already provided in OASIS-2 tabular data)

**Method:** since OASIS-2 does not ship with a FreeSurfer parcellation, this pipeline uses an atlas-based ROI segmentation (e.g., a standard MNI-registered atlas such as Harvard-Oxford or AAL, applied after affine/nonlinear registration to template space) to derive per-region volumes. This is the single biggest new engineering component relative to the current CNN-classification pipeline and is described fully in Phase 2 below.

### Level 2 — Region network (structural proxy) ✅ *implementable now, proxy only*

OASIS-2 has no DTI/fMRI, so true structural/functional connectivity is unavailable. Instead, build a **cohort-derived co-atrophy network**: an edge between region *i* and region *j* is weighted by the correlation of their volumetric decline across subjects and visits. This is a well-established proxy in structural-imaging AD literature (co-atrophy networks approximate — but do not equal — true connectivity) and must be labeled as such everywhere it's surfaced to a user.

🔶 *Interface only, future work:* true structural connectivity (DTI-derived) and functional connectivity (fMRI-derived) network layers, and default-mode/salience/executive network overlays, are stubbed as optional inputs to the graph builder (Phase 3) so the architecture doesn't need to change when DTI/fMRI data is added later.

### Level 3 — Cell populations 🔶 *interface only — not available in OASIS-2*

OASIS-2 contains no single-cell or histological data. This level is represented purely as a **schema stub**: a `CellPopulationLayer` interface that downstream modules can query, which today always returns "not available" but defines the exact fields (`cell_type`, `state`, `gene_expression_vector`, `pathology_exposure`, `vulnerability_score`) so a future snRNA-seq/spatial-transcriptomics integration slots in without refactoring Levels 1–2 or the propagation/prediction modules.

### Level 4 — Individual cellular states 🔶 *interface only*

Same treatment as Level 3 — defined in the schema, not populated.

---

## 4. The regional vulnerability record (this project's version of the "cellular disease map")

For every region *r* at every visit *t*, compute a structured record:

```
R_i = (x, y, z, name, V_t, ΔV, nWBV_t, CDR_t, MMSE_t, age_t, network_centrality, vulnerability_score)
```

where:
- `(x, y, z)` = ROI centroid in template space
- `V_t` = normalized regional volume at visit *t* (normalized by eTIV to control for head size)
- `ΔV` = volume change vs. subject's own baseline visit
- `nWBV_t`, `CDR_t`, `MMSE_t`, `age_t` = subject-level clinical/derived covariates at visit *t* (from OASIS-2 tabular data)
- `network_centrality` = the region's weighted degree in the co-atrophy graph (Level 2)
- `vulnerability_score` = model output, defined below

Assign a discrete status band for interpretability, mirroring the original 5-tier scheme but redefined on measurable structural criteria:

| Band | Criterion (subject-relative, within-cohort z-scored) |
|---|---|
| GREEN — apparently stable | ΔV within ±0.5 SD of the subject's own trajectory and cohort norm |
| YELLOW — vulnerable | ΔV between 0.5–1.5 SD below expected, OR high network centrality + neighboring regions already YELLOW/ORANGE |
| ORANGE — early structural change | ΔV between 1.5–2.5 SD below expected, consistent across ≥2 visits |
| RED — strongly affected | ΔV > 2.5 SD below expected, consistent across ≥2 visits |
| BLACK — severe/advanced | ΔV > 2.5 SD **and** subject CDR ≥ 1 **and** region among the top-3 most atrophied for that subject |

As in the original proposal, this pipeline outputs a **probability**, not a deterministic label:

```
P(region r shows clinically meaningful further atrophy at next visit | current trajectory, network position, clinical covariates)
```

---

## 5. Propagation graph (region → region)

Build a directed, weighted propagation graph per subject:

```
Region A (baseline atrophy)
        │
        ↓
 ┌──────────────┐
 │ Region B     │
 │  affected    │
 └──────────────┘
   ↙          ↘
  ↓            ↓
Region C     Region D
 W=0.71        W=0.54
  │
  ↓
Region E
 W=0.29
```

Edge weight:

```
W_ij = f( co_atrophy_correlation_ij,   -- Level 2 proxy network
          spatial_adjacency_ij,        -- Euclidean/anatomical distance
          shared_pathology_exposure_ij,-- both regions' ΔV trend
          network_centrality_j )       -- future work: molecular_similarity_ij (🔶)
```

`f` is learned (gradient-boosted trees or a small GNN edge-scoring head — see Phase 4) rather than hand-specified, using historical visit-to-visit transitions in the OASIS-2 longitudinal subjects as supervision: "given B was already affected at visit *t*, was C/D/E affected at visit *t+1*?"

This directly answers the "direction of spread" objective from the original proposal, at region-level resolution instead of cell-level.

---

## 6. Learning progression instead of classification

Reframe training away from static "Demented vs. Nondemented" classification and toward **trajectory modeling**, using the fact that OASIS-2 subjects have repeat visits:

```
Healthy/Stable
     ↓
Subtle regional decline (subclinical)
     ↓
Early regional pathology (CDR 0.5)
     ↓
Established regional pathology (CDR 1)
     ↓
Advanced pathology (CDR 2+)
```

For each visit pair *(t, t+1)* per subject, compute:

```
ΔRegionState = RegionState_{t+1} − RegionState_t
```

and train the model to answer: *given the region-level state and network position at visit t, what is ΔRegionState at t+1?* This is a genuine improvement over the current pipeline's cross-sectional CNN classifier (Phase/CNN track in the existing notebook), because it uses the longitudinal structure of OASIS-2 that is currently underexploited.

---

## 7. Digital twin simulation and counterfactuals

Once a subject's regional twin is built:

**Scenario A — natural trajectory (no intervention modeled).** Roll the learned transition model forward for *N* simulated visit-intervals using the subject's own historical rate of change, producing a projected regional-vulnerability map with uncertainty bands (not a single point estimate).

**Scenario B — counterfactual sensitivity analysis** 🔶 *simplified version, not a treatment-effect model*. Since OASIS-2 has no treatment/intervention data, we cannot simulate "with treatment" in the clinical sense the original proposal describes. Instead, this pipeline supports a defensible, weaker counterfactual: *"if this subject's atrophy rate matched the cohort's slowest-declining quartile instead of their observed rate, how would the projected vulnerability map change?"* This is explicitly labeled as a **hypothesis-generating sensitivity analysis**, not a treatment simulation, to avoid overclaiming.

---

## 8. Interpretability layer ("interpretability for all")

Every prediction ships with a decomposed explanation, generated via feature attribution (SHAP or integrated gradients, depending on the model — see Phase 5) rather than a bare probability:

```
Prediction: Hippocampus (L), Subject OAS2_00XX
Predicted risk of further atrophy at next visit: 74%

Why?
                    74% RISK
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
  Trajectory       Network         Clinical
   slope             centrality      covariates
   +0.29             +0.21           +0.15
   (CDR trend, age)
```

Rendered in plain language:

> "The predicted risk is primarily driven by this region's own declining volume trend across prior visits, its structural proximity (in the co-atrophy network) to already-affected regions, and the subject's rising CDR score."

### Audience-specific views (retained from the original design, unchanged in spirit)

| Audience | What they see |
|---|---|
| Patient/family (plain-language summary) | "Your memory-related brain regions show a moderate predicted risk of further change. This is a probability, not a certainty." |
| Clinician | Regional risk map, feature contributions, uncertainty interval, projected trajectory, CDR/MMSE context |
| Researcher | Full co-atrophy propagation graph, region embeddings, model calibration curves, ablation results |
| ML engineer | Architecture diagram, per-fold metrics, feature importances, calibration, confidence intervals |

---

## 9. Data actually used (OASIS-2) vs. data the architecture reserves room for

| Data | Status in OASIS-2 | Role |
|---|---|---|
| T1 structural MRI | ✅ present | 3D anatomy, ROI volumetrics |
| CDR | ✅ present | Clinical staging label / supervision signal |
| MMSE | ✅ present | Clinical covariate |
| Age, sex, education, SES | ✅ present | Covariates |
| nWBV, eTIV, ASF | ✅ present (derived, already in tabular data) | Global atrophy proxy, normalization |
| Longitudinal visits (2–5/subject) | ✅ present | Enables trajectory modeling (Section 6) |
| FDG-PET / amyloid PET / tau PET | 🔶 not in OASIS-2 | Reserved interface for future multimodal fusion |
| DTI / fMRI | 🔶 not in OASIS-2 | Reserved interface for true connectivity layer |
| CSF/blood biomarkers, genetics | 🔶 not in OASIS-2 | Reserved interface |
| snRNA-seq / snATAC-seq / spatial transcriptomics | 🔶 not in OASIS-2 | Reserved interface (Level 3–4 schema stub) |
| Histopathology | 🔶 not in OASIS-2 | Reserved interface (validation layer only) |

---

## 10. Proposed architecture

```
T1 MRI (multi-visit) ──────────────┐
CDR / MMSE / demographics ─────────┤
nWBV / eTIV / ASF ──────────────────┤
🔶 PET / DTI / fMRI (future) ───────┤
🔶 genetics / CSF (future) ─────────┤
                                    ↓
                          PREPROCESSING + ROI
                             SEGMENTATION
                                    ↓
                          3D REGION GRAPH BUILDER
                        (nodes = ROIs, edges = co-atrophy
                         proxy network, Level 2)
                                    ↓
                    🔶 CELL/MOLECULAR LAYER (schema stub, Levels 3-4)
                                    ↓
                    REGIONAL DIGITAL TWIN STATE
                                    ↓
       ┌────────────┼─────────────┐
       ↓            ↓             ↓
  Region state   Propagation   Trajectory
   prediction      model        model
  (GBM / GNN)    (edge scorer)  (temporal)
       │            │             │
       └────────────┼─────────────┘
                    ↓
             FUTURE SIMULATOR
              (forward rollout,
               uncertainty bands)
                    ↓
       ┌────────────┼────────────┐
       ↓            ↓            ↓
   Risk map    Time-horizon   Direction
  (per region)   estimate    (propagation graph)
       │            │            │
       └────────────┼────────────┘
                    ↓
      INTERPRETABLE, AUDIENCE-AWARE REPORT
     (SHAP attributions + plain-language layer)
```

A graph neural network (region graph) combined with a temporal model (per-subject visit sequence) and a gradient-boosted baseline for calibration comparison is the recommended concrete stack — see Phase 4 for the specific model choices and why a GNN+temporal hybrid beats extending the current CNN-only track.

---

## 11. Implementation phases

This replaces/extends the existing six-phase Colab pipeline. Phases 0–3 map closely to your current notebook (preprocessing → feature extraction); Phases 4–9 are new.

### Phase 0 — Environment & data audit
- Re-confirm Colab namespace-only execution constraint (no `.py` files written to disk; all modules share the notebook namespace; tqdm progress bars throughout) — unchanged from current setup.
- Load OASIS-2 tabular data (subject demographics, CDR, MMSE, nWBV, eTIV, ASF, visit number, days-from-baseline) and the T1 MRI volumes.
- Audit: subjects with ≥2 visits (required for trajectory modeling), missing-data patterns, CDR-transition subjects (subjects whose CDR changes across visits — these are the highest-value training examples for Phase 4/6).
- Output: a per-subject visit manifest (`subject_id, visit_id, days_from_baseline, CDR, MMSE, nWBV, has_mri`).

### Phase 1 — Preprocessing (extends current pipeline)
- Reuse existing skull-stripping / bias-field correction / intensity normalization steps.
- **New:** rigid + affine (and optionally nonlinear) registration of every subject/visit T1 volume to a common template space (e.g., MNI152), which is required before atlas-based ROI extraction in Phase 2. This is the key new preprocessing step not present in the current CNN-classification pipeline.
- Guard against the malformed-4D-volume issue already fixed in the current notebook (squeeze + `ndim != 3` check) — apply the same guard here since it affects any step that loads raw NIfTI volumes.
- QC step: flag any registration with abnormally low similarity metric (e.g., normalized mutual information below a threshold) for manual review rather than silently including a bad registration.

### Phase 2 — Regional feature extraction (new)
- Apply the template-space atlas (Harvard-Oxford or AAL, chosen for license/availability) to each registered volume to extract per-ROI volumes for the regions listed in Section 3, Level 1.
- Normalize each ROI volume by eTIV to remove head-size confounds.
- Compute `ΔV` per region per subject relative to that subject's baseline visit.
- Output: a long-format table `(subject_id, visit_id, region_name, volume_norm, delta_v_from_baseline)` — this becomes the core feature store for everything downstream.
- Sanity checks: total of all ROI volumes should track nWBV/eTIV trends already in the OASIS-2 tabular data; use this as an independent cross-check on segmentation quality.

### Phase 3 — Region graph construction (new)
- Compute the cohort-level co-atrophy correlation matrix (Level 2 proxy network) from the Phase 2 feature store: correlate `ΔV` trajectories across all subject-visit pairs, region by region.
- Add spatial-adjacency edges from ROI centroid distances (Section 5) as a second edge-weight component.
- Build one shared population-level graph (used as a structural prior) and, per subject, a personalized node-feature overlay (that subject's own `V_t`, `ΔV`, CDR, MMSE) on top of the shared graph topology.
- Define and implement the `CellPopulationLayer` schema stub (Level 3–4, Section 3) as an interface class that returns "not available" — this is intentionally lightweight in this phase, just enough to prove the rest of the pipeline can call it without special-casing.

### Phase 4 — Vulnerability & propagation modeling (new)
- **Vulnerability score model:** train a model to predict `P(region shows clinically meaningful further atrophy at next visit)` using node features (own trajectory, network centrality, clinical covariates) — start with gradient-boosted trees as an interpretable, fast-to-calibrate baseline, then compare against a GNN variant that consumes the full region graph from Phase 3.
- **Propagation edge scorer:** train the `W_ij` function from Section 5 using historical visit-to-visit transitions as supervision (did region B's decline precede region C's decline in subjects where both eventually declined?).
- Both models are trained with subject-level, not visit-level, cross-validation splits (a subject's visits must never span train/test) — mirroring the leakage discipline already established in your existing CV folds.
- Compare against the existing CNN slice-classification track as a baseline: does region-level trajectory modeling out-predict the current cross-sectional CNN on held-out future-visit CDR/MMSE, and does it do so with usable per-region interpretability that the CNN cannot provide?

### Phase 5 — Interpretability layer (new)
- Attach SHAP (for the GBM path) or integrated-gradients/GNNExplainer-style attribution (for the GNN path) to every vulnerability-score prediction.
- Build the plain-language template renderer described in Section 8 that turns raw attributions into the audience-specific summaries (patient/family, clinician, researcher, ML engineer).
- Validate that attributions are stable (similar inputs → similar attributions) as a basic sanity/robustness check before trusting them for the report layer.

### Phase 6 — Trajectory / digital-twin simulation (new)
- Implement the forward-rollout simulator (Section 7, Scenario A): given a subject's twin state at their last observed visit, roll forward using the learned transition model for *N* simulated intervals, propagating uncertainty (e.g., via ensembling or quantile regression) rather than emitting a single deterministic future map.
- Implement the sensitivity-analysis counterfactual (Section 7, Scenario B) and clearly label it as hypothesis-generating, not a treatment-effect estimate.
- Output: per-subject projected regional-vulnerability maps at future simulated visit-intervals, with confidence bands.

### Phase 7 — Validation (extends current CV setup)
- Primary validation: for subjects with ≥3 visits, train on visits 1..t and evaluate the Phase 6 rollout's predicted regional states against the actually-observed visit t+1 (and t+2 where available) — this is the direct analog of the original proposal's "baseline vs. 12/24/36-month" validation design, adapted to whatever visit spacing OASIS-2 actually has (report the real spacing distribution; do not assume fixed 12-month intervals).
- Metrics: regional prediction accuracy/AUC, CDR/MMSE-transition prediction accuracy, calibration (reliability diagrams), uncertainty coverage (are the confidence bands actually calibrated?), and generalization across the existing 5-fold subject-level CV.
- Explicitly report where the model fails (e.g., subjects with irregular visit spacing, subjects with only 2 visits) rather than only reporting aggregate metrics.

### Phase 8 — Reporting & interface stubs for future data
- Finalize the audience-specific report templates (Section 8) as the primary user-facing deliverable.
- Document the 🔶 interface stubs (PET, DTI/fMRI, genetics, snRNA-seq/spatial transcriptomics, histopathology validation) as a clearly separated "Future Work / Extension Points" appendix in the notebook, including the exact schema each stub expects, so integrating ADNI-style multimodal data later is a data-plumbing task, not an architecture rewrite.
- Package final outputs: per-subject regional digital twin state, propagation graph, projected trajectories, and interpretable report — consistent with the existing notebook's deliverable style (all in-notebook, tqdm-tracked, no external `.py` files).

### Phase 9 — Limitations & scientific safeguards (documentation, not code)
- State explicitly, in the notebook and any derived report: this system predicts **region-level structural vulnerability**, not individual-cell fate; the co-atrophy network is a **statistical proxy**, not measured connectivity; the counterfactual analysis is a **sensitivity analysis**, not a treatment simulation; and the molecular/cellular layers are **unpopulated interface stubs** pending future multimodal data.
- This mirrors the "crucial scientific safeguard" in the original proposal (Section 12 there) and is treated as a first-class deliverable, not an afterthought — the same posture as the honest scope note at the top of this document.

---

## 12. Prediction resolution hierarchy (target design vs. current implementation)

| Level | Description | Status on OASIS-2 |
|---|---|---|
| 1 | Brain region | ✅ implemented (Phases 2–4) |
| 2 | Region network (co-atrophy proxy) | ✅ implemented, proxy only (Phase 3) |
| 3 | Cell type | 🔶 schema stub only (Phase 3) |
| 4 | Cellular state | 🔶 schema stub only (Phase 3) |
| 5 | Spatial cellular neighborhood | 🔶 not started — requires spatial transcriptomics |

---

## 13. What's genuinely novel in this OASIS-2 version relative to the current notebook

The existing pipeline is a cross-sectional CNN classifier (per-slice, per-subject) plus whatever the other five phases in the current notebook cover. This plan adds, concretely:

1. Atlas-based regional volumetrics as a first-class feature store (Phase 2) — currently the pipeline works at the whole-scan/slice level, not the region level.
2. A co-atrophy proxy network and a trained propagation-edge scorer (Phases 3–4) — currently there is no inter-region relational modeling at all.
3. Genuine trajectory/longitudinal modeling using the repeat-visit structure OASIS-2 already provides but the current CNN track does not exploit (Phases 4, 6).
4. A dedicated interpretability layer with audience-specific report rendering (Phase 5, Section 8) — currently the pipeline outputs metrics/predictions but not structured, audience-aware explanations.
5. A forward-simulation module with uncertainty bands and a clearly-scoped counterfactual sensitivity analysis (Phases 6–7) — entirely new capability.
6. Explicit, documented interface stubs for the molecular/cellular layers (Levels 3–5) so the system is architecturally ready for ADNI/PET/spatial-transcriptomics integration without a rewrite — new, and directly addresses the ambition of the pasted proposal while staying honest about OASIS-2's actual contents.

---

## 14. Proposed name

**OASIS-CELLTWIN (Regional Scope): An Interpretable Digital Twin for Structural Mapping and Prediction of Alzheimer's Disease Progression on Longitudinal MRI**

— with the full multimodal/cellular version (once richer data is available) retaining the original **AD-CELLTWIN** / **BrainTwin-AD** naming from the source proposal.
