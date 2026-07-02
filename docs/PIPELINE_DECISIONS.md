# Pipeline Decisions Log

> This document records every methodological and technical decision made during the implementation of the Rhetoric Detection Pipeline. Each decision includes its justification (statistical, theoretical, or practical) and traceability to the relevant literature or data observation. This document is intended to feed directly into the paper's Methodology section.

---

## Table of Contents

1. [Data Collection](#1-data-collection)
2. [Preprocessing](#2-preprocessing)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Sampling](#4-sampling)
5. [Taxonomy](#5-taxonomy)
6. [Classification](#6-classification)
7. [Annotation](#7-annotation)
8. [Evaluation](#8-evaluation)

---

## 1. Data Collection

### 1.1 Data Source

| Item | Decision |
|------|----------|
| **Source** | Câmara dos Deputados Open Data API (dadosabertos.camara.leg.br/api/v2) |
| **Legislature** | 57th (Feb 2023 – present) |
| **Collection period** | 2023-02-01 to 2024-12-31 |
| **Justification** | Current legislature provides contemporary political discourse; 2-year window balances temporal coverage with topical coherence. Legislature 57 started Feb 2023 after the 2022 elections, providing a politically active period with fresh mandates. |

### 1.2 Corpus Size

| Item | Value |
|------|-------|
| **Raw speeches collected** | 19,428 |
| **Unique deputies** | 371 |
| **Unique parties** | 21 |
| **Collection date** | 2026-07-01 |

### 1.3 Political Spectrum Classification

| Item | Decision |
|------|----------|
| **Method** | Party-level classification (left/center/right) |
| **Source** | Zucco & Power (2024) ideological estimates for Brazilian parties |
| **Justification** | Party-level is more stable than individual-level for this sample size; Zucco & Power is the standard reference for Brazilian legislative ideology estimates. |
| **Distribution** | Right: 10,532 (54.2%), Left: 6,655 (34.3%), Center: 2,240 (11.5%) |
| **Imbalance handling** | Addressed via stratified sampling in later stages |

### 1.4 Session Type Handling

| Item | Decision |
|------|----------|
| **Observation** | The API field `tipSessao` returned "Unknown" for all speeches. The actual content type is captured in `legislative_phase` (faseEvento). |
| **Relevant phases** | "Ordem do Dia" (10,122 / 52.1%) and "Breves Comunicações" (8,247 / 42.4%) |
| **Excluded phases** | "Encerramento" (855), "Homenagem" (132), "Comissão Geral" (63), "Abertura" (8) |
| **Justification** | Ordem do Dia and Breves Comunicações are the argumentative sessions where deputies present political positions and debate. Encerramento/Abertura are procedural. Homenagem speeches are tributes with different rhetorical structure. Comissão Geral has different dynamics. |
| **Filter field** | `legislative_phase` (not `session_type`) |

### 1.5 Persistence Strategy

| Item | Decision |
|------|----------|
| **Format** | One JSON file per speech |
| **Filename** | `{PARTY}_{DeputyName}_{date}_{hash8}.json` |
| **Resumability** | Checkpoint file (`_checkpoint.json`) per deputy |
| **Justification** | Individual files allow incremental collection and easy inspection. Deterministic filenames prevent duplicates on re-runs. Checkpoint enables resuming after interruption without re-downloading. |

---

## 2. Preprocessing

### 2.1 Filter Pipeline

| Step | Purpose | Justification | Implementation |
|------|---------|---------------|----------------|
| Legislative phase filter | Retain only argumentative phases (Ordem do Dia, Breves Comunicações) | See §1.4 — procedural sessions lack political rhetoric | `LegislativePhaseFilter` — set membership check |
| Monologue isolation | Extract deputy's continuous speech before any presidente interjection | Interjections from "O SR. PRESIDENTE" or "A SRA. PRESIDENTA" introduce a different speaker; only the deputy's own words should be classified | `MonologueIsolator` — regex for speaker-change pattern |
| Formality removal | Strip opening identification block ("O SR. NAME (Party - UF. Sem revisão do orador.) -") | This is metadata, not rhetorical content; including it would bias the classifier toward procedural language | `FormalityRemover` — regex for opening block + revision notes |
| Text normalization | NFKC Unicode normalization, collapse whitespace | Consistency for TF-IDF and sentence detection; eliminates encoding artifacts | `TextNormalizer` |
| Deduplication | TF-IDF cosine similarity ≥ 0.75, per-deputy | Deputies occasionally repeat template speeches; per-deputy scope avoids removing legitimately similar arguments from different deputies | `Deduplicator` — scikit-learn TfidfVectorizer + cosine_similarity, greedy first-occurrence retention |
| Minimum length | ≥ 3 sentences | Segmentation requires 3-5 sentences per segment; speeches below this threshold cannot produce a single valid segment | `MinimumLengthFilter` — regex sentence boundary detection |

**Design pattern:** Strategy pattern (`FilterStep` ABC) allows adding/removing/reordering filters without modifying the pipeline orchestrator. Each filter has a single responsibility.

### 2.2 Actual Attrition (run 2026-07-02)

| Step | Input | Output | Removed | Rate |
|------|-------|--------|---------|------|
| Raw corpus (1 file had parse error) | 19,428 | 19,427 | 1 | 0.0% |
| Legislative phase filter | 19,427 | 18,369 | 1,058 | 5.5% |
| Monologue isolation | 18,369 | 18,296 | 73 | 0.4% |
| Formality removal | 18,296 | 18,296 | 0 | 0.0% |
| Text normalization | 18,296 | 18,296 | 0 | 0.0% |
| Deduplication (TF-IDF cosine ≥ 0.75) | 18,296 | 18,275 | 21 | 0.1% |
| Minimum length (< 3 sentences) | 18,275 | 18,006 | 269 | 1.5% |
| **Final usable corpus** | **19,427** | **18,006** | **1,421** | **7.3%** |

**Interpretation:** Very low attrition (7.3%) indicates high-quality collection. The largest filter is legislative phase (procedural sessions), which is expected. The extremely low deduplication rate (0.1%) suggests deputies rarely give near-identical speeches — the corpus has high content diversity.

---

## 3. Exploratory Data Analysis

*Completed 2026-07-02. Full report: `results/eda/eda_report.md`. Notebook: `notebooks/03_eda.ipynb`.*

### 3.1 Corpus Characterization

| Metric | Value |
|--------|-------|
| Processed speeches | 18,006 |
| Unique deputies | 366 |
| Unique parties | 19 |
| Median word count | 256 |
| Median sentence count | 15 |
| Mean segments/speech | 4.3 (at 4 sentences/segment) |
| Total available segments | ~76,826 |

### 3.2 Spectrum Distribution (post-filtering)

| Spectrum | Speeches | Deputies | Gini |
|----------|----------|----------|------|
| Left | 6,358 (35.3%) | 83 | 0.592 |
| Center | 1,835 (10.2%) | 83 | 0.609 |
| Right | 9,813 (54.5%) | 200 | 0.723 |

### 3.3 Key Statistical Findings

1. **No length confound**: η² = 0.000075 (spectrum explains <0.01% of word count variance) — length is not confounded with political position.
2. **Session type balance**: All spectrums have ~45-56% of each session type — no systematic bias.
3. **Deputy concentration**: Gini = 0.693 overall. Top 31 deputies produce 50% of speeches. Stratified sampling must control for this (cap per-deputy contributions).
4. **Temporal coverage**: 4 semesters with smallest stratum = 366 speeches (center/2024-S2). All sampling options (50-200/stratum) are feasible.

### 3.4 Sampling Decision

| Item | Decision |
|------|----------|
| **Recommended option** | B: 75 speeches/stratum × 12 strata = 900 speeches |
| **Segments produced** | ~3,870 |
| **Annotation target** | 600 segments (gold standard subset) |
| **Power (H2)** | Detects OR ≥ 1.5 with Bonferroni-corrected α |
| **Precision (H3)** | 95% CI half-width ≈ ±0.012 on macro-F1 |
| **Annotation time** | ~30 hours per annotator |
| **Justification** | Balances statistical rigor with annotation feasibility within deadline |

---

## 4. Sampling

*To be documented after implementation.*

### 4.1 Stratification Strategy

| Item | Decision |
|------|----------|
| **Strata** | political_spectrum × temporal_period |
| **Temporal periods** | 4 (quarterly or semester-based, TBD by EDA) |
| **Target per stratum** | TBD (informed by EDA and power analysis) |
| **Justification** | Balanced representation ensures bias tests (H2) have equal power across spectrum. Temporal stratification controls for political events. |

### 4.2 Segmentation

| Item | Decision |
|------|----------|
| **Segment size** | 3–5 sentences |
| **Justification** | SemEval-2020 Task 11 and SemEval-2023 Task 3 use span-level annotation on passages of similar granularity. 3–5 sentences provide enough context for technique identification while maintaining annotation feasibility. |
| **Traceability** | Each segment has ID `{speech_id}_seg{NNN}` with char offsets back to the processed speech |

---

## 5. Taxonomy

### 5.1 Technique Selection

| Technique | Source | Expected Frequency |
|-----------|--------|-------------------|
| Loaded Language | Da San Martino et al. (2020), Piskorski et al. (2023) | 23.7% |
| Name Calling | Da San Martino et al. (2020), Piskorski et al. (2023) | 18.5% |
| Doubt | Piskorski et al. (2023) | 12.5% |
| Appeal to Fear | Da San Martino et al. (2020) | 8.0% |
| Causal Oversimplification | Da San Martino et al. (2020) | 6.0% |
| Flag-Waving | Da San Martino et al. (2020) | 5.0% |

| Item | Decision |
|------|----------|
| **Count** | 6 techniques |
| **Justification** | Selected for (a) relevance to political discourse, (b) sufficient expected frequency for statistical power, (c) established inter-annotator agreement in SemEval shared tasks, (d) coverage of different manipulation strategies (emotional, logical, identity-based). |
| **Language** | Technique names in English (canonical from SemEval); definitions and examples in Portuguese (target language for annotation and classification prompts). |

---

## 6. Classification

*To be documented after implementation.*

### 6.1 Model Selection

| Model | Type | Justification |
|-------|------|---------------|
| GPT-4o | Commercial LLM (few-shot) | State-of-the-art performance baseline; zero-shot/few-shot capability in Portuguese |
| LLaMA 3.x (8B, Q4) | Open-source LLM (few-shot) | Reproducibility; local execution; comparison with commercial model |
| BERTimbau | Fine-tuned encoder | Portuguese-specific pre-training; supervised baseline; computational efficiency |

### 6.2 Prompt Design

*To be documented after pilot.*

### 6.3 Temperature and Parameters

| Model | Temperature | Justification |
|-------|-------------|---------------|
| GPT-4o | 0.0 | Deterministic outputs for reproducibility |
| LLaMA | 0.0 | Same rationale |
| BERTimbau | N/A (argmax/threshold) | Supervised; threshold tuned on dev set |

---

## 7. Annotation

*To be documented after annotation protocol design.*

### 7.1 Annotators

| Item | Decision |
|------|----------|
| **Count** | 2 (Brenno + Gustavo) |
| **Calibration** | Iterative rounds until κ ≥ 0.40 per technique |
| **Separation** | Calibration segments excluded from evaluation set |

### 7.2 Agreement Target

| Item | Decision |
|------|----------|
| **Metric** | Cohen's κ (per technique and global) |
| **Threshold** | κ ≥ 0.40 (moderate agreement, Landis & Koch 1977) |
| **Justification** | κ < 0.40 indicates the technique definition is too ambiguous for reliable human annotation, making model evaluation unreliable. |

---

## 8. Evaluation

*To be documented after implementation.*

### 8.1 Hypotheses

| ID | Hypothesis | Test |
|----|-----------|------|
| H1 | Annotators achieve κ ≥ 0.40 per technique | Cohen's κ |
| H2 | Models show no systematic political bias | McNemar + Odds Ratio on minimal pairs, Bonferroni correction (18 tests) |
| H3 | Model comparison | Per-technique F1, Macro-F1, Bootstrap CI (B=10,000), McNemar between model pairs |

### 8.2 Statistical Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Significance level (α) | 0.05 | Standard |
| Bonferroni correction | 18 tests (3 models × 6 techniques) | Controls family-wise error rate for H2 |
| Bootstrap iterations | 10,000 | Standard for CI estimation (Efron & Tibshirani, 1993) |
| Random seed | 42 | Reproducibility |

---

## Changelog

| Date | Section | Change |
|------|---------|--------|
| 2026-07-02 | 1. Data Collection | Initial documentation of collection decisions |
| 2026-07-02 | 5. Taxonomy | Documented technique selection and justification |
| 2026-07-02 | 8. Evaluation | Documented hypothesis testing framework |
| 2026-07-02 | 2. Preprocessing | Implemented and documented filter pipeline; recorded actual attrition (7.3% total removal) |
| 2026-07-02 | 3. EDA | Full EDA complete: 7 figures, stratification analysis, power analysis, sampling recommendation (Option B: 75/stratum) |
