# When Fair Becomes Unfair: Individual Costs of Group Fairness in Recidivism Risk Assessment

A CS 516 (Responsible Data Science and Algorithmic Fairness) research project studying the individual-level costs of enforcing group fairness constraints on risk assessment models.

## Overview

When we enforce **Equalized Odds** (equal false positive and false negative rates across racial groups) on a recidivism risk model, two defendants with nearly identical legal profiles can receive different risk classifications due to the randomization inherent in the post-processor. This project quantifies how often this happens, where it concentrates, and whether joint individual+group fairness constraints can reduce the inconsistency.

**Core question**: Can we achieve group fairness without sacrificing individual fairness — and at what cost?

## Key Findings

### Phase 1 — Baseline Analysis
- Enforcing Equalized Odds (Hardt et al.) reduces the FPR racial gap from **18.0 pp → 0.7 pp**
- But individual inconsistency **triples: 4.4% → 12.5%** (+8.1 pp, 95% CI [6.7%, 9.6%])
- Inconsistency is **not uniform**: pairs closest to decision thresholds face a 35% inconsistency rate (Q1) vs 0% for distant pairs
- A relaxed EO tolerance (ε=0.10) reduces inconsistency by ~2 pp with modest accuracy cost

### Phase 2 — Extension: Joint Optimization
- A joint QP (Equalized Odds + Lipschitz individual fairness) is feasible across 50/54 tested (L, ε) grid points
- **No strict dominance** among three methods — each excels on a different axis:
  - **Hardt EO**: best binary group fairness (FPR gap 0.41%)
  - **Joint QP**: best individual fairness (IR 1.89%) but soft-vs-hard EO gap remains
  - **Two-stage pipeline**: first method to achieve FPR gap < 1% **and** IR < 10% simultaneously (FPR gap 0.80%, IR 8.58%)

## Dataset

[ProPublica COMPAS dataset](https://github.com/propublica/compas-analysis) — 5,278 individuals (60.2% Black, 39.8% White) after standard filters. Race is never used as a model feature; it is used only for group fairness evaluation.

Features used for prediction: age, sex, juvenile felony/misdemeanor/other counts, prior count, charge degree.

## Repository Structure

```
src/
├── config.py                  # Global constants, paths, grids
├── data_prep.py               # COMPAS loading, filtering, feature engineering
├── baseline_model.py          # Logistic regression + isotonic calibration
├── eo_postprocessor.py        # Hardt et al. EO post-processor
├── similarity.py              # KNN and rule-based pair construction
├── metrics.py                 # Inconsistency rate, bootstrap CI, decomposition
├── analysis.py                # H1/H2/H3 hypothesis tests, robustness checks
├── make_comparison_figures.py # Cross-method figure generation
└── extension/
    ├── joint_postprocessor.py # Joint EO + Lipschitz QP solver (CVXPY/OSQP)
    ├── feasibility.py         # Feasibility frontier analysis and plots
    ├── comparison.py          # Three-method head-to-head + radar chart
    └── two_stage.py           # Lipschitz QP → Hardt EO pipeline

notebooks/
├── run_pipeline.ipynb         # Phase 1 end-to-end pipeline
└── extension_analysis.ipynb   # Phase 2 analysis

data/
└── compas_raw.csv             # Raw COMPAS data

results/                       # All outputs (CSVs, pickled models, figures)
```

## Methods

| Component | Method |
|-----------|--------|
| Classifier | Logistic regression + isotonic calibration |
| Group fairness | Hardt, Price & Srebro (2016) randomized EO post-processor |
| Similarity | KNN (k=5, Euclidean/Cosine) in scaled feature space; rule-based matching |
| Inconsistency metric | Fraction of similar pairs with discordant decisions; 95% bootstrap CI (10k resamples) |
| Individual fairness | Lipschitz constraint \|p_i − p_j\| ≤ L·d_ij on KNN graph edges |
| Joint optimizer | Quadratic program via CVXPY + OSQP |

## Phase 2 Method Comparison

| Method | FPR Gap | IR (all pairs) | IR (cross-race) | Accuracy |
|--------|---------|----------------|-----------------|----------|
| Hardt EO | **0.41%** | 12.41% | 20.34% | 65.81% |
| Joint QP (L=0.05, ε=0.05) | 13.98% | **1.89%** | **1.31%** | 65.53% |
| Two-stage (L=0.10) | 0.80% | 8.58% | 16.14% | **66.00%** |

## Setup

```bash
pip install -r requirements.txt
```

**Core dependencies**: numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, cvxpy, osqp

Run Phase 1 via `notebooks/run_pipeline.ipynb` or call `src/analysis.py` directly.  
Run Phase 2 via `notebooks/extension_analysis.ipynb`.

## Results

Full results are in `results/`. Key summaries:

- `results/table1_baseline_metrics.csv` — Baseline model performance
- `results/table2_eo_metrics.csv` — Post-EO metrics
- `results/table3_hypothesis_results.csv` — H1/H2/H3 test outcomes
- `results/extension/method_comparison.csv` — Three-method comparison
- `results/extension/figures/` — All generated figures (300 DPI)

## References

- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS*.
- Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. *ITCS*.
- Angwin, J. et al. (2016). Machine bias. *ProPublica*.
