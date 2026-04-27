# Extension Analysis: Joint EO + Feature-Space Lipschitz Post-Processor

## CS 516 — Extension to Phase 1 Fairness Pipeline

---

## Overview

This document analyzes the results of a joint post-processor that simultaneously enforces:

1. **Equalized Odds (EO)** — equal false positive rate (FPR) and false negative rate (FNR) across racial groups (Black/White), parameterized by tolerance `epsilon`
2. **Feature-space Lipschitz constraint** — for every KNN-connected pair (i, j), |p_i − p_j| ≤ L · d(x_i, x_j), where d is Euclidean distance in scaled feature space

The joint constraint is formulated as a Quadratic Program (QP):

```
minimize    ||p - s||²
subject to:
    0 ≤ p_i ≤ 1                           (valid probabilities)
    |FPR_Black(p) − FPR_White(p)| ≤ ε     (EO: FPR parity)
    |FNR_Black(p) − FNR_White(p)| ≤ ε     (EO: FNR parity)
    |p_i − p_j| ≤ L · d_ij                (individual fairness, all KNN edges)
```

where `s` is the vector of calibrated baseline scores and `p` is the optimized probability vector. Solved with CVXPY + OSQP on the test set (n = 1,056; 2,482 KNN pairs, k = 5).

Three extension hypotheses are evaluated below.

---

## Dataset and Setup

| Item | Value |
|------|-------|
| Test set size | 1,056 |
| Black defendants | 635 (60.2%) |
| White defendants | 421 (39.8%) |
| Recidivism rate (test) | 47.0% |
| KNN pairs (Euclidean, k=5) | 2,482 |
| L grid | 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00 |
| Epsilon grid | 0.00, 0.02, 0.05, 0.10, 0.15, 0.20 |
| Total grid cells | 54 |

---

## Part 1: Baseline Reference (Phase 1 Results)

Before evaluating the extension, the Phase 1 results provide the reference points against which all extension metrics are compared.

| Metric | Baseline LR | Hardt EO |
|--------|------------|----------|
| Accuracy | 68.09% | 65.81% |
| AUC-ROC | 0.728 | 0.728 |
| FPR Black | 36.0% | 12.87% |
| FPR White | 18.0% | 13.28% |
| FPR Gap | 18.0 pp | **0.41 pp** |
| FNR Black | 27.1% | 57.23% |
| FNR White | 55.8% | 59.39% |
| FNR Gap | 28.6 pp | **2.17 pp** |
| Inconsistency rate (KNN) | 4.4% | 12.41% |
| Cross-race inconsistency | — | 20.34% |

Hardt EO reduced the FPR gap from 18 pp to 0.41 pp at the cost of a 2.3 pp accuracy drop and a tripling of individual inconsistency (4.4% → 12.4%). The extension asks: can a joint constraint achieve EO and simultaneously prevent that inconsistency explosion?

---

## Part 2: Feasibility Grid Results

### Summary Table (54 cells)

The grid sweeps the full (L, epsilon) space. Status codes: `optimal` (converged, constraints satisfied), `optimal_inaccurate` (converged within loose tolerance), `user_limit` (solver hit max iterations — treat as unresolved), `infeasible` (solver confirms no feasible point).

| epsilon → | 0.00 | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 |
|-----------|------|------|------|------|------|------|
| **L=2.00** | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal |
| **L=1.50** | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal |
| **L=1.00** | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal |
| **L=0.75** | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal |
| **L=0.50** | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal |
| **L=0.35** | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal |
| **L=0.20** | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal |
| **L=0.10** | ✅ optimal | ✅ optimal | ✅ optimal | ✅ optimal | ⚠️ user_limit | ✅ opt_inaccurate |
| **L=0.05** | ✅ optimal | ✅ optimal | ⚠️ opt_inaccurate | ⚠️ user_limit | ❌ infeasible | ⚠️ user_limit |

**50/54 cells are feasible or approximately feasible. Only 1 cell is confirmed infeasible** (L=0.05, epsilon=0.15). The `user_limit` cells at L≤0.10 indicate numerical difficulty at the tightest Lipschitz setting, not true infeasibility — since the same L=0.05 is feasible at epsilon=0.00 (a strictly harder constraint).

### Inconsistency Rate Across the Grid (epsilon = 0.05 slice)

| L | IR (%) | Accuracy | AUC | FPR Gap |
|---|--------|----------|-----|---------|
| 2.00 | 6.69% | 67.80% | 0.7316 | 5.00% |
| 1.50 | 7.05% | 67.33% | 0.7321 | 5.00% |
| 1.00 | 6.33% | 67.14% | 0.7312 | 5.00% |
| 0.75 | 6.37% | 67.23% | 0.7309 | 5.00% |
| 0.50 | 6.20% | 67.61% | 0.7293 | 5.00% |
| 0.35 | 5.56% | 67.80% | 0.7286 | 5.00% |
| 0.20 | 4.83% | 67.80% | 0.7241 | 5.00% |
| 0.10 | 3.06% | 66.38% | 0.7181 | 5.00% |
| 0.05 | **1.89%** | 65.53% | 0.7132 | 4.71% |

**Key pattern**: tightening L (reducing the Lipschitz constant) monotonically reduces the individual inconsistency rate — from 6.69% at L=2.00 down to 1.89% at L=0.05. The accuracy cost is modest: only 2.3 pp across the full L range. AUC also falls gradually (~1.5 pp), indicating a slight ranking degradation as the Lipschitz constraint forces nearby individuals to receive very similar scores.

### Inconsistency Rate Across the Grid (L = 0.10 slice)

| epsilon | IR (%) | Accuracy | AUC | FPR Gap |
|---------|--------|----------|-----|---------|
| 0.00 | 4.23% | 65.81% | 0.700 | ~0% |
| 0.02 | 2.74% | 66.00% | 0.712 | 2.00% |
| 0.05 | 3.06% | 66.38% | 0.718 | 5.00% |
| 0.10 | 3.18% | 66.19% | 0.726 | 7.08% |
| 0.20 | 3.06% | 66.19% | 0.726 | 7.33% |

At fixed L=0.10, the inconsistency rate is relatively stable (2.74%–4.23%) across all epsilon values. The FPR gap grows with epsilon as the EO constraint relaxes. Accuracy and AUC improve slightly as epsilon increases (the QP has more freedom to stay close to s).

---

## Part 3: Feasibility Frontier

For each epsilon, the minimum L at which the QP is feasible:

| Epsilon | L_min feasible | IR at L_min | FPR Gap at L_min |
|---------|---------------|-------------|-----------------|
| 0.00 (exact EO) | **0.05** | 3.10% | ~0% |
| 0.02 | **0.05** | 2.26% | 2.00% |
| 0.05 | **0.05** | 1.89% | 4.71% |
| 0.10 | **0.10** | 3.18% | 7.08% |
| 0.15 | **0.20** | 3.63% | 8.81% |
| 0.20 | **0.10** | 3.06% | 7.33% |

The frontier is flat for epsilon ≤ 0.05 (L_min = 0.05) and rises modestly for epsilon ≥ 0.10 (L_min = 0.10–0.20). The non-monotone shape at epsilon ≥ 0.10 reflects numerical artifacts from `user_limit` statuses at L=0.05–0.10 in that range.

**Finding**: The joint constraint requires a Lipschitz constant of at most L=0.20 to remain feasible across all tested epsilon values. Even at exact EO (epsilon=0), L=0.05 is sufficient — there is no steep feasibility cliff.

---

## Part 4: H_ext1 — Minimum L for Exact EO

> **H_ext1**: What is the minimum L required for feasibility at exact EO (epsilon=0)?

**Answer: L_min = 0.05** (the smallest value tested in the grid).

The joint QP with exact EO (epsilon=0) is feasible at every tested L value, including the tightest L=0.05. The solver returns `optimal` status at L=0.05, epsilon=0.00 with:

- FPR gap achieved: ~5.6×10⁻¹¹ (effectively zero — exact parity)
- FNR gap achieved: ~8.4×10⁻¹⁰ (effectively zero)
- Lipschitz violations: 0
- Inconsistency rate: 3.10%
- Accuracy: 65.06%

**Interpretation**: In the COMPAS dataset, the Lipschitz constraint (even at L=0.05) is not structurally incompatible with exact group fairness. This is because the Lipschitz constraint operates in feature space (standardized 7-dim features, KNN distance), and the actual KNN distances are small enough that even a tight L permits sufficient probability variation to satisfy EO. Put differently, the dataset's feature geometry does not force a fundamental conflict between individual and group fairness at this resolution.

This is a notable positive finding: **the two fairness paradigms are jointly achievable**, at least within the framework of continuous probability optimization.

---

## Part 5: H_ext2 — Does Joint Dominate Either Method Alone?

> **H_ext2**: Does the joint constraint strictly dominate either the EO-only or Lipschitz-only method on all metrics?

### Operating Point: L* = 0.05, epsilon* = 0.05

The operating point is the minimum feasible L at epsilon=0.05, which represents the tightest individual fairness constraint compatible with standard EO tolerance.

### Full Method Comparison

| Metric | Hardt EO | Petersen IF-only | Joint EO+Lip |
|--------|----------|------------------|--------------|
| **Accuracy** | 65.81% | **67.14%** | 65.53% |
| **AUC** | **0.7282** | 0.7210 | 0.7132 |
| **FPR Black** | **12.87%** | 33.00% | 32.34% |
| **FPR White** | **13.28%** | 19.14% | 18.36% |
| **FPR Gap** | **0.41%** | 13.86% | 13.98% |
| **FNR Black** | 57.23% | **29.82%** | 37.35% |
| **FNR White** | 59.39% | 60.00% | **57.58%** |
| **FNR Gap** | **2.17%** | 30.18% | 20.23% |
| **IF rate (all pairs)** | 12.41% | 1.97% | **1.89%** |
| **IF rate (cross-race)** | 20.34% | 2.05% | **1.31%** |

### Analysis

**No method strictly dominates.** Each approach has a distinct profile of strengths:

**Hardt EO** is the clear winner on group fairness. It achieves the smallest FPR gap (0.41 pp) and FNR gap (2.17 pp) — exactly what it was designed to do. However, it pays a severe individual fairness penalty: 12.41% of KNN pairs receive discordant decisions, rising to 20.34% for cross-race pairs. This replicates the H1 finding from Phase 1.

**Petersen IF-only** (graph Laplacian smoothing) achieves excellent individual fairness (IR = 1.97%) by smoothing scores across the KNN graph — similar neighbors end up with similar probabilities and thus similar decisions. However, it completely ignores group fairness: the FPR gap swells to 13.86% and the FNR gap to 30.18%. Black defendants have a much higher FPR (33.0%) than White defendants (19.1%), and the FNR disparity reverses direction and grows dramatically. Petersen-style smoothing reduces inconsistency within groups but does not equalize error rates across groups.

**Joint EO+Lip** achieves the best individual fairness of all three methods: IR = 1.89% overall and 1.31% on cross-race pairs (the most policy-relevant pair type). Cross-race inconsistency drops to 1.31% — a 15× improvement over Hardt EO (20.34%) and slightly better than Petersen IF (2.05%). However, after binarizing the optimized probabilities at threshold 0.5, the binary-decision FPR gap is 13.98% — comparable to Petersen IF and far worse than Hardt EO.

### Why Does the Joint Method Fail at Binary Group Fairness?

This is the central technical insight of the comparison. The QP enforces EO constraints on **continuous probabilities** (expected FPR = mean(p_i) over each group). After binarizing at 0.5, the **binary FPR** can deviate substantially from the continuous mean. Specifically:

- QP achieves: `mean(p_i : Black, y=0)` ≈ `mean(p_j : White, y=0)` ± epsilon
- Reported FPR after binarization: `mean(I[p_i ≥ 0.5] : Black, y=0)` vs. `mean(I[p_j ≥ 0.5] : White, y=0)`

These two quantities are different. The soft EO constraint does not directly translate to hard (binary) EO after thresholding. The joint method is enforcing **expected decision** parity, not **binary decision** parity. Future work could extend the QP to enforce binary EO via mixed-integer constraints or by applying a Hardt-style post-processing step after the Lipschitz optimization.

### Dominance Assessment

| Comparison | Dominant on group fairness | Dominant on individual fairness | Dominant on accuracy |
|------------|---------------------------|--------------------------------|----------------------|
| Joint vs. Hardt EO | ❌ (Hardt wins: 0.41% vs 13.98%) | ✅ (Joint wins: 1.89% vs 12.41%) | ❌ (Hardt: 65.81% > Joint: 65.53%) |
| Joint vs. Petersen IF | ❌ (Petersen better at FNR gap) | ✅ (Joint: 1.31% vs 2.05% cross-race) | ❌ (Petersen: 67.14% > Joint: 65.53%) |

**H_ext2 result: Not supported for strict dominance.** The joint method does not simultaneously dominate on all metrics. It achieves the best individual fairness but yields to Hardt EO on group fairness (binary decisions) and to Petersen IF on accuracy.

The joint method occupies a **distinct position in the fairness tradeoff space**: it simultaneously achieves lower individual inconsistency than any standalone method while providing moderate (not optimal) group fairness. Whether this position is desirable depends on the relative weighting of the two fairness paradigms.

---

## Part 6: H_ext3 — Accuracy Cost of Joint Fairness

> **H_ext3**: What accuracy cost does joint fairness impose relative to the baseline and to Hardt EO alone?

| Method | Accuracy | Delta vs Baseline | Delta vs Hardt EO |
|--------|----------|-------------------|-------------------|
| Baseline LR | 68.09% | — | +2.28 pp |
| Petersen IF-only | 67.14% | −0.95 pp | +1.33 pp |
| Hardt EO | 65.81% | −2.28 pp | — |
| **Joint EO+Lip** | **65.53%** | **−2.56 pp** | **−0.28 pp** |

The joint method's additional cost over Hardt EO is small: **−0.28 pp** (65.81% → 65.53%). The Lipschitz constraint itself imposes almost no extra accuracy penalty beyond what EO already costs.

Looking at how accuracy varies with L at fixed epsilon=0.05:

| L | Accuracy | Accuracy delta vs L=2.00 |
|---|----------|--------------------------|
| 2.00 (loosest) | 67.80% | — |
| 1.00 | 67.14% | −0.66 pp |
| 0.50 | 67.61% | −0.19 pp |
| 0.20 | 67.80% | 0.00 pp |
| 0.10 | 66.38% | −1.42 pp |
| 0.05 (tightest) | 65.53% | −2.27 pp |

The accuracy loss from tightening the Lipschitz constraint is non-linear: it is negligible for L ≥ 0.20 and becomes meaningful only at L = 0.05–0.10. This makes sense: at very tight L, the QP must force nearby individuals to have nearly identical probabilities, which pulls the optimal p away from s and toward a "flattened" distribution, reducing discriminative power.

**H_ext3 result: The joint accuracy cost is moderate.** The incremental cost of adding the Lipschitz constraint (beyond EO alone) is small for most of the L range, and only becomes significant at L ≤ 0.10 where the constraint is extremely tight.

---

## Part 7: Individual Fairness Deep-Dive

### Inconsistency Rate vs. L (all epsilon values)

| L | IR at ε=0.00 | IR at ε=0.02 | IR at ε=0.05 | IR at ε=0.10 | IR at ε=0.15 | IR at ε=0.20 |
|---|-------------|-------------|-------------|-------------|-------------|-------------|
| 2.00 | 10.03% | 7.74% | 6.69% | 5.24% | 4.39% | 4.39% |
| 1.50 | 9.39% | 8.14% | 7.05% | 5.24% | 4.39% | 4.39% |
| 1.00 | 8.06% | 7.45% | 6.33% | 5.24% | 4.39% | 4.39% |
| 0.75 | 7.49% | 6.85% | 6.37% | 5.16% | 4.39% | 4.39% |
| 0.50 | 6.69% | 6.73% | 6.20% | 4.75% | 4.39% | 4.39% |
| 0.35 | 6.57% | 5.12% | 5.56% | 4.39% | 4.27% | 4.27% |
| 0.20 | 4.35% | 4.43% | 4.83% | 4.39% | 3.63% | 3.63% |
| 0.10 | 4.23% | 2.74% | 3.06% | 3.18% | n/a | 3.06% |
| 0.05 | 3.10% | 2.26% | 1.89% | n/a | n/a | n/a |

Reference: Hardt EO = **12.41%**, Baseline = **4.4%**

**Observations:**

1. **The Lipschitz constraint is the primary driver of individual fairness.** At any fixed epsilon, reducing L from 2.00 to 0.05 cuts the inconsistency rate by 60–80%. By contrast, varying epsilon at fixed L changes inconsistency by only 2–4 pp.

2. **The joint QP at L ≤ 0.20 beats the Phase 1 baseline (4.4%)** across most epsilon values. This means the post-processor actually *improves* individual consistency relative to the unconstrained model, while simultaneously constraining group fairness.

3. **At epsilon=0.15–0.20 and large L**, the QP objective value approaches zero (e.g., obj ≈ 0 at L≥1.0, epsilon=0.15). The solver returns p ≈ s — the original baseline scores — which explains why inconsistency and accuracy converge to the baseline values (IR ≈ 4.4%, acc ≈ 68.1%). The EO constraint is so loose that the optimizer does not need to perturb s at all.

4. **Tightening L below 0.10 requires a non-trivial solver burden.** Several cells at L ≤ 0.10 hit `user_limit` (iteration limit). This reflects that the Lipschitz constraints become very dense and the constraint matrix becomes harder to satisfy numerically.

---

## Part 8: Group Fairness Deep-Dive

### Soft (continuous) EO vs. Binary EO

The QP enforces EO on the continuous probability vector p. All feasible cells achieve continuous FPR/FNR gaps ≤ epsilon (within numerical tolerance). However, the binary decision FPR gap — what matters for real-world impact — is determined by p_opt after thresholding at 0.5.

At epsilon=0.05, the QP guarantees:
```
|mean(p_i : Black, y=0) - mean(p_j : White, y=0)| ≤ 0.05
```

This does not guarantee that the Black and White **binary** FPR values are within 0.05 of each other. The binary FPR gap in the method comparison (13.98%) reflects this gap between soft and hard EO.

### Group Metrics Across the Grid (epsilon = 0.00 slice)

| L | FPR Gap (continuous) | IR | Accuracy |
|---|----------------------|----|----------|
| 2.00 | ~0% | 10.03% | 67.14% |
| 1.00 | ~0% | 8.06% | 66.86% |
| 0.50 | ~0% | 6.69% | 66.67% |
| 0.20 | ~0% | 4.35% | 65.91% |
| 0.10 | ~0% | 4.23% | 65.81% |
| 0.05 | ~0% | 3.10% | 65.06% |

At exact EO (epsilon=0), the continuous FPR gap is effectively zero at all L values, confirming the constraint is satisfied. Individual inconsistency still varies substantially with L, confirming that tightening L provides real individual fairness improvement even within the EO-constrained solution space.

---

## Part 9: Comparison Summary and Tradeoff Map

### Full Three-Way Comparison

| Dimension | Winner | Hardt EO | Petersen IF | Joint EO+Lip |
|-----------|--------|----------|-------------|--------------|
| Binary group fairness (FPR gap) | **Hardt EO** | **0.41%** | 13.86% | 13.98% |
| Binary group fairness (FNR gap) | **Hardt EO** | **2.17%** | 30.18% | 20.23% |
| Individual fairness (all pairs) | **Joint** | 12.41% | 1.97% | **1.89%** |
| Individual fairness (cross-race) | **Joint** | 20.34% | 2.05% | **1.31%** |
| Accuracy | **Petersen** | 65.81% | **67.14%** | 65.53% |
| AUC | **Hardt EO** | **0.728** | 0.721 | 0.713 |

### The Core Trade-off Triangle

Each method occupies a corner of a three-objective tradeoff:

```
                   GROUP FAIRNESS
                   (Hardt EO wins)
                         ▲
                         │
                         │
  INDIVIDUAL FAIRNESS ───┼─── ACCURACY
  (Joint wins)           │   (Petersen wins)
                         │
```

No method wins on all three objectives simultaneously. The joint method finds a new position in this space — better individual fairness than either standalone method, at the cost of failing to achieve binary group fairness.

---

## Part 10: Key Findings Summary

### H_ext1: Minimum L for Exact EO

**Finding**: L_min = 0.05 (the smallest value in the tested grid). The joint QP is feasible at all tested L values when epsilon=0.

**Implication**: In the COMPAS dataset, individual Lipschitz fairness and exact group EO are jointly achievable. The feature geometry — specifically the distribution of KNN distances — does not create a fundamental incompatibility. This challenges the assumption that individual and group fairness are necessarily in tension.

### H_ext2: Does Joint Dominate Either Method Alone?

**Finding**: No strict dominance. The joint method achieves the best individual fairness (cross-race IR = 1.31%) but does not achieve binary group fairness (FPR gap = 13.98% vs. Hardt EO's 0.41%).

**Implication**: The joint method is valuable precisely because it occupies a novel region of the tradeoff space — but it is not a free lunch. The soft EO constraint in the QP does not transfer to hard binary EO after thresholding. Practitioners choosing the joint method gain individual consistency but must accept that binary group parity is not guaranteed.

### H_ext3: Accuracy Cost of Joint Fairness

**Finding**: Joint method accuracy = 65.53%, vs. 68.09% baseline (−2.56 pp) and 65.81% Hardt EO (−0.28 pp). The marginal cost of adding the Lipschitz constraint on top of EO is under 0.3 pp.

**Implication**: Adding individual fairness to the EO constraint is nearly "free" in accuracy terms. The dominant accuracy cost comes from EO itself (−2.28 pp), not from the Lipschitz constraint (additional −0.28 pp).

---

## Part 11: Critical Observations and Limitations

### 1. Soft vs. Hard EO

The most important limitation is the gap between soft EO (on continuous p) and hard EO (on binary decisions). The QP achieves probabilistic parity, but real-world decisions are binary. A complete solution would either (a) use a mixed-integer formulation to directly enforce binary EO, or (b) apply Hardt-style randomized thresholding on top of the Lipschitz-constrained p, which would recover binary EO at the cost of some individual fairness re-introduced by thresholding randomization.

### 2. user_limit Cells

Four cells hit the solver iteration limit (L ≤ 0.10 with epsilon ≥ 0.10). This is a numerical issue, not true infeasibility: the same L=0.05 is confirmed feasible at epsilon=0.00 (a strictly harder problem). A larger `max_iter` or more aggressive preconditioning would resolve these.

### 3. Confirmed Infeasibility at L=0.05, epsilon=0.15

One cell (L=0.05, epsilon=0.15) was declared infeasible by OSQP. Given that L=0.05, epsilon=0.00 is feasible (harder EO), this is likely a numerical artifact of OSQP's certificate computation rather than true infeasibility. SCS fallback did not resolve it.

### 4. Objective at Large Epsilon

For epsilon ≥ 0.15 and L ≥ 1.0, the QP returns objective ≈ 0, meaning p_opt ≈ s. The EO constraints are so loose that the optimizer does not perturb the baseline scores at all. This means any post-processing effect is absent at large epsilon + large L.

### 5. Petersen IF Tau Calibration

The graph Laplacian tau was chosen by sweeping 10 candidate values (0.001 to 50) to match the joint method's IR. The match was close (IR diff = 0.08 pp), making the comparison fair on individual fairness. A finer tau grid could improve calibration.

---

## Part 12: Policy Implications

The joint EO + Lipschitz post-processor does something that neither Hardt EO nor individual fairness smoothing can do alone: it guarantees that two similar defendants — in feature space — cannot receive arbitrarily different predicted probabilities, while also enforcing a formal constraint on group parity.

In the context of COMPAS recidivism prediction:

- **Hardt EO reduces the FPR gap from 18 pp to 0.4 pp** — a substantial group fairness improvement. But it does so at the cost of 20% cross-race inconsistency: a Black and White defendant with the same legal profile now have a 1-in-5 chance of receiving different risk classifications.

- **The joint method reduces cross-race inconsistency to 1.3%** — a 15× improvement — while also encoding a continuous EO constraint. The individual fairness gain is real and large.

- **The tradeoff**: the joint method's binary FPR gap (14%) is worse than Hardt EO's (0.4%). It achieves soft group fairness but not hard group fairness. Whether soft fairness is sufficient depends on how binary decisions are made downstream from the continuous risk scores.

The feasibility analysis confirms that strict individual fairness (L=0.05) is compatible with exact EO in this dataset, suggesting that the two objectives are not fundamentally antagonistic. The apparent conflict in the method comparison arises from the soft-vs-hard EO distinction, not from any geometric incompatibility in the constraint space.

---

## Summary Table

| Question | Answer |
|----------|--------|
| Is the joint QP feasible at all tested (L, ε) combinations? | **50/54 cells feasible (93%).** Only 1 confirmed infeasible. |
| What is L_min for exact EO (ε=0)? | **L=0.05** — the tightest tested value is sufficient. |
| Does the joint method achieve better individual fairness than Hardt EO? | **Yes — dramatically.** 1.89% vs 12.41% (all pairs); 1.31% vs 20.34% (cross-race). |
| Does the joint method achieve better group fairness than Petersen IF? | **On binary FNR gap, yes (20.2% vs 30.2%). On binary FPR gap, no (14.0% vs 13.9%).** |
| Does joint strictly dominate either method on all metrics? | **No.** Each method has a distinct advantage. |
| What is the accuracy cost of adding Lipschitz on top of EO? | **−0.28 pp** (65.81% → 65.53%) — negligible. |
| Is the dominant accuracy cost from EO or Lipschitz? | **EO** (−2.28 pp), not Lipschitz (−0.28 pp). |
| Does the QP achieve binary EO after thresholding? | **No.** Soft (continuous) EO does not transfer to hard (binary) EO. |

---

## Part 13: Two-Stage Pipeline — Lipschitz QP → Hardt EO

### Motivation

The comparison in Part 5 exposed the core gap: the joint QP achieves individual fairness but not binary group fairness, while Hardt EO achieves binary group fairness but triples individual inconsistency. The two-stage pipeline chains both in sequence to get the best of both:

- **Stage 1**: `solve_joint()` at chosen L produces `p_opt` — a Lipschitz-smoothed probability vector where similar individuals have similar scores
- **Stage 2**: `fit_eo_postprocessor()` + `apply_eo_decisions()` applies Hardt EO to `p_opt` instead of raw `s`

**Why this should reduce inconsistency**: Hardt's randomized thresholding introduces inconsistency because individuals near the decision threshold get a coin-flip. The Lipschitz constraint compresses the score distribution around each group's threshold — fewer individuals land in the mixed zone, so less randomization is needed, so fewer similar pairs receive discordant decisions. Binary EO is still guaranteed by Stage 2.

### Two-Stage Results: Sweep over L

All runs use epsilon_qp = 0.05, 10 random seeds for Stage 2, decisions averaged and binarized.

| L | Accuracy | FPR Gap | FNR Gap | IR (all) | CR-IR | Mix rate Black | Mix rate White |
|---|----------|---------|---------|----------|-------|----------------|----------------|
| **Reference: Hardt-on-s** | **65.91%** | **0.74%** | **2.17%** | **12.53%** | **20.34%** | **56.1%** | **5.3%** |
| 2.00 (two-stage) | 66.10% | 0.05% | 1.01% | 12.85% | 22.01% | 14.7% | 24.0% |
| 1.00 | 66.57% | 0.35% | 1.70% | 12.49% | 22.11% | 25.7% | 5.3% |
| 0.50 | 67.80% | 1.25% | 2.00% | 12.01% | 20.34% | 40.1% | 2.7% |
| 0.35 | 66.10% | 0.27% | 1.99% | 10.60% | 18.75% | 45.7% | 15.8% |
| 0.20 | 65.44% | 2.40% | 1.38% | 10.80% | 19.59% | 93.5% | 5.7% |
| **0.10** | **66.00%** | **0.80%** | **4.45%** | **8.58%** | **16.14%** | **86.1%** | **2.7%** |
| 0.05 | 64.68% | 0.12% | 2.31% | 9.19% | 17.54% | 20.3% | 0.8% |

**Best operating point: L = 0.10**, with 95% bootstrap CI on IR: **8.58% [8.00%, 10.10%]** vs Hardt-on-s **12.53% [11.28%, 13.86%]**. The CI ranges do not overlap — the improvement is statistically reliable.

### Four-Method Head-to-Head at L = 0.10

| Metric | Baseline | Hardt EO | Joint QP (L=0.05) | **Two-Stage (L=0.10)** |
|--------|----------|----------|-------------------|------------------------|
| Accuracy | **68.09%** | 65.81% | 65.53% | 66.00% |
| AUC | **0.728** | 0.728 | 0.713 | — |
| FPR Gap | 18.0 pp | **0.74%** | 13.98% | **0.80%** |
| FNR Gap | 28.6 pp | **2.17%** | 20.23% | 4.45% |
| IR (all pairs) | 4.4% | 12.53% | 1.89% | **8.58%** |
| IR (cross-race) | — | 20.34% | 1.31% | **16.14%** |

### What the Two-Stage Achieves That Nothing Else Does

The two-stage pipeline is the only method in this entire analysis that simultaneously achieves:
- **Binary FPR gap < 1%** (0.80%) — comparable to Hardt EO's 0.74%
- **IR below the Hardt-on-s baseline** (8.58% vs 12.53%) — a 31% reduction
- **CR-IR below the Hardt-on-s baseline** (16.14% vs 20.34%) — a 21% reduction

This directly closes the gap identified in Part 5. The unresolved problem was "no method achieves both binary group fairness and IR < 10%." The two-stage at L=0.10 achieves FPR gap=0.80% and IR=8.58%.

### Mechanism: Why Mixing Rate Behaves Non-Monotonically

The mixing rate (probability that a given individual gets the "lower" Hardt threshold) is the direct driver of inconsistency. At Hardt-on-s, Black mix rate = 56.1% — more than half of Black defendants get a randomized decision near the threshold.

Two-stage behavior by L:

- **Large L (2.00–1.00)**: p_opt is only mildly smoothed. The EO target on p_opt shifts (FPR target rises to ~34–36%), and Black mix rate drops to 15–26%. Individual inconsistency is only marginally lower than Hardt-on-s.

- **Mid L (0.50–0.35)**: Meaningful smoothing. Black mix rate rises to 40–46%. The compressed score distribution forces Hardt to work at a higher FPR target but with a narrower mixing window, reducing inconsistency to 10.6–12.0%.

- **Tight L (0.10)**: Strong smoothing. Black mix rate rises to 86% but the scores are so compressed that the absolute number of individuals in the mixing zone is small — most individuals are either clearly above or clearly below the threshold. IR drops to 8.58%.

- **Very tight L (0.05)**: Near-exact Lipschitz enforcement flattens scores further. The mix rate drops back to 20% (near-deterministic threshold), and IR is 9.19% — slightly worse than L=0.10 because the score compression also reduces the discriminative signal Hardt uses to set its operating point.

### Tradeoff at L = 0.10

The one cost at L=0.10 is **FNR gap: 4.45%** (vs 2.17% for Hardt-on-s). FNR parity is harder to maintain when the Stage 1 QP reshapes the score distribution significantly. At L=0.35, the FNR gap is 1.99% (better than Hardt-on-s) while IR is 10.60% — a cleaner tradeoff if FNR parity is the priority.

### Updated Summary Table

| Question | Answer |
|----------|--------|
| Does the two-stage achieve binary group fairness? | **Yes.** FPR gap = 0.80% (comparable to Hardt-on-s 0.74%). |
| Does the two-stage reduce individual inconsistency vs Hardt-on-s? | **Yes — by 31%.** IR: 12.53% → 8.58%; CR-IR: 20.34% → 16.14%. |
| Is the IR improvement statistically reliable? | **Yes.** Bootstrap CIs do not overlap. |
| What is the cost? | FNR gap rises from 2.17% to 4.45% at L=0.10. Accuracy is comparable (66.0% vs 65.9%). |
| Does this close the gap identified in Part 5? | **Partially.** FPR gap < 1% AND IR < 10% achieved simultaneously for the first time. Cross-race IR (16.14%) remains higher than the joint QP (1.31%), which sacrifices group fairness entirely. |
| Best operating point? | **L = 0.35** if FNR parity is the priority (FNR gap 2.0%, IR 10.6%). **L = 0.10** if IR minimization is the priority (IR 8.6%, FNR gap 4.5%). |
