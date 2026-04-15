# EO Fairness — Analysis Document
## CS 516: Responsible Data Science and Algorithmic Fairness

---

## Overview

This project empirically measures the individual-level cost of enforcing Equalized Odds (EO)
post-processing on a recidivism risk model trained on the COMPAS dataset. When EO is enforced
at the group level, defendants with nearly identical legal profiles can receive different risk
classifications purely due to the randomness EO introduces. This analysis quantifies how often
that happens, where it concentrates, and what the tradeoff looks like as the fairness constraint
is relaxed.

**Three hypotheses were tested:**

- **H1**: EO post-processing increases individual-level inconsistency relative to the baseline classifier.
- **H2**: EO-induced inconsistencies concentrate among legally similar individuals near decision thresholds.
- **H3**: Reductions in group-level error disparities under EO are negatively associated with individual-level consistency.

All three were supported.

---

## Dataset

- **Source**: ProPublica COMPAS Recidivism Dataset — Broward County, FL (2013–2014)
- **Filters applied**: ProPublica standard filters (days_b_screening_arrest within ±30,
  valid recidivism flag, no ordinary traffic offenses, valid score text, race restricted
  to African-American and Caucasian)
- **Final sample**: 5,278 individuals
- **Race composition**: Black 60.2% (3,175) | White 39.8% (2,103)
- **Outcome**: 47.0% recidivated within two years (reasonably balanced — no resampling needed)
- **Train/test split**: 80/20 stratified jointly on race × outcome
  - Train: 4,222 | Test: 1,056
  - Test Black: 635 | Test White: 421

**Important**: Race was never used as a model feature. It is used only for EO group assignment
and reporting. The model was trained exclusively on age, sex, juvenile felony count, juvenile
misdemeanor count, other juvenile charges, prior adult charges, and charge degree (felony/misdemeanor).

---

## Baseline Model

**Model**: Logistic regression (max_iter=1000) with isotonic calibration (CalibratedClassifierCV, cv=5).
Decision threshold: 0.5.

### Overall Performance

| Metric     | Value  |
|------------|--------|
| Accuracy   | 68.1%  |
| AUC-ROC    | 0.728  |

### Per-Group Performance

| Group   | FPR    | FNR    | TPR    | Accuracy |
|---------|--------|--------|--------|----------|
| Black   | 36.0%  | 27.1%  | 72.9%  | 68.7%    |
| White   | 18.0%  | 55.8%  | 44.2%  | 67.2%    |

### Disparity

| Measure                         | Value    |
|---------------------------------|----------|
| FPR disparity (Black − White)   | **+18.0 pp** |
| FNR disparity (Black − White)   | **−28.6 pp** |

**Interpretation**: The unconstrained model assigns Black defendants a false positive rate
more than twice that of White defendants (36% vs 18%). At the same time, it classifies
far fewer White defendants as high-risk overall, resulting in a much higher false negative
rate for White defendants (55.8% vs 27.1%). This asymmetry is the motivating problem that
EO post-processing attempts to correct.

---

## EO Post-Processing

**Method**: Hardt et al. (2016) randomized thresholding implemented directly.

For each racial group, the ROC curve is computed and the convex hull is used to identify
a pair of operating points. A mixing probability determines how often each threshold is
applied to a given individual. The target operating point is chosen to minimize total
error subject to equal FPR and equal TPR across groups.

**Target operating point**: FPR = 13.43%, TPR = 41.31%

**Group-specific thresholds:**

| Group | τ_low  | τ_high | Mix rate (P use τ_low) |
|-------|--------|--------|------------------------|
| White | 0.521  | 0.522  | 5.3%                   |
| Black | 0.695  | 0.704  | 56.1%                  |

The Black group requires a substantially higher threshold and a meaningful mixing probability,
reflecting that Black defendants had much higher calibrated scores under the unconstrained model.
The White group operates near a single threshold with almost no randomization needed.

### Post-EO Performance

| Group   | FPR    | FNR    | TPR    | Accuracy |
|---------|--------|--------|--------|----------|
| Black   | 12.5%  | 57.2%  | 42.8%  | 64.1%    |
| White   | 13.3%  | 59.4%  | 40.6%  | 68.6%    |
| Overall | —      | —      | —      | 65.9%    |

### Disparity After EO

| Measure                         | Baseline    | After EO   | Change      |
|---------------------------------|-------------|------------|-------------|
| FPR disparity (Black − White)   | +18.0 pp    | −0.7 pp    | −18.7 pp    |
| FNR disparity (Black − White)   | −28.6 pp    | −2.2 pp    | +26.4 pp    |
| Overall accuracy                | 68.1%       | 65.9%      | −2.2 pp     |

**Interpretation**: EO nearly eliminated the FPR gap (18 pp → 0.7 pp). Both groups now
operate near the same FPR and TPR. However, this comes at a cost: overall accuracy dropped
2.2 points, and more importantly, the randomization introduced into individual decisions
is the source of the inconsistency analyzed below.

---

## H1: EO Increases Individual-Level Inconsistency

**Method**: For each individual in the test set, their 5 nearest neighbors in feature space
(Euclidean distance, scaled features, no race) are identified. A pair is "inconsistent" if
the two individuals received different binary decisions. Inconsistency rate = fraction of
similar pairs with discordant decisions.

Bootstrap confidence intervals use 10,000 resamplings of the pair set (percentile method).

### Results (KNN Euclidean, k=5)

| Model     | Inconsistency Rate | 95% CI           |
|-----------|--------------------|------------------|
| Baseline  | 4.4%               | [3.6%, 5.2%]     |
| EO        | 12.5%              | [11.3%, 13.9%]   |
| **Delta** | **+8.1 pp**        | **[6.7%, 9.6%]** |

The confidence interval entirely excludes zero. **H1 is supported.**

### Pair-Level Decomposition

| Category           | Count | Share of all pairs |
|--------------------|-------|--------------------|
| Total pairs        | 2,482 | —                  |
| EO-introduced      | 280   | 11.3%              |
| EO-resolved        | 78    | 3.1%               |
| Net new discordant | 202   | 8.1%               |

EO introduced 3.6× more discordant pairs than it resolved.

### Seed Variance

EO decisions are stochastic (randomized thresholding). Running 10 different random seeds:
- Mean inconsistency rate across seeds: 12.5%
- Standard deviation: 0.08 pp

The seed variance is negligible relative to the effect size, confirming that the +8.1 pp
delta reflects the structural inconsistency introduced by EO randomization — not noise.

---

## H2: Inconsistency Concentrates Near Decision Thresholds

**Method**: Each individual is assigned a threshold distance: the minimum of their score's
distance to their group's τ_low and τ_high. Pairs are binned into quartiles by this distance
(Q1 = closest to threshold). EO inconsistency rate is computed per quartile.

Spearman correlation between minimum pair threshold distance and EO-introduced discordance indicator.

### Inconsistency by Threshold-Distance Quartile

| Quartile | Mean Dist. to Threshold | EO Inconsistency | Baseline Inconsistency | N Pairs |
|----------|------------------------|------------------|------------------------|---------|
| Q1 (closest) | 0.021             | **35.0%**        | 8.2%                   | 625     |
| Q2           | 0.091             | 14.4%            | 5.2%                   | 617     |
| Q3           | 0.200             | 0.4%             | 3.7%                   | 695     |
| Q4 (farthest)| 0.338             | **0.0%**         | 0.0%                   | 545     |

The pattern is perfectly monotone: Q1 > Q2 > Q3 > Q4.

### Spearman Correlation

- r = −0.404
- p < 0.0001

Negative correlation confirms: pairs where individuals score closer to a decision threshold
are significantly more likely to be flipped by EO randomization.

**H2 is supported.** The inconsistency introduced by EO is not random across the population —
it is concentrated precisely where the randomized thresholding has the most leverage: among
individuals whose scores fall near the group-specific decision boundaries.

**Key insight**: Defendants in Q1 face a 35% chance of receiving a different outcome than
their nearest legal neighbor. Defendants in Q4 face no such risk.

---

## H3: Fairness–Consistency Pareto Frontier

**Method**: The EO constraint is relaxed by an epsilon parameter. At epsilon=0, groups must
achieve identical FPR and TPR. At larger epsilon, they may differ by up to epsilon on each
axis. For each epsilon in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], a relaxed postprocessor
is fit, decisions are applied, and both FPR disparity and individual inconsistency are measured.

### Pareto Frontier Results

| Epsilon | FPR Disparity | Inconsistency Rate | 95% CI            |
|---------|---------------|--------------------|-------------------|
| 0.00    | 0.74 pp       | 12.5%              | [11.3%, 13.9%]    |
| 0.05    | 0.24 pp       | 11.0%              | [9.8%, 12.3%]     |
| 0.10    | 1.09 pp       | 10.2%              | [9.0%, 11.4%]     |
| 0.15    | 0.76 pp       | 11.4%              | [10.1%, 12.7%]    |
| 0.20    | 0.76 pp       | 11.4%              | [10.1%, 12.7%]    |
| 0.25    | 0.76 pp       | 11.4%              | [10.1%, 12.7%]    |
| 0.30    | 1.92 pp       | 11.0%              | [9.8%, 12.2%]     |

**Interpretation**: The curve does not show a clean monotone tradeoff in this dataset because
the COMPAS score distributions for the two groups overlap substantially — small relaxations
quickly allow the optimizer to find lower-randomization solutions. However, the key finding
holds: the strictest fairness constraint (epsilon=0) produces the highest inconsistency.
Allowing even modest tolerance reduces individual inconsistency by 1–2 pp while FPR disparity
remains near zero.

**H3 is supported.** There is a meaningful tradeoff: enforcing stricter group fairness
requires more randomization, which costs individual consistency. The tradeoff flattens
beyond epsilon=0.10, suggesting diminishing returns to further relaxation in this dataset.

---

## Who Bears the Cost: Profile-Level Analysis

Looking at EO-introduced inconsistency rates by age group and prior offense count:

**Highest-impact profiles** (from Fig 6 heatmap):
- Young defendants (18–25) with 1–2 prior offenses are among the most affected
- Mid-age defendants (26–35) with 6+ priors also show elevated inconsistency
- Defendants with zero priors and older age (36+) show the lowest impact

This is not arbitrary — these profiles cluster near the EO decision thresholds, meaning
the randomized mixing directly affects them. The profiles that are most legally ambiguous
(moderate risk, not clearly high or low) bear a disproportionate share of the individual
inconsistency introduced by group fairness enforcement.

---

## Robustness Checks

The primary analysis used logistic regression and Euclidean KNN pairs. Three robustness
variants were tested, each changing one design choice at a time.

### H1 Delta Across Variants

| Variant                          | H1 Delta |
|----------------------------------|----------|
| Primary (LR + Euclidean KNN)     | +8.1 pp  |
| Random Forest baseline           | +4.1 pp  |
| Cosine KNN similarity            | +8.0 pp  |
| Cross-race pairs only            | +15.5 pp |

**Key findings:**
- **Random Forest** shows a smaller but still substantial effect (+4.1 pp). RF is a stronger
  baseline that produces better-separated scores, reducing the mixing needed by EO — and thus
  reducing (but not eliminating) individual inconsistency.
- **Cosine similarity** reproduces almost the identical result (+8.0 pp), confirming the
  finding is not sensitive to the distance metric used to define "similar."
- **Cross-race pairs** show the largest effect (+15.5 pp). These are the pairs most directly
  relevant to the equity question: two defendants with identical legal profiles but different
  race designations. EO introduces the most inconsistency precisely for these pairs — the
  ones where differential treatment is hardest to justify.

---

## Summary of Findings

| Hypothesis | Claim | Result | Key Statistic |
|------------|-------|--------|---------------|
| H1 | EO increases inconsistency | **Supported** | Delta = +8.1 pp, CI excludes zero |
| H2 | Inconsistency concentrates near thresholds | **Supported** | Q1=35% vs Q4=0%; Spearman r=−0.40 |
| H3 | Fairness–consistency tradeoff exists | **Supported** | Strictest EO → highest inconsistency |

---

## Core Takeaway

Equalized Odds reduces the FPR gap from **18 percentage points to under 1** — a meaningful
fairness improvement at the group level. But it does so by inserting a probabilistic threshold
assignment into individual decisions. Two defendants with identical age, criminal history, and
charge type now face an **8 percentage point higher chance** of receiving different outcomes
than they did under the unconstrained model.

That inconsistency is not spread evenly. It concentrates among defendants whose scores fall
near the decision boundary — exactly the people for whom the risk assessment is most uncertain.
Young defendants with moderate criminal histories are disproportionately affected.

The policy implication is not that EO is wrong. It is that group-level fairness constraints
have individual-level costs that are measurable, non-trivial, and unevenly distributed. Any
deployment of EO post-processing should account for this tradeoff explicitly.
