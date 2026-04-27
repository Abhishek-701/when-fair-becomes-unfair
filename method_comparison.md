# Method Comparison: Three Post-Processors for Fair Recidivism Prediction

## COMPAS Dataset — Broward County, FL | Test Set: n = 1,056

---

## Why This Comparison Matters

A
divism prediction model is only as fair as the decisions it produces. The baseline logistic regression achieves 68% accuracy but assigns Black defendants a false positive rate more than twice that of White defendants (36% vs 18%). Three post-processing methods have been developed and evaluated to correct this. Each method represents a different answer to the same question:

> **How do you make an algorithmic decision system fairer — and what does fairness cost?**

The comparison below shows that fairness is not a single dimension. Group fairness (equal error rates across races) and individual fairness (similar defendants get similar decisions) are distinct, sometimes conflicting objectives. No single method wins on everything. Understanding what each method does — and what it sacrifices — is essential for anyone deploying risk assessment tools.

---

## The Three Methods

---

### Method 1: Hardt EO (Equalized Odds Post-Processing)

**Origin**: Hardt, Price & Srebro (2016), *"Equality of Opportunity in Supervised Learning"*

**Core idea**: After the model produces calibrated probability scores, adjust the decision rule separately for each racial group so that both groups operate at the same false positive rate and true positive rate. Mathematically, this means finding a single (FPR, TPR) operating point on the convex hull of each group's ROC curve and using randomized thresholding to reach it.

**How it works mechanically**:
Each defendant is assigned to their racial group. For each group, the EO postprocessor fits two thresholds (τ_low, τ_high) and a mixing probability. When a decision is made for a specific individual, a coin is flipped with probability equal to the mixing rate. Heads → use τ_low (more permissive); tails → use τ_high (more restrictive). The mixing probability is chosen so the group's expected FPR matches the joint target.

For Black defendants in this dataset, the mixing rate is **56.1%** — more than half of all Black defendants near the threshold get a randomized decision on every prediction run. White defendants have a mixing rate of only **5.3%**, because their score distribution is already closer to the EO target.

**What it is designed to optimize**: Binary group fairness. It directly constrains the binary decisions so that FPR and FNR are equal across groups. It makes no claim about individual consistency.

**Key limitation**: The coin-flip randomization that achieves group parity is the same mechanism that introduces individual inconsistency. Two Black defendants with identical legal profiles can receive opposite decisions purely because of how the random seed resolves. This is not a bug — it is the direct cost of the EO constraint when score distributions differ between groups.

---

### Method 2: Joint QP (Lipschitz-Constrained Quadratic Program)

**Origin**: Extension developed in this project, combining Hardt EO with Petersen-style individual fairness in a unified optimization.

**Core idea**: Instead of adjusting binary decisions after the fact, solve a Quadratic Program that finds the probability vector `p` closest to the original scores `s` while simultaneously satisfying:
1. A soft group fairness constraint: the mean predicted probability for Black defendants (among true negatives) must equal that for White defendants within tolerance epsilon
2. A Lipschitz constraint: for every pair of KNN-connected defendants (i, j), |p_i − p_j| ≤ L · d(x_i, x_j)

The second constraint is the individual fairness component. It says: two defendants who are similar in feature space (age, priors, charge type) cannot receive predicted probabilities that differ by more than L times their distance. Smaller L = stricter individual fairness.

**How it works mechanically**:
The QP is built using CVXPY with an OSQP backend. The objective is a sum of squared deviations from the original scores. The Lipschitz constraints form a sparse linear system (one inequality per KNN edge, per direction). The EO constraints are two linear constraints on the group means of p. OSQP solves this in seconds for n=1,056.

The output `p_opt` is a continuous probability vector, not binary decisions. Binary decisions are obtained by thresholding at 0.5.

**What it is designed to optimize**: Individual fairness via the Lipschitz constraint, with group fairness as a soft regularizer. The QP ensures that the continuous probabilities are both individually consistent and group-fair in expectation.

**Key limitation**: The EO constraint in the QP is enforced on **continuous** probabilities (expected FPR = mean of p for the group), not on **binary** decisions. After thresholding at 0.5, the binary FPR gap can be large even though the continuous constraint was satisfied. In practice, the binary FPR gap after thresholding is 13.98% — worse than the unconstrained baseline.

---

### Method 3: Two-Stage Pipeline (Lipschitz QP → Hardt EO)

**Origin**: Developed in this project as a direct response to the gap identified between Methods 1 and 2.

**Core idea**: Chain both methods sequentially to get binary group fairness (from Stage 2) and reduced individual inconsistency (from Stage 1):

```
raw scores s
     │
     ▼
[Stage 1: solve_joint()]
  Lipschitz QP at chosen L
     │
     ▼
p_opt  (smoothed scores — similar defendants have similar values)
     │
     ▼
[Stage 2: fit_eo_postprocessor() → apply_eo_decisions()]
  Hardt EO applied to p_opt
     │
     ▼
binary decisions  (binary EO guaranteed; lower inconsistency than Hardt-on-s)
```

**How it works mechanically**:
Stage 1 produces `p_opt` in which KNN-similar defendants have been forced closer together in probability space. Stage 2 applies Hardt's randomized thresholding to `p_opt` rather than to the raw scores `s`. Because `p_opt` is smoother than `s`, fewer defendants fall in the randomized mixing zone near each group's threshold — and those who do receive less disruptive randomization because their neighbors already have similar scores. Binary EO is still guaranteed by Stage 2's mechanism. Decisions are averaged across 10 random seeds to reduce seed-specific variance.

**What it is designed to optimize**: Both binary group fairness (guaranteed by Hardt in Stage 2) and reduced individual inconsistency (structural benefit from Stage 1 smoothing). It does not minimize inconsistency to the same level as the standalone Joint QP, but it achieves both objectives simultaneously — something neither method alone can do.

**Key limitation**: The FNR gap increases relative to Method 1 at tight L values (4.45% at L=0.10 vs 2.17% for Hardt-on-s). The QP reshapes the score distribution in a way that Hardt EO can equalize FPR well but has slightly more difficulty equalizing FNR.

---

## Side-by-Side Comparison

All metrics on the test set (n = 1,056). Two-stage results shown at L = 0.10 (best overall) and L = 0.35 (best FNR parity).

---

### Comparison 1: Accuracy

| Method | Accuracy | Delta vs Baseline |
|--------|----------|------------------|
| Baseline LR (unconstrained) | 68.09% | — |
| Method 1: Hardt EO | 65.81% | −2.28 pp |
| Method 2: Joint QP (L=0.05) | 65.53% | −2.56 pp |
| Method 3: Two-stage (L=0.10) | 66.00% | −2.09 pp |
| Method 3: Two-stage (L=0.35) | 66.10% | −1.99 pp |

**Why it matters**: Accuracy is the most visible metric to practitioners and courts reviewing the system. Every fairness intervention sacrifices some predictive performance because it moves decisions away from the pure risk-minimizing threshold toward a fairness-constrained one.

**Reading the numbers**: All three methods lose roughly 2 percentage points — about 21 additional wrong predictions out of 1,056. The two-stage actually recovers 0.1–0.3 pp of accuracy compared to Method 1, because the QP smoothing stage lets Hardt find a better-positioned operating point on the ROC curve. The Joint QP loses the most (−2.56 pp) because its Lipschitz constraint forces probabilities farthest from the original scores.

**Key insight**: The accuracy cost of achieving fairness is nearly the same across all three methods. The choice between them is not primarily about accuracy — it is about *which kind* of fairness you are buying with that shared cost.

---

### Comparison 2: Group Fairness — FPR Gap

| Method | FPR Black | FPR White | FPR Gap | Change vs Baseline |
|--------|-----------|-----------|---------|-------------------|
| Baseline LR | 36.0% | 18.0% | **18.0 pp** | — |
| Method 1: Hardt EO | 12.87% | 13.28% | **0.41%** | −17.6 pp |
| Method 2: Joint QP (L=0.05) | 32.34% | 18.36% | **13.98%** | −4.0 pp |
| Method 3: Two-stage (L=0.10) | ~13% | ~14% | **0.80%** | −17.2 pp |
| Method 3: Two-stage (L=0.35) | ~13% | ~13% | **0.27%** | −17.7 pp |

**Why it matters**: FPR parity means that a Black defendant and a White defendant with the same true recidivism risk face the same probability of being incorrectly flagged as high-risk. A high false positive rate for Black defendants means innocent (non-recidivist) Black defendants are more likely to be detained, denied bail, or given harsher sentences. This is the central racial fairness concern in risk assessment and the direct motivation for ProPublica's COMPAS analysis.

**Reading the numbers**: Method 1 and the Two-stage pipeline both achieve near-perfect FPR parity (< 1 pp gap), because both use Hardt's mechanism in their final decision stage. The Joint QP fails on this metric: its binary FPR gap is 13.98% — not much better than the unconstrained baseline. This is the central weakness of Method 2 when measured on binary decisions.

**Key insight**: FPR parity requires a binary decision mechanism that directly enforces it. Methods that optimize continuous probabilities (Method 2) and then hard-threshold do not inherit continuous parity. Methods that operate directly on binary decisions (Methods 1 and 3) do.

---

### Comparison 3: Group Fairness — FNR Gap

| Method | FNR Black | FNR White | FNR Gap | Direction |
|--------|-----------|-----------|---------|-----------|
| Baseline LR | 27.1% | 55.8% | **28.6 pp** (Black lower) | Black benefits |
| Method 1: Hardt EO | 57.23% | 59.39% | **2.17%** | Near-equal |
| Method 2: Joint QP (L=0.05) | 37.35% | 57.58% | **20.23%** | Black benefits |
| Method 3: Two-stage (L=0.10) | ~42% | ~46% | **4.45%** | Near-equal |
| Method 3: Two-stage (L=0.35) | ~42% | ~44% | **1.99%** | Near-equal |

**Why it matters**: FNR parity means that Black and White defendants with the same true recidivism risk face the same probability of being incorrectly cleared as low-risk. A high FNR for one group means actual recidivists from that group are more likely to be released — which creates public safety and differential-treatment concerns. EO requires both FPR parity *and* FNR parity simultaneously.

**Reading the numbers**: The baseline model has wildly unequal FNR (27.1% for Black, 55.8% for White). Method 1 achieves near-perfect parity (2.17% gap) — the best of all three methods on this metric. The Joint QP (Method 2) leaves a 20.23% FNR gap — Black defendants are still flagged at much higher rates, just not as extreme as the baseline. The two-stage at L=0.10 achieves 4.45% gap (reasonable), and at L=0.35 achieves 1.99% (almost as good as Method 1).

**Key insight**: The two-stage at L=0.35 is the only configuration that achieves both FPR gap < 1% and FNR gap < 2% simultaneously, making it the best-performing method on pure group fairness when both error rates are considered.

---

### Comparison 4: Individual Fairness — Overall Inconsistency Rate

| Method | IR | 95% CI | Delta vs Baseline |
|--------|-----|--------|-------------------|
| Baseline LR | 4.4% | [3.6%, 5.2%] | — |
| Method 1: Hardt EO | 12.41% | [11.3%, 13.9%] | **+8.0 pp** ↑ |
| Method 2: Joint QP (L=0.05) | 1.89% | — | **−2.5 pp** ↓ |
| Method 3: Two-stage (L=0.10) | 8.58% | [8.0%, 10.1%] | **+4.2 pp** ↑ |
| Method 3: Two-stage (L=0.35) | 10.60% | — | **+6.2 pp** ↑ |

Inconsistency rate = fraction of KNN-similar pairs (5 nearest neighbors, Euclidean distance) that receive different binary decisions.

**Why it matters**: Individual fairness asks a different question than group fairness: not "are the *groups* treated equally?" but "are *similar individuals* treated equally?" Two defendants with the same age, same prior offense count, same charge type, and same juvenile history should not receive opposite risk classifications. When they do, the system is treating legally indistinguishable people differently — a violation of basic procedural fairness independent of any group-level consideration.

**Reading the numbers**: Method 1 makes this dramatically *worse* — it triples the inconsistency rate. The randomized coin-flip is the mechanism: similar defendants near the threshold receive independent coin tosses, so discordant pairs are common. Method 2 reduces inconsistency below the baseline (1.89%) — the Lipschitz constraint directly enforces that similar individuals get similar probabilities. Method 3 sits in the middle: better than Method 1, worse than Method 2.

**Key insight**: The Joint QP (Method 2) is the only method that improves individual fairness below the unconstrained baseline. All other methods that enforce binary group parity increase inconsistency because any randomized threshold mechanism introduces decision noise for individuals near the boundary.

---

### Comparison 5: Individual Fairness — Cross-Race Inconsistency Rate

| Method | Cross-Race IR | Delta vs Method 1 |
|--------|--------------|-------------------|
| Method 1: Hardt EO | 20.34% | — |
| Method 2: Joint QP (L=0.05) | 1.31% | −19.0 pp |
| Method 3: Two-stage (L=0.10) | 16.14% | −4.2 pp |
| Method 3: Two-stage (L=0.35) | 18.75% | −1.6 pp |

Cross-race IR = inconsistency rate computed only on pairs where the two defendants belong to different racial groups.

**Why it matters**: Cross-race inconsistency is the most policy-relevant individual fairness metric. It directly captures the scenario where a Black defendant and a White defendant with the same legal profile receive different risk classifications. This is neither a group fairness failure nor a random individual fairness failure — it is differential treatment of people who are identical in every measured dimension except race. Even though race is not used as a model feature, the post-processing step can introduce race-correlated inconsistency through the group-specific thresholds.

**Reading the numbers**: Under Method 1, 1 in 5 cross-race similar pairs receives discordant decisions. Under Method 2, this drops to 1 in 76 — a 15× improvement, the most dramatic result in the entire analysis. Under Method 3, it drops to 1 in 6 (16.14%) — a meaningful improvement over Method 1 but nowhere near Method 2.

**Key insight**: Cross-race inconsistency under Method 1 (20.34%) is the most ethically concerning number in the analysis. Two defendants who differ only in race and receive different algorithmic risk scores have not received fair treatment by any reasonable standard. Method 2 nearly eliminates this. Method 3 reduces it moderately while preserving binary group parity.

---

### Comparison 6: The Mixing Rate — Mechanism Behind the Numbers

| Method | Mix rate: Black | Mix rate: White | Interpretation |
|--------|----------------|-----------------|---------------|
| Method 1: Hardt-on-s | 56.1% | 5.3% | Majority of Black defendants get coin-flip decisions |
| Method 3: Two-stage (L=2.00) | 14.7% | 24.0% | Low smoothing — QP shifts target, modest mixing |
| Method 3: Two-stage (L=0.35) | 45.7% | 15.8% | Moderate smoothing — mixing shifts but stays high |
| Method 3: Two-stage (L=0.10) | 86.1% | 2.7% | Tight smoothing — high mix rate but narrow window |
| Method 3: Two-stage (L=0.05) | 20.3% | 0.8% | Very tight — near-deterministic |

**Why it matters**: The mixing rate is the direct driver of individual inconsistency in any Hardt-style method. A mixing rate of 56.1% for Black defendants means that in repeated predictions for the same individual (or for their KNN-similar neighbor), the decision will be different more than half the time. This is the mechanism that H2 in Phase 1 identified and that explains the entire inconsistency problem.

**Reading the numbers**: Stage 1 QP fundamentally changes the score distribution that Stage 2 Hardt operates on, shifting the EO operating point and the mixing rates. At L=0.10, the Black mixing rate *rises* to 86% — but this is in a world where the score distribution is so compressed that even a high mixing rate produces fewer inconsistencies, because the scores at which individuals sit relative to the threshold are more tightly clustered. At L=0.35, the mix rate (45.7%) is lower than Method 1's 56.1%, and the EO target is similarly positioned — but the smoother p_opt means fewer KNN pairs straddle the threshold boundary.

**Key insight**: Mixing rate alone does not determine inconsistency — it interacts with the score distribution's shape near the threshold. The QP's contribution is to reshape that distribution so the mixing mechanism has less opportunity to produce discordant pairs.

---

## Unified Scorecard

Ratings: ✅ Strong / ⚠️ Moderate / ❌ Weak

| Objective | Method 1: Hardt EO | Method 2: Joint QP | Method 3: Two-stage |
|-----------|-------------------|-------------------|---------------------|
| Binary FPR parity | ✅ 0.41% gap | ❌ 13.98% gap | ✅ 0.27–0.80% gap |
| Binary FNR parity | ✅ 2.17% gap | ❌ 20.23% gap | ✅ 2.0–4.5% gap |
| Individual fairness (all pairs) | ❌ 12.41% (worse than baseline) | ✅ 1.89% (better than baseline) | ⚠️ 8.58–10.60% |
| Cross-race fairness | ❌ 20.34% | ✅ 1.31% | ⚠️ 16.14–18.75% |
| Accuracy | ⚠️ 65.81% | ⚠️ 65.53% | ⚠️ 66.00–66.10% |
| Deterministic decisions | ❌ Randomized | ✅ Deterministic | ❌ Randomized |
| Binary EO guaranteed | ✅ Yes | ❌ No | ✅ Yes |

---

## When to Use Each Method

**Use Method 1 (Hardt EO)** when:
- Legal or regulatory compliance requires demonstrably equal binary error rates across groups
- Individual consistency is not a formal requirement
- The simplest, most auditable post-hoc correction is preferred
- You can tolerate the randomization — e.g., when the system is used for population-level policy rather than individual case decisions

**Use Method 2 (Joint QP)** when:
- Individual consistency is the primary fairness concern
- Decisions are made by continuous probability scores (not binary flags), so soft EO is sufficient
- You want deterministic predictions (same score always → same output)
- Group fairness will be monitored separately at the reporting level

**Use Method 3 (Two-stage)** when:
- Both binary group fairness and reduced individual inconsistency are required
- The deployment context requires auditable group parity (like a legal standard) but also faces scrutiny for treating similar individuals differently
- FPR gap must be below 1% but you can accept FNR gap up to ~4.5%
- Operating point: L = 0.35 for best joint FPR + FNR parity; L = 0.10 for lowest inconsistency rate

---

## The Fundamental Tradeoff, Stated Plainly

All three methods are trying to correct the same baseline disparity. Each makes a different choice about what "fair" means:

- **Method 1** says: fair means equal error rates across groups. It achieves this — but introduces a new problem: similar defendants can now receive opposite decisions, purely due to randomization. It fixes the group problem and creates an individual problem.

- **Method 2** says: fair means similar defendants get similar decisions. It achieves this — but cannot guarantee that the binary decisions satisfy group parity. It fixes the individual problem and leaves the group problem partially unaddressed.

- **Method 3** says: both matter, and there is a way to get most of both. By smoothing the scores before applying Hardt, it reduces (but does not eliminate) the individual inconsistency that Hardt introduces, while preserving Hardt's binary group fairness guarantee. It is a genuine improvement over Method 1 at no meaningful accuracy cost, but cross-race inconsistency (16%) remains far higher than Method 2 (1.3%).

No method simultaneously achieves FPR gap < 1%, FNR gap < 3%, and IR < 5%. That combination remains an open problem in this dataset — and likely reflects a genuine tension between the amount of distribution-correction that binary EO requires and the smoothness that low inconsistency demands.
