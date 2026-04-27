# Extension: Joint EO + Feature-Space Lipschitz Post-Processor

You are extending an existing fairness research pipeline. The first phase
(baseline logistic regression + Equalized Odds post-processing + inconsistency
measurement) is already complete. Do not touch or refactor any existing code.
Your job is to add a self-contained extension in `src/extension/`.

## What already exists (read-only)

- `src/config.py` — RANDOM_SEED, paths, K_NEIGHBORS, EPSILON_GRID
- `src/data_prep.py` — produces X_nr (scaled, race-excluded), y, race, train/test splits
- `src/baseline_model.py` — calibrated logistic regression, outputs `s_i` (averaged probabilities)
- `src/eo_postprocessor.py` — Hardt EO, decisions stored as averaged probabilities across 10 seeds
- `src/similarity.py` — KNN graph already built with Euclidean distance on X_nr, saved to
  `results/knn_pairs_euclidean.csv` with columns: id_i, id_j, distance, race_i, race_j, pair_type
- `src/metrics.py` — inconsistency_rate(), bootstrap_ci(), decompose_inconsistency()
- `results/baseline_scores.csv` — columns: id, score, decision_baseline
- `results/eo_decisions_mean.csv` — columns: id, decision_EO (averaged across seeds)
- `results/knn_pairs_euclidean.csv` — the KNN similarity graph
- `results/eo_postprocessor.json` — fitted EO parameters (thresholds, mix rates per group)

## Your task

Implement a novel joint post-processor that simultaneously enforces:
1. Equalized Odds (equal FPR and FNR across racial groups)
2. A feature-space Lipschitz constraint: |p_i - p_j| ≤ L · d(x_i, x_j) for
   all KNN-connected pairs (i, j), where d is the Euclidean distance already
   computed in the similarity graph

This is a Quadratic Program (QP). The objective is to minimize deviation from
the baseline calibrated scores while satisfying both constraint sets jointly.

---

## File structure to create

```
src/extension/
    __init__.py
    joint_postprocessor.py   # core QP solver
    feasibility.py           # feasibility analysis across (L, epsilon) grid
    comparison.py            # compare all three methods head-to-head
notebooks/
    extension_analysis.ipynb # end-to-end notebook for the extension
results/extension/           # all outputs go here
```

---

## Module 1: `src/extension/joint_postprocessor.py`

### Problem formulation

Given:
- `s` — vector of calibrated baseline scores (n,), from results/baseline_scores.csv
- `race` — binary vector (n,), 1=Black 0=White
- `y` — true labels (n,)
- `pairs` — KNN edge list with distances, from results/knn_pairs_euclidean.csv
- `L` — Lipschitz constant (hyperparameter, controls individual fairness strictness)
- `epsilon` — EO relaxation tolerance (0 = exact EO, >0 = approximate)

Solve:

    minimize    ||p - s||²
    subject to:
        0 ≤ p_i ≤ 1                          for all i         (valid probabilities)
        |FPR_Black(p) - FPR_White(p)| ≤ eps  (EO: FPR parity)
        |FNR_Black(p) - FNR_White(p)| ≤ eps  (EO: FNR parity)
        |p_i - p_j| ≤ L · d_ij               for all (i,j) in KNN edges

### Linearizing the EO constraints

FPR_g(p) = sum(p_i for i in group g with y_i=0) / |{i: race_i=g, y_i=0}|

This is linear in p. So the EO constraints become:

    |mean(p_i : race_i=1, y_i=0) - mean(p_i : race_i=0, y_i=0)| ≤ eps
    |mean(p_i : race_i=1, y_i=1) - mean(p_i : race_i=0, y_i=1)| ≤ eps  [FNR via TPR]

Each absolute value inequality splits into two linear constraints.

### Linearizing the Lipschitz constraints

|p_i - p_j| ≤ L · d_ij splits into:
    p_i - p_j ≤ L · d_ij
    p_j - p_i ≤ L · d_ij

With K=5 neighbors per node, this gives ~5n pairwise constraints — tractable.

### Implementation

Use `cvxpy` to set up and solve the QP. Do NOT use scipy.optimize — cvxpy
handles the constraint structure cleanly and is the standard tool for this.

```python
import cvxpy as cp
import numpy as np
import pandas as pd

def solve_joint(s, y, race, pairs_df, L, epsilon, verbose=False):
    """
    Solve the joint EO + Lipschitz QP.

    Parameters
    ----------
    s : np.ndarray (n,)
        Baseline calibrated scores.
    y : np.ndarray (n,)
        True binary labels.
    race : np.ndarray (n,)
        Binary race indicator (1=Black, 0=White).
    pairs_df : pd.DataFrame
        KNN pairs with columns [id_i, id_j, distance]. ids are integer
        positional indices into s/y/race arrays (0-based).
    L : float
        Lipschitz constant. Larger = looser individual fairness constraint.
    epsilon : float
        EO tolerance. 0 = exact parity.
    verbose : bool
        Pass to cp.Problem.solve().

    Returns
    -------
    dict with keys:
        'p_opt'     : np.ndarray (n,) — optimal decision probabilities
        'status'    : str — solver status ('optimal', 'infeasible', etc.)
        'obj_value' : float — objective value at solution
        'fpr_gap'   : float — achieved |FPR_Black - FPR_White|
        'fnr_gap'   : float — achieved |FNR_Black - FNR_White|
        'lipschitz_violations' : int — number of KNN pairs violating the
                                 Lipschitz constraint at solution (should be 0)
    """
```

### Critical implementation notes

- Map id_i, id_j from pairs_df to positional indices. The pairs_df uses the
  same integer ids as results/baseline_scores.csv column 'id'. Build an
  id→index mapping before constructing constraints.
- If the solver returns 'infeasible', return the dict with p_opt=None and
  status='infeasible'. Do not raise an exception — infeasibility is an
  expected and scientifically interesting outcome for small L + small epsilon.
- Use solver=cp.OSQP with max_iter=10000. OSQP handles large sparse QPs well.
  Fall back to cp.SCS if OSQP fails.
- After solving, always verify: count Lipschitz violations and EO gap on the
  returned p_opt. Log warnings if either is nonzero beyond floating point noise
  (threshold: 1e-4).

### Function: `binarize(p_opt, threshold=0.5) -> np.ndarray`

Convert continuous p_opt to binary decisions at given threshold.
Return binary array.

### Function: `solve_joint_grid(s, y, race, pairs_df, L_grid, epsilon_grid) -> pd.DataFrame`

Run solve_joint over the full (L, epsilon) grid. Return a dataframe with
columns: L, epsilon, status, obj_value, fpr_gap, fnr_gap,
lipschitz_violations, inconsistency_rate, accuracy, auc.

Compute inconsistency_rate by importing from src.metrics.
Save to results/extension/grid_results.csv.

---

## Module 2: `src/extension/feasibility.py`

### Function: `compute_feasibility_frontier(grid_df) -> pd.DataFrame`

From the grid results, identify for each epsilon value the minimum L at which
the QP is feasible. Return a dataframe with columns: epsilon, L_min_feasible,
inconsistency_rate_at_Lmin, fpr_gap_at_Lmin.

This is the core novel finding: the feasibility frontier in (L, epsilon) space
tells you exactly how much individual fairness you must sacrifice to achieve
group fairness at each level of EO strictness.

### Function: `plot_feasibility_heatmap(grid_df, save_path)`

Heatmap where x=epsilon, y=L, cell color = 'feasible' (green) or 'infeasible'
(red). Overlay contour lines for inconsistency rate on the feasible region.
Save to results/extension/figures/feasibility_heatmap.png at 300 DPI.

### Function: `plot_three_way_frontier(grid_df, save_path)`

3D surface or 2D Pareto plot showing the tradeoff between:
- Group fairness (FPR gap, x-axis)  
- Individual fairness (min feasible L, y-axis)
- Accuracy (color or z-axis)

Save to results/extension/figures/three_way_frontier.png at 300 DPI.

---

## Module 3: `src/extension/comparison.py`

### Function: `compare_all_methods(s, y, race, pairs_df, L_star, epsilon_star) -> pd.DataFrame`

Compare three post-processors head-to-head at a chosen operating point
(L_star, epsilon_star):

1. **Hardt EO** — load from results/eo_decisions_mean.csv (already computed)
2. **Petersen-style IF only** — graph Laplacian smoothing with NO EO constraint.
   Implement as: minimize ||p - s||² + tau * p^T L_graph p, where L_graph is
   the graph Laplacian of the KNN graph. Sweep tau to match the same
   inconsistency rate as the joint method for fair comparison.
3. **Joint (yours)** — solve_joint at (L_star, epsilon_star)

For each method compute:
- accuracy, AUC
- FPR_Black, FPR_White, FPR gap
- FNR_Black, FNR_White, FNR gap  
- inconsistency_rate (KNN pairs, bootstrapped CI)
- inconsistency_rate_cross_race (cross-race pairs only)

Return a formatted comparison dataframe. Save to
results/extension/method_comparison.csv.

### Function: `plot_comparison_radar(comparison_df, save_path)`

Radar/spider chart with 6 axes: accuracy, AUC, FPR gap (inverted), FNR gap
(inverted), individual consistency, cross-race consistency. One line per
method. Save to results/extension/figures/radar_comparison.png at 300 DPI.

---

## Grids to use

```python
L_GRID = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00]
EPSILON_GRID_EXT = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
```

Start at large L (loose individual fairness) and small epsilon (strict EO) and
work inward. The infeasible region will be at small L + small epsilon. This
ordering helps fail fast on infeasible cells rather than waiting for the solver
to time out.

---

## Notebook: `notebooks/extension_analysis.ipynb`

Structure:

1. **Setup** — load all existing results (scores, EO decisions, KNN pairs, y, race)
2. **Single solve demo** — run solve_joint at L=1.0, epsilon=0.05, print diagnostics
3. **Feasibility grid** — run solve_joint_grid, show heatmap
4. **Feasibility frontier** — plot L_min_feasible vs epsilon
5. **Operating point selection** — pick L_star, epsilon_star on the frontier
   that balances all three objectives; justify the choice explicitly
6. **Three-way comparison** — compare_all_methods at chosen operating point
7. **Key findings** — markdown cells stating H1-H3 analog results for the extension:
   - What is the minimum L required for feasibility at exact EO (epsilon=0)?
   - Does the joint constraint strictly dominate either alone?
   - What accuracy cost does joint fairness impose?

---

## Dependencies to add to requirements.txt

```
cvxpy>=1.4.0
osqp>=0.6.0
```

---

## Constraints

- Import RANDOM_SEED from src.config in every module.
- Every function must have a docstring with Parameters and Returns sections.
- The notebook must run top-to-bottom on a clean environment.
- Do not modify any existing src/ files outside src/extension/.
- Save every intermediate result as a CSV before plotting — plots should be
  regeneratable from CSVs without re-solving.
- If cvxpy is not installed, raise ImportError with a clear message pointing
  to the install command: `pip install cvxpy osqp`.
