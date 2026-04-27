"""
Extension Module 1: Joint EO + Feature-Space Lipschitz post-processor via QP.

Simultaneously enforces:
  1. Equalized Odds (equal FPR and FNR across racial groups)
  2. Feature-space Lipschitz constraint: |p_i - p_j| <= L * d(x_i, x_j)
     for all KNN-connected pairs.

The objective is to minimize ||p - s||^2 (deviation from baseline scores).
Implemented as a Quadratic Program using cvxpy with OSQP backend.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

# Ensure src/ is on the path so sibling modules are importable
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    import cvxpy as cp
except ImportError as exc:
    raise ImportError(
        "cvxpy is required for the joint post-processor. "
        "Install with: pip install cvxpy osqp"
    ) from exc

from config import RANDOM_SEED, RESULTS_DIR  # noqa: E402 (after sys.path patch)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EXT_RESULTS_DIR = os.path.join(RESULTS_DIR, "extension") + os.sep

L_GRID = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00]
EPSILON_GRID_EXT = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]

_FEASIBLE_STATUSES = {"optimal", "optimal_inaccurate"}


def solve_joint(s, y, race, pairs_df, L, epsilon, ids=None, verbose=False):
    """
    Solve the joint EO + Lipschitz Quadratic Program.

    Minimizes ||p - s||^2 subject to:
      - 0 <= p_i <= 1 for all i
      - |FPR_Black(p) - FPR_White(p)| <= epsilon
      - |FNR_Black(p) - FNR_White(p)| <= epsilon
      - |p_i - p_j| <= L * d_ij for all KNN edges (i, j)

    Parameters
    ----------
    s : np.ndarray, shape (n,)
        Baseline calibrated scores.
    y : np.ndarray, shape (n,)
        True binary labels.
    race : np.ndarray, shape (n,)
        Binary race indicator (1=Black, 0=White).
    pairs_df : pd.DataFrame
        KNN pairs with columns [id_i, id_j, distance]. IDs match those in
        results/baseline_scores.csv. If ids is None, IDs are treated as
        0-based positional indices into s/y/race.
    L : float
        Lipschitz constant. Larger values allow more individual-fairness
        relaxation (looser constraint).
    epsilon : float
        EO tolerance. 0 = exact parity.
    ids : np.ndarray or None
        Original IDs corresponding to positions 0..n-1 in s, y, race.
        Required when pairs_df uses non-positional IDs (the common case).
    verbose : bool
        If True, pass verbose flag to the CVXPY solver.

    Returns
    -------
    dict with keys:
        'p_opt'               : np.ndarray (n,) or None if infeasible
        'status'              : str solver status string
        'obj_value'           : float or None
        'fpr_gap'             : float — |FPR_Black - FPR_White| at solution
        'fnr_gap'             : float — |FNR_Black - FNR_White| at solution
        'lipschitz_violations': int — KNN pairs violating Lipschitz at solution
    """
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=int)
    race = np.asarray(race, dtype=int)
    n = len(s)

    # --- Build id -> positional index mapping ---
    if ids is not None:
        id_to_pos = {int(orig_id): pos for pos, orig_id in enumerate(ids)}
    else:
        id_to_pos = {i: i for i in range(n)}

    # Filter pairs to those where both endpoints are in our index
    mapped_i = pairs_df["id_i"].map(id_to_pos)
    mapped_j = pairs_df["id_j"].map(id_to_pos)
    valid_mask = mapped_i.notna() & mapped_j.notna()
    pairs_valid = pairs_df[valid_mask]

    pos_i = mapped_i[valid_mask].astype(int).values
    pos_j = mapped_j[valid_mask].astype(int).values
    distances = pairs_valid["distance"].values.astype(float)

    # --- Group masks for EO constraints ---
    black_neg = (race == 1) & (y == 0)   # Black, y=0 → FPR group
    white_neg = (race == 0) & (y == 0)   # White, y=0 → FPR group
    black_pos = (race == 1) & (y == 1)   # Black, y=1 → TPR/FNR group
    white_pos = (race == 0) & (y == 1)   # White, y=1 → TPR/FNR group

    n_bn, n_wn = black_neg.sum(), white_neg.sum()
    n_bp, n_wp = black_pos.sum(), white_pos.sum()

    if min(n_bn, n_wn, n_bp, n_wp) == 0:
        raise ValueError(
            "A race×label group is empty — cannot form EO constraints. "
            f"Counts: Black-neg={n_bn}, White-neg={n_wn}, "
            f"Black-pos={n_bp}, White-pos={n_wp}"
        )

    # --- CVXPY problem ---
    p = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(p - s))
    constraints = [p >= 0, p <= 1]

    # EO FPR constraint: |mean(p : Black, y=0) - mean(p : White, y=0)| <= eps
    fpr_black_expr = cp.sum(p[black_neg]) / n_bn
    fpr_white_expr = cp.sum(p[white_neg]) / n_wn
    constraints += [
        fpr_black_expr - fpr_white_expr <= epsilon,
        fpr_white_expr - fpr_black_expr <= epsilon,
    ]

    # EO FNR constraint via TPR parity:
    # FNR = 1 - TPR, so FNR parity <=> TPR parity
    tpr_black_expr = cp.sum(p[black_pos]) / n_bp
    tpr_white_expr = cp.sum(p[white_pos]) / n_wp
    constraints += [
        tpr_black_expr - tpr_white_expr <= epsilon,
        tpr_white_expr - tpr_black_expr <= epsilon,
    ]

    # Lipschitz constraints (vectorized via sparse incidence matrix)
    n_pairs = len(pos_i)
    if n_pairs > 0:
        rows = np.concatenate([np.arange(n_pairs), np.arange(n_pairs)])
        cols = np.concatenate([pos_i, pos_j])
        vals = np.concatenate([np.ones(n_pairs), -np.ones(n_pairs)])
        E = sp.csr_matrix((vals, (rows, cols)), shape=(n_pairs, n))
        rhs = L * distances
        constraints += [E @ p <= rhs, (-E) @ p <= rhs]

    problem = cp.Problem(objective, constraints)

    # --- Solve ---
    try:
        problem.solve(
            solver=cp.OSQP,
            max_iter=10000,
            eps_abs=1e-5,
            eps_rel=1e-5,
            verbose=verbose,
        )
    except cp.SolverError:
        logger.warning("OSQP failed; falling back to SCS.")
        try:
            problem.solve(solver=cp.SCS, verbose=verbose)
        except cp.SolverError:
            return _infeasible_result("solver_error")

    status = problem.status or "unknown"

    if status not in _FEASIBLE_STATUSES:
        return _infeasible_result(status)

    p_opt = np.clip(np.array(p.value, dtype=float), 0.0, 1.0)

    # --- Post-solve verification ---
    fpr_b = float(p_opt[black_neg].mean())
    fpr_w = float(p_opt[white_neg].mean())
    tpr_b = float(p_opt[black_pos].mean())
    tpr_w = float(p_opt[white_pos].mean())
    fpr_gap = abs(fpr_b - fpr_w)
    fnr_gap = abs((1.0 - tpr_b) - (1.0 - tpr_w))

    if fpr_gap > epsilon + 1e-4:
        logger.warning("FPR gap %.4f exceeds epsilon=%.4f", fpr_gap, epsilon)
    if fnr_gap > epsilon + 1e-4:
        logger.warning("FNR gap %.4f exceeds epsilon=%.4f", fnr_gap, epsilon)

    lip_violations = 0
    if n_pairs > 0:
        diff = np.abs(p_opt[pos_i] - p_opt[pos_j])
        lip_violations = int((diff > L * distances + 1e-4).sum())
        if lip_violations > 0:
            logger.warning(
                "%d Lipschitz violations at solution (L=%.2f)", lip_violations, L
            )

    return {
        "p_opt": p_opt,
        "status": status,
        "obj_value": float(problem.value),
        "fpr_gap": fpr_gap,
        "fnr_gap": fnr_gap,
        "lipschitz_violations": lip_violations,
    }


def _infeasible_result(status):
    """Return a standardized dict for infeasible or failed solves."""
    return {
        "p_opt": None,
        "status": status,
        "obj_value": None,
        "fpr_gap": None,
        "fnr_gap": None,
        "lipschitz_violations": None,
    }


def binarize(p_opt, threshold=0.5):
    """
    Convert continuous probabilities to binary decisions.

    Parameters
    ----------
    p_opt : np.ndarray, shape (n,)
        Continuous decision probabilities in [0, 1].
    threshold : float
        Decision threshold (default 0.5).

    Returns
    -------
    np.ndarray, shape (n,)
        Binary decisions (0 or 1).
    """
    return (np.asarray(p_opt, dtype=float) >= threshold).astype(int)


def solve_joint_grid(s, y, race, pairs_df, L_grid, epsilon_grid, ids=None):
    """
    Run solve_joint over the full (L, epsilon) Cartesian grid.

    Iterates epsilon from small to large (outer), and L from large to small
    (inner). Exploits monotonicity: if (L, eps) is infeasible, all smaller L
    values are also infeasible for the same eps, so they are skipped.

    Parameters
    ----------
    s : np.ndarray, shape (n,)
        Baseline calibrated scores.
    y : np.ndarray, shape (n,)
        True binary labels.
    race : np.ndarray, shape (n,)
        Binary race indicator (1=Black, 0=White).
    pairs_df : pd.DataFrame
        KNN pairs with columns [id_i, id_j, distance].
    L_grid : list of float
        Values of Lipschitz constant L to sweep.
    epsilon_grid : list of float
        Values of EO tolerance epsilon to sweep.
    ids : np.ndarray or None
        Original IDs corresponding to positions in s, y, race.

    Returns
    -------
    pd.DataFrame
        Columns: L, epsilon, status, obj_value, fpr_gap, fnr_gap,
                 lipschitz_violations, inconsistency_rate, accuracy, auc.
        Saved to results/extension/grid_results.csv.
    """
    from metrics import inconsistency_rate as _ir

    os.makedirs(EXT_RESULTS_DIR, exist_ok=True)

    L_sorted = sorted(L_grid, reverse=True)   # large → small
    eps_sorted = sorted(epsilon_grid)          # small → large

    rows = []

    for eps in eps_sorted:
        hit_infeasible = False
        for L in L_sorted:
            if hit_infeasible:
                # Monotonicity: smaller L → infeasible for this eps too
                rows.append(_make_row(L, eps, _infeasible_result("infeasible")))
                continue

            logger.info("Grid solve: L=%.2f, epsilon=%.2f", L, eps)
            result = solve_joint(s, y, race, pairs_df, L=L, epsilon=eps, ids=ids)
            row = _make_row(L, eps, result)

            if result["p_opt"] is not None:
                p_opt = result["p_opt"]
                decisions = binarize(p_opt)

                dec_series = pd.Series(
                    decisions, index=(ids if ids is not None else np.arange(len(s)))
                )
                row["inconsistency_rate"] = _ir(pairs_df, dec_series, dec_series)
                row["accuracy"] = float((decisions == np.asarray(y)).mean())
                try:
                    row["auc"] = float(roc_auc_score(np.asarray(y), p_opt))
                except ValueError:
                    row["auc"] = None
            else:
                hit_infeasible = True

            rows.append(row)
            logger.info(
                "  status=%s fpr_gap=%s fnr_gap=%s",
                row["status"], row["fpr_gap"], row["fnr_gap"],
            )

    grid_df = pd.DataFrame(rows)
    out_path = EXT_RESULTS_DIR + "grid_results.csv"
    grid_df.to_csv(out_path, index=False)
    logger.info("Grid results saved to %s", out_path)
    return grid_df


def _make_row(L, eps, result):
    """Build a grid-result row dict from a solve_joint result."""
    return {
        "L": L,
        "epsilon": eps,
        "status": result["status"],
        "obj_value": result["obj_value"],
        "fpr_gap": result["fpr_gap"],
        "fnr_gap": result["fnr_gap"],
        "lipschitz_violations": result["lipschitz_violations"],
        "inconsistency_rate": None,
        "accuracy": None,
        "auc": None,
    }
