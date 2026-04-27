"""
Extension Module 3: Head-to-head comparison of three post-processors.

  1. Hardt EO         — pre-computed decisions from results/eo_decisions_mean.csv
  2. Petersen IF-only — graph Laplacian smoothing, no EO constraint
  3. Joint EO+Lip     — solve_joint at (L_star, epsilon_star)

All three methods are evaluated on the same metrics for a fair comparison.
"""

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.metrics import roc_auc_score

_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config import RANDOM_SEED, RESULTS_DIR  # noqa: E402
from extension.joint_postprocessor import binarize, solve_joint  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EXT_RESULTS_DIR = os.path.join(RESULTS_DIR, "extension") + os.sep
EXT_FIGURES_DIR = os.path.join(RESULTS_DIR, "extension", "figures") + os.sep


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _group_metrics(y, decisions, race, scores=None):
    """
    Compute FPR, FNR, accuracy per racial group and overall AUC.

    Parameters
    ----------
    y : np.ndarray
        True binary labels.
    decisions : np.ndarray
        Binary decisions (0 or 1).
    race : np.ndarray
        Binary race indicator (1=Black, 0=White).
    scores : np.ndarray or None
        Continuous scores for AUC computation.

    Returns
    -------
    dict
        Keys: fpr_black, fpr_white, fpr_gap, fnr_black, fnr_white, fnr_gap,
              accuracy, auc.
    """
    def _rates(y_g, d_g):
        tp = int(((d_g == 1) & (y_g == 1)).sum())
        fp = int(((d_g == 1) & (y_g == 0)).sum())
        tn = int(((d_g == 0) & (y_g == 0)).sum())
        fn = int(((d_g == 0) & (y_g == 1)).sum())
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        return fpr, fnr

    mask_b, mask_w = race == 1, race == 0
    fpr_b, fnr_b = _rates(y[mask_b], decisions[mask_b])
    fpr_w, fnr_w = _rates(y[mask_w], decisions[mask_w])

    auc = None
    if scores is not None:
        try:
            auc = float(roc_auc_score(y, scores))
        except ValueError:
            pass

    return {
        "fpr_black": fpr_b,
        "fpr_white": fpr_w,
        "fpr_gap": abs(fpr_b - fpr_w),
        "fnr_black": fnr_b,
        "fnr_white": fnr_w,
        "fnr_gap": abs(fnr_b - fnr_w),
        "accuracy": float((decisions == y).mean()),
        "auc": auc,
    }


def _laplacian_smooth(s, pairs_df, tau, ids=None):
    """
    Graph Laplacian smoothing: minimize ||p - s||^2 + tau * p^T L_graph p.

    Closed-form solution: p = (I + tau * L_graph)^{-1} s.
    Uses the unweighted KNN graph Laplacian.

    Parameters
    ----------
    s : np.ndarray, shape (n,)
        Baseline scores.
    pairs_df : pd.DataFrame
        KNN pairs with id_i, id_j columns.
    tau : float
        Regularisation strength (larger = more smoothing).
    ids : np.ndarray or None
        Original IDs for building id→position mapping.

    Returns
    -------
    np.ndarray, shape (n,)
        Smoothed probabilities, clipped to [0, 1].
    """
    n = len(s)

    if ids is not None:
        id_to_pos = {int(orig): pos for pos, orig in enumerate(ids)}
    else:
        id_to_pos = {i: i for i in range(n)}

    valid = pairs_df["id_i"].map(id_to_pos).notna() & pairs_df["id_j"].map(id_to_pos).notna()
    pv = pairs_df[valid]
    pi = pv["id_i"].map(id_to_pos).astype(int).values
    pj = pv["id_j"].map(id_to_pos).astype(int).values

    # Symmetric adjacency (unweighted)
    data = np.ones(len(pi) * 2)
    rows = np.concatenate([pi, pj])
    cols = np.concatenate([pj, pi])
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    degrees = np.array(A.sum(axis=1)).ravel()
    D = sp.diags(degrees)
    L_graph = D - A

    M = sp.eye(n, format="csr") + tau * L_graph
    p_smooth = spla.spsolve(M, s)
    return np.clip(p_smooth, 0.0, 1.0)


def _inconsistency(decisions_arr, pairs_df, ids):
    """Compute inconsistency rate for a decisions array."""
    from metrics import inconsistency_rate as _ir

    dec = pd.Series(
        decisions_arr,
        index=(ids if ids is not None else np.arange(len(decisions_arr))),
    )
    return _ir(pairs_df, dec, dec)


def _cross_race_ir(decisions_arr, pairs_df, ids):
    """Compute cross-race inconsistency rate."""
    from metrics import inconsistency_rate as _ir

    cross = pairs_df[pairs_df["pair_type"] == "cross_race"]
    if cross.empty:
        return float("nan")
    dec = pd.Series(
        decisions_arr,
        index=(ids if ids is not None else np.arange(len(decisions_arr))),
    )
    return _ir(cross, dec, dec)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_all_methods(s, y, race, pairs_df, L_star, epsilon_star, ids=None):
    """
    Compare three post-processors head-to-head at operating point (L_star, epsilon_star).

    Methods:
      1. Hardt EO        — loaded from results/eo_decisions_mean.csv
      2. Petersen IF-only — graph Laplacian smoothing, tau swept to match joint IR
      3. Joint EO+Lip    — solve_joint(L_star, epsilon_star)

    Parameters
    ----------
    s : np.ndarray, shape (n,)
        Baseline calibrated scores.
    y : np.ndarray, shape (n,)
        True binary labels.
    race : np.ndarray, shape (n,)
        Binary race indicator (1=Black, 0=White).
    pairs_df : pd.DataFrame
        KNN pairs with columns [id_i, id_j, distance, pair_type].
    L_star : float
        Chosen Lipschitz constant for the joint method.
    epsilon_star : float
        Chosen EO tolerance for the joint method.
    ids : np.ndarray or None
        Original IDs corresponding to positions in s, y, race.

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per method.
        Saved to results/extension/method_comparison.csv.
    """
    os.makedirs(EXT_RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Method 1: Hardt EO
    # ------------------------------------------------------------------
    eo_path = os.path.join(RESULTS_DIR, "eo_decisions_mean.csv")
    eo_df = pd.read_csv(eo_path, index_col=0)
    eo_col = eo_df.columns[0]  # actual column name is 'decision_EO_mean'

    if ids is not None:
        eo_decisions = eo_df.loc[ids, eo_col].values.round().astype(int)
    else:
        eo_decisions = eo_df.iloc[: len(s)][eo_col].values.round().astype(int)

    eo_m = _group_metrics(y, eo_decisions, race, scores=s)
    eo_ir = _inconsistency(eo_decisions, pairs_df, ids)
    eo_cr = _cross_race_ir(eo_decisions, pairs_df, ids)

    # ------------------------------------------------------------------
    # Method 3: Joint (solved first to get target IR for tau sweep)
    # ------------------------------------------------------------------
    logger.info("Solving joint method: L=%.2f, epsilon=%.2f", L_star, epsilon_star)
    joint_res = solve_joint(
        s, y, race, pairs_df, L=L_star, epsilon=epsilon_star, ids=ids
    )

    if joint_res["p_opt"] is None:
        logger.warning(
            "Joint infeasible at (L=%.2f, eps=%.2f); falling back to threshold on s.",
            L_star, epsilon_star,
        )
        joint_p = s
    else:
        joint_p = joint_res["p_opt"]

    joint_decisions = binarize(joint_p)
    joint_m = _group_metrics(y, joint_decisions, race, scores=joint_p)
    joint_ir = _inconsistency(joint_decisions, pairs_df, ids)
    joint_cr = _cross_race_ir(joint_decisions, pairs_df, ids)
    target_ir = joint_ir

    # ------------------------------------------------------------------
    # Method 2: Petersen IF-only (sweep tau to match joint IR)
    # ------------------------------------------------------------------
    tau_grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    best_tau, best_diff = tau_grid[0], float("inf")

    for tau in tau_grid:
        p_smooth = _laplacian_smooth(s, pairs_df, tau, ids=ids)
        d_smooth = binarize(p_smooth)
        ir_smooth = _inconsistency(d_smooth, pairs_df, ids)
        diff = abs(ir_smooth - target_ir)
        if diff < best_diff:
            best_diff, best_tau = diff, tau

    logger.info(
        "Petersen IF: best tau=%.4f (IR diff=%.4f from target=%.4f)",
        best_tau, best_diff, target_ir,
    )

    p_pet = _laplacian_smooth(s, pairs_df, best_tau, ids=ids)
    pet_decisions = binarize(p_pet)
    pet_m = _group_metrics(y, pet_decisions, race, scores=p_pet)
    pet_ir = _inconsistency(pet_decisions, pairs_df, ids)
    pet_cr = _cross_race_ir(pet_decisions, pairs_df, ids)

    # ------------------------------------------------------------------
    # Assemble DataFrame
    # ------------------------------------------------------------------
    def _row(name, m, ir, cr):
        return {
            "method": name,
            "accuracy": round(m["accuracy"], 4),
            "auc": round(m["auc"], 4) if m["auc"] is not None else None,
            "fpr_black": round(m["fpr_black"], 4),
            "fpr_white": round(m["fpr_white"], 4),
            "fpr_gap": round(m["fpr_gap"], 4),
            "fnr_black": round(m["fnr_black"], 4),
            "fnr_white": round(m["fnr_white"], 4),
            "fnr_gap": round(m["fnr_gap"], 4),
            "inconsistency_rate": round(ir, 4),
            "inconsistency_rate_cross_race": (
                round(cr, 4) if not np.isnan(cr) else None
            ),
        }

    comparison_df = pd.DataFrame(
        [
            _row("Hardt EO", eo_m, eo_ir, eo_cr),
            _row("Petersen IF-only", pet_m, pet_ir, pet_cr),
            _row("Joint EO+Lip", joint_m, joint_ir, joint_cr),
        ]
    )

    out_path = EXT_RESULTS_DIR + "method_comparison.csv"
    comparison_df.to_csv(out_path, index=False)
    logger.info("Method comparison saved to %s", out_path)
    return comparison_df


def plot_comparison_radar(comparison_df, save_path):
    """
    Radar/spider chart comparing all three post-processors on six axes.

    Axes (all normalized so higher = better):
      accuracy, AUC, FPR gap (inverted), FNR gap (inverted),
      individual consistency (1 - IR), cross-race consistency (1 - IR_cr).

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of compare_all_methods.
    save_path : str
        Destination file path for the PNG figure (300 DPI).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    methods = comparison_df["method"].tolist()
    labels = [
        "Accuracy",
        "AUC",
        "FPR Gap\n(inverted)",
        "FNR Gap\n(inverted)",
        "IF Consistency",
        "Cross-Race\nConsistency",
    ]

    def _safe(val, default=0.0):
        if val is None:
            return default
        v = float(val)
        return default if np.isnan(v) else v

    data = []
    for _, row in comparison_df.iterrows():
        data.append(
            [
                _safe(row["accuracy"]),
                _safe(row["auc"]),
                max(0.0, 1.0 - _safe(row["fpr_gap"])),
                max(0.0, 1.0 - _safe(row["fnr_gap"])),
                max(0.0, 1.0 - _safe(row["inconsistency_rate"])),
                max(0.0, 1.0 - _safe(row["inconsistency_rate_cross_race"])),
            ]
        )

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#0077BB", "#EE7733", "#009988"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for method, vals, color in zip(methods, data, colors):
        v = vals + vals[:1]
        ax.plot(angles, v, "o-", linewidth=2, color=color, label=method)
        ax.fill(angles, v, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Method Comparison: Radar Chart\n(higher = better on all axes)",
        size=12,
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Radar comparison chart saved to %s", save_path)
