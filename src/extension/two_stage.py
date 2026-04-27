"""
Extension Module 4: Two-stage pipeline — Lipschitz QP followed by Hardt EO.

Stage 1: solve_joint() produces p_opt, a Lipschitz-smoothed probability vector.
         Similar individuals (KNN-connected) are forced to have similar scores.
Stage 2: fit_eo_postprocessor() + apply_eo_decisions() applies Hardt EO to
         p_opt instead of raw baseline scores s.

Why this should work:
  - Hardt EO guarantees binary group fairness regardless of the input scores.
  - The Lipschitz constraint narrows the score distribution around each group's
    EO threshold — fewer individuals land in the randomized mixing zone.
  - Less mixing → less decision randomness → lower inconsistency than Hardt-on-s.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config import RANDOM_SEED, RESULTS_DIR
from eo_postprocessor import fit_eo_postprocessor, apply_eo_decisions
from extension.joint_postprocessor import solve_joint

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EXT_RESULTS_DIR = os.path.join(RESULTS_DIR, "extension") + os.sep
EXT_FIGURES_DIR = os.path.join(RESULTS_DIR, "extension", "figures") + os.sep


def run_two_stage(s, y, race, pairs_df, L, epsilon_qp=0.05, ids=None,
                  n_seeds=10, random_seed=RANDOM_SEED):
    """
    Run the two-stage Lipschitz QP → Hardt EO pipeline.

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
    L : float
        Lipschitz constant for Stage 1 QP.
    epsilon_qp : float
        EO tolerance for Stage 1 QP (passed to solve_joint; does not
        directly control binary EO — Stage 2 Hardt ensures that).
    ids : np.ndarray or None
        Original IDs corresponding to positions in s, y, race.
    n_seeds : int
        Number of random seeds for Stage 2 Hardt decisions.
    random_seed : int
        Base random seed.

    Returns
    -------
    dict with keys:
        'p_opt'        : np.ndarray — Lipschitz-smoothed scores from Stage 1
        'qp_status'    : str — solver status from Stage 1
        'postprocessor': dict — fitted Hardt EO parameters for Stage 2
        'decisions'    : pd.Series — binary decisions (mean of n_seeds runs)
        'decisions_all': pd.DataFrame — per-seed binary decisions
        'metrics'      : dict — accuracy, FPR/FNR by group, inconsistency rates
    """
    from metrics import inconsistency_rate as _ir

    index = pd.Index(ids) if ids is not None else pd.RangeIndex(len(s))

    # ── Stage 1: Lipschitz QP ────────────────────────────────────────────────
    logger.info("Stage 1: QP solve at L=%.2f, epsilon_qp=%.2f", L, epsilon_qp)
    qp_result = solve_joint(
        s, y, race, pairs_df, L=L, epsilon=epsilon_qp, ids=ids
    )

    if qp_result["p_opt"] is None:
        logger.warning("Stage 1 QP infeasible — falling back to raw scores s.")
        p_opt = s.copy()
        qp_status = qp_result["status"]
    else:
        p_opt = qp_result["p_opt"]
        qp_status = qp_result["status"]
        logger.info(
            "Stage 1 done: obj=%.4f  fpr_gap=%.4f  fnr_gap=%.4f  lip_viol=%d",
            qp_result["obj_value"], qp_result["fpr_gap"],
            qp_result["fnr_gap"], qp_result["lipschitz_violations"],
        )

    # Convert to pd.Series for eo_postprocessor interface
    p_series  = pd.Series(p_opt, index=index)
    y_series  = pd.Series(y, index=index)
    r_series  = pd.Series(race, index=index)

    # ── Stage 2: Fit Hardt EO on p_opt ──────────────────────────────────────
    logger.info("Stage 2: fitting Hardt EO on Lipschitz-smoothed scores.")
    postprocessor = fit_eo_postprocessor(
        p_series, y_series, r_series, random_seed=random_seed
    )
    logger.info(
        "EO target — FPR: %.4f, TPR: %.4f",
        postprocessor["target_fpr"], postprocessor["target_tpr"],
    )
    logger.info(
        "Mix rate — Black: %.3f, White: %.3f",
        postprocessor["group_1"]["mix_rate"],
        postprocessor["group_0"]["mix_rate"],
    )

    # Apply across n_seeds and average (same protocol as Phase 1)
    seed_decisions = {}
    for seed in range(n_seeds):
        d = apply_eo_decisions(p_series, r_series, postprocessor,
                               random_seed=random_seed + seed)
        seed_decisions[f"seed_{seed}"] = d.values

    decisions_all = pd.DataFrame(seed_decisions, index=index)
    decisions_mean_cont = decisions_all.mean(axis=1)          # continuous mean
    decisions_mean_bin  = (decisions_mean_cont >= 0.5).astype(int)  # binary

    # ── Metrics ──────────────────────────────────────────────────────────────
    d_arr = decisions_mean_bin.values
    y_arr = np.asarray(y)
    r_arr = np.asarray(race)

    mask_b, mask_w = r_arr == 1, r_arr == 0

    def _rates(y_g, d_g):
        tp = ((d_g == 1) & (y_g == 1)).sum()
        fp = ((d_g == 1) & (y_g == 0)).sum()
        tn = ((d_g == 0) & (y_g == 0)).sum()
        fn = ((d_g == 0) & (y_g == 1)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        return float(fpr), float(fnr)

    fpr_b, fnr_b = _rates(y_arr[mask_b], d_arr[mask_b])
    fpr_w, fnr_w = _rates(y_arr[mask_w], d_arr[mask_w])

    # Inconsistency rate (mean-binarized decisions)
    dec_series = decisions_mean_bin
    overall_ir = _ir(pairs_df, dec_series, dec_series)

    cross = pairs_df[pairs_df["pair_type"] == "cross_race"]
    cross_ir = _ir(cross, dec_series, dec_series) if not cross.empty else float("nan")

    # Per-seed inconsistency to show variance
    seed_irs = []
    for col in decisions_all.columns:
        sd = pd.Series(decisions_all[col].values, index=index)
        seed_irs.append(_ir(pairs_df, sd, sd))

    try:
        auc = float(roc_auc_score(y_arr, p_opt))
    except ValueError:
        auc = None

    metrics = {
        "accuracy": float((d_arr == y_arr).mean()),
        "auc": auc,
        "fpr_black": fpr_b,
        "fpr_white": fpr_w,
        "fpr_gap": abs(fpr_b - fpr_w),
        "fnr_black": fnr_b,
        "fnr_white": fnr_w,
        "fnr_gap": abs(fnr_b - fnr_w),
        "inconsistency_rate": overall_ir,
        "inconsistency_rate_cross_race": cross_ir,
        "seed_ir_mean": float(np.mean(seed_irs)),
        "seed_ir_std": float(np.std(seed_irs)),
        "qp_obj": qp_result["obj_value"],
        "qp_fpr_gap_soft": qp_result["fpr_gap"],
        "mix_rate_black": postprocessor["group_1"]["mix_rate"],
        "mix_rate_white": postprocessor["group_0"]["mix_rate"],
    }

    return {
        "p_opt": p_opt,
        "qp_status": qp_status,
        "postprocessor": postprocessor,
        "decisions": decisions_mean_bin,
        "decisions_all": decisions_all,
        "metrics": metrics,
    }


def sweep_two_stage(s, y, race, pairs_df, L_values, epsilon_qp=0.05,
                    ids=None, n_seeds=10):
    """
    Run the two-stage pipeline across a range of L values.

    Parameters
    ----------
    s, y, race : np.ndarray
        Baseline scores, labels, race indicators.
    pairs_df : pd.DataFrame
        KNN pairs.
    L_values : list of float
        Lipschitz constants to sweep.
    epsilon_qp : float
        EO tolerance for the QP stage.
    ids : np.ndarray or None
        Original IDs.
    n_seeds : int
        Seeds for Hardt stage.

    Returns
    -------
    pd.DataFrame
        One row per L value with all metrics.
        Saved to results/extension/two_stage_sweep.csv.
    """
    os.makedirs(EXT_RESULTS_DIR, exist_ok=True)
    rows = []
    for L in L_values:
        logger.info("Two-stage sweep: L=%.2f", L)
        result = run_two_stage(s, y, race, pairs_df, L=L,
                               epsilon_qp=epsilon_qp, ids=ids, n_seeds=n_seeds)
        row = {"L": L, "epsilon_qp": epsilon_qp, **result["metrics"]}
        rows.append(row)
        m = result["metrics"]
        logger.info(
            "  acc=%.4f  fpr_gap=%.4f  fnr_gap=%.4f  IR=%.4f  CR-IR=%.4f  "
            "mix_B=%.3f  mix_W=%.3f",
            m["accuracy"], m["fpr_gap"], m["fnr_gap"],
            m["inconsistency_rate"], m["inconsistency_rate_cross_race"],
            m["mix_rate_black"], m["mix_rate_white"],
        )

    df = pd.DataFrame(rows)
    out = EXT_RESULTS_DIR + "two_stage_sweep.csv"
    df.to_csv(out, index=False)
    logger.info("Two-stage sweep saved to %s", out)
    return df
