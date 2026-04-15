"""
Module 4: Hardt et al. Equalized Odds post-processor with randomized thresholding.
Implements the core EO logic directly.
"""

import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from config import RANDOM_SEED, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _find_joint_operating_point(fpr0, tpr0, fpr1, tpr1, epsilon=0.0):
    """
    Find the joint (FPR_target, TPR_target) that minimizes total error subject
    to |FPR_g - FPR_g'| <= epsilon and |TPR_g - TPR_g'| <= epsilon.

    For epsilon=0 (strict EO): both groups must operate at the same (FPR, TPR).
    The achievable common TPR at a given FPR is min(TPR0(FPR), TPR1(FPR)).
    We pick FPR that minimizes (1-TPR) + FPR (total error at that operating point).

    For epsilon>0: groups may differ by up to epsilon; we do a 2D grid search.

    Parameters
    ----------
    fpr0, tpr0 : np.ndarray
        ROC curve for group 0 (White), sorted by increasing FPR.
    fpr1, tpr1 : np.ndarray
        ROC curve for group 1 (Black), sorted by increasing FPR.
    epsilon : float
        Maximum allowed per-axis disparity.

    Returns
    -------
    tuple (target_fpr, target_tpr)
    """
    fpr_grid = np.linspace(0.0, 1.0, 500)

    # Interpolate each group's max achievable TPR at each FPR value
    tpr0_at_fpr = np.interp(fpr_grid, fpr0, tpr0)
    tpr1_at_fpr = np.interp(fpr_grid, fpr1, tpr1)

    if epsilon == 0.0:
        # Both groups must share the same (FPR, TPR).
        # At FPR=f, the achievable common TPR = min(TPR0(f), TPR1(f)).
        common_tpr = np.minimum(tpr0_at_fpr, tpr1_at_fpr)
        error = (1.0 - common_tpr) + fpr_grid   # FNR + FPR (total error proxy)
        best_idx = int(np.argmin(error))
        return float(fpr_grid[best_idx]), float(common_tpr[best_idx])
    else:
        # Relaxed: groups may operate at different FPR/TPR within epsilon of each other.
        n = 100
        fg = np.linspace(0.0, 1.0, n)
        t0 = np.interp(fg, fpr0, tpr0)
        t1 = np.interp(fg, fpr1, tpr1)

        best_error = np.inf
        best_fpr = 0.5
        best_tpr = 0.5

        for i in range(n):
            for j in range(n):
                if abs(fg[i] - fg[j]) <= epsilon and abs(t0[i] - t1[j]) <= epsilon:
                    err = (1.0 - t0[i] + fg[i] + 1.0 - t1[j] + fg[j]) / 4.0
                    if err < best_error:
                        best_error = err
                        best_fpr = (fg[i] + fg[j]) / 2.0
                        best_tpr = (t0[i] + t1[j]) / 2.0

        return float(best_fpr), float(best_tpr)


def _find_operating_points(fpr, tpr, thresholds, target_fpr, target_tpr):
    """
    Identify the two adjacent ROC curve points that bracket target_fpr and
    compute the mixing probability by linear interpolation.

    sklearn's roc_curve returns len(fpr) = len(thresholds) + 1: fpr[0]=0 is
    the no-positive-predictions point, and fpr[i] uses thresholds[i-1] for i>=1.

    Parameters
    ----------
    fpr, tpr : np.ndarray
        ROC curve arrays (sorted by increasing FPR).
    thresholds : np.ndarray
        Decision thresholds (length = len(fpr) - 1).
    target_fpr : float
        Target false positive rate.
    target_tpr : float
        Target true positive rate (used for logging only).

    Returns
    -------
    dict with threshold_low, threshold_high, mix_rate.
        threshold_low  : lower threshold → more positives → higher FPR/TPR
        threshold_high : higher threshold → fewer positives → lower FPR/TPR
        mix_rate       : P(use threshold_low) to achieve target_fpr
    """
    n = len(fpr)

    for i in range(n - 1):
        f0, f1 = fpr[i], fpr[i + 1]
        if f0 <= target_fpr <= f1:
            denom = f1 - f0
            mix = float((target_fpr - f0) / denom) if denom > 1e-10 else 0.0
            mix = float(np.clip(mix, 0.0, 1.0))

            # Threshold at fpr[i] is thresholds[i-1] for i>=1, else thresholds[0]+1
            if i == 0:
                thresh_at_i = float(thresholds[0]) + 1.0
            else:
                thresh_at_i = float(thresholds[i - 1])

            # Threshold at fpr[i+1] is thresholds[i]
            thresh_at_i1 = float(thresholds[i])

            # thresh_at_i > thresh_at_i1 (higher threshold = fewer positives = lower FPR)
            return {
                'threshold_low': thresh_at_i1,   # lower thresh → point (fpr[i+1], tpr[i+1])
                'threshold_high': thresh_at_i,   # higher thresh → point (fpr[i],   tpr[i])
                'mix_rate': mix,                  # P(use threshold_low)
            }

    # Fallback: use the point closest to target_fpr
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    th = float(thresholds[max(idx - 1, 0)]) if idx > 0 else float(thresholds[0])
    return {'threshold_low': th, 'threshold_high': th, 'mix_rate': 0.0}


def fit_eo_postprocessor(scores, y, race, random_seed=RANDOM_SEED):
    """
    Fit the Hardt et al. randomized EO post-processor.

    For each group computes an ROC curve, identifies the optimal joint operating
    point (FPR_target, TPR_target) satisfying equalized odds, then finds the two
    bracketing ROC points and mixing probability for each group.

    Parameters
    ----------
    scores : pd.Series
        Calibrated probability scores.
    y : pd.Series
        True binary labels.
    race : pd.Series
        Binary race indicator (1=Black, 0=White).
    random_seed : int
        Random seed (stored for reproducibility metadata).

    Returns
    -------
    dict
        Postprocessor parameters per group. Saved to results/eo_postprocessor.json.
    """
    s = scores.values
    y_arr = y.values
    r = race.values

    mask0 = r == 0  # White
    mask1 = r == 1  # Black

    fpr0, tpr0, thresh0 = roc_curve(y_arr[mask0], s[mask0])
    fpr1, tpr1, thresh1 = roc_curve(y_arr[mask1], s[mask1])

    target_fpr, target_tpr = _find_joint_operating_point(fpr0, tpr0, fpr1, tpr1, epsilon=0.0)
    logger.info("EO target — FPR: %.4f, TPR: %.4f", target_fpr, target_tpr)

    ops0 = _find_operating_points(fpr0, tpr0, thresh0, target_fpr, target_tpr)
    ops1 = _find_operating_points(fpr1, tpr1, thresh1, target_fpr, target_tpr)

    postprocessor = {
        'target_fpr': float(target_fpr),
        'target_tpr': float(target_tpr),
        'group_0': {**ops0, 'target_fpr': float(target_fpr), 'target_tpr': float(target_tpr)},
        'group_1': {**ops1, 'target_fpr': float(target_fpr), 'target_tpr': float(target_tpr)},
    }

    with open(RESULTS_DIR + 'eo_postprocessor.json', 'w') as f:
        json.dump(postprocessor, f, indent=2)
    logger.info("Saved EO postprocessor to %seo_postprocessor.json", RESULTS_DIR)

    return postprocessor


def apply_eo_decisions(scores, race, postprocessor, random_seed=RANDOM_SEED):
    """
    Apply the fitted EO postprocessor to produce binary decisions.

    For each individual in group g:
      - Draw u ~ Uniform(0,1)
      - If u < mix_rate: apply threshold_low (more permissive → higher FPR/TPR)
      - Else: apply threshold_high (more restrictive → lower FPR/TPR)

    Parameters
    ----------
    scores : pd.Series
        Calibrated probability scores.
    race : pd.Series
        Binary race indicator (1=Black, 0=White).
    postprocessor : dict
        Output of fit_eo_postprocessor().
    random_seed : int
        Random seed for Bernoulli draws.

    Returns
    -------
    pd.Series
        Binary EO decisions indexed like scores.
    """
    rng = np.random.default_rng(random_seed)
    s = scores.values
    r = race.values
    decisions = np.zeros(len(s), dtype=int)

    for group_key, mask in [('group_0', r == 0), ('group_1', r == 1)]:
        params = postprocessor[group_key]
        thresh_low  = params['threshold_low']
        thresh_high = params['threshold_high']
        mix_rate    = params['mix_rate']

        n_group = int(mask.sum())
        u = rng.uniform(size=n_group)
        group_scores = s[mask]

        use_low = u < mix_rate
        group_decisions = np.where(
            use_low,
            (group_scores >= thresh_low).astype(int),
            (group_scores >= thresh_high).astype(int),
        )
        decisions[mask] = group_decisions

    return pd.Series(decisions, index=scores.index, name='decision_EO')


def compute_eo_metrics(y_test, decision_eo, race_test):
    """
    Compute per-group and overall metrics for EO post-processed decisions.

    Parameters
    ----------
    y_test : pd.Series
        True binary labels.
    decision_eo : pd.Series
        Binary EO decisions.
    race_test : pd.Series
        Binary race indicator (1=Black, 0=White).

    Returns
    -------
    dict
        Overall and per-group metrics. Saved to results/eo_metrics.json.
    """
    from baseline_model import _group_metrics

    y = y_test.values
    d = decision_eo.values
    r = race_test.values

    overall_acc = float((d == y).mean())
    metrics_black = _group_metrics(y[r == 1], d[r == 1])
    metrics_white = _group_metrics(y[r == 0], d[r == 0])

    fpr_disparity = metrics_black['fpr'] - metrics_white['fpr']
    fnr_disparity = metrics_black['fnr'] - metrics_white['fnr']

    result = {
        'overall': {'accuracy': overall_acc},
        'black': metrics_black,
        'white': metrics_white,
        'fpr_disparity': fpr_disparity,
        'fnr_disparity': fnr_disparity,
    }

    print("\n=== EO Post-Processed Metrics ===")
    print(f"  Overall accuracy: {overall_acc:.1%}")
    print(f"  {'Group':<10} {'FPR':>7} {'FNR':>7} {'TPR':>7} {'Acc':>7}")
    print(f"  {'Black':<10} {metrics_black['fpr']:>7.1%} {metrics_black['fnr']:>7.1%} "
          f"{metrics_black['tpr']:>7.1%} {metrics_black['accuracy']:>7.1%}")
    print(f"  {'White':<10} {metrics_white['fpr']:>7.1%} {metrics_white['fnr']:>7.1%} "
          f"{metrics_white['tpr']:>7.1%} {metrics_white['accuracy']:>7.1%}")
    print(f"  FPR disparity (Black - White): {fpr_disparity:+.3f}")
    print(f"  FNR disparity (Black - White): {fnr_disparity:+.3f}")

    with open(RESULTS_DIR + 'eo_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    logger.info("Saved EO metrics to %seo_metrics.json", RESULTS_DIR)

    return result


def fit_eo_relaxed(scores, y, race, epsilon, random_seed=RANDOM_SEED):
    """
    Fit a relaxed EO postprocessor allowing group FPR/TPR to differ by up to epsilon.

    Parameters
    ----------
    scores : pd.Series
        Calibrated probability scores.
    y : pd.Series
        True binary labels.
    race : pd.Series
        Binary race indicator.
    epsilon : float
        Maximum allowed FPR/TPR disparity between groups.
    random_seed : int
        Random seed (metadata only).

    Returns
    -------
    dict
        Relaxed postprocessor parameters.
    """
    s = scores.values
    y_arr = y.values
    r = race.values

    fpr0, tpr0, thresh0 = roc_curve(y_arr[r == 0], s[r == 0])
    fpr1, tpr1, thresh1 = roc_curve(y_arr[r == 1], s[r == 1])

    target_fpr, target_tpr = _find_joint_operating_point(fpr0, tpr0, fpr1, tpr1, epsilon=epsilon)
    logger.info("Relaxed EO (eps=%.2f) target — FPR: %.4f, TPR: %.4f",
                epsilon, target_fpr, target_tpr)

    ops0 = _find_operating_points(fpr0, tpr0, thresh0, target_fpr, target_tpr)
    ops1 = _find_operating_points(fpr1, tpr1, thresh1, target_fpr, target_tpr)

    return {
        'epsilon': epsilon,
        'target_fpr': float(target_fpr),
        'target_tpr': float(target_tpr),
        'group_0': {**ops0, 'target_fpr': float(target_fpr), 'target_tpr': float(target_tpr)},
        'group_1': {**ops1, 'target_fpr': float(target_fpr), 'target_tpr': float(target_tpr)},
    }


def run_multi_seed_eo(scores, race, postprocessor, n_seeds=10):
    """
    Run EO decision-making across multiple random seeds to quantify seed variance.

    Parameters
    ----------
    scores : pd.Series
        Calibrated probability scores.
    race : pd.Series
        Binary race indicator.
    postprocessor : dict
        Fitted postprocessor from fit_eo_postprocessor().
    n_seeds : int
        Number of seeds to run (default 10).

    Returns
    -------
    pd.DataFrame
        Rows = individuals, columns = seed_0..seed_{n_seeds-1}.
        Saves results/eo_decisions_all_seeds.csv and results/eo_decisions_mean.csv.
    """
    all_decisions = {}
    for seed in range(n_seeds):
        decisions = apply_eo_decisions(scores, race, postprocessor, random_seed=seed)
        all_decisions[f'seed_{seed}'] = decisions.values

    decisions_df = pd.DataFrame(all_decisions, index=scores.index)
    decisions_df.to_csv(RESULTS_DIR + 'eo_decisions_all_seeds.csv')

    mean_decisions = decisions_df.mean(axis=1)
    mean_decisions.name = 'decision_EO_mean'
    mean_decisions.to_csv(RESULTS_DIR + 'eo_decisions_mean.csv', header=True)

    logger.info("Saved multi-seed EO decisions (%d seeds)", n_seeds)
    return decisions_df
