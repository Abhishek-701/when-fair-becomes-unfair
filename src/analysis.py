"""
Module 7: Hypothesis tests H1–H3, epsilon-relaxation sweep, and robustness checks.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from config import (
    BOOTSTRAP_N, EPSILON_GRID, K_NEIGHBORS, RANDOM_SEED, RESULTS_DIR,
)
from metrics import (
    inconsistency_with_ci,
    decompose_inconsistency,
    inconsistency_by_threshold_distance,
    inconsistency_rate,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_h1(pair_decomp_df: pd.DataFrame, n_bootstrap: int = BOOTSTRAP_N) -> dict:
    """
    H1: EO post-processing increases individual-level inconsistency.

    Computes delta = inconsistency_rate(EO) - inconsistency_rate(baseline) and
    bootstraps a 95% CI for delta. H1 is supported if the CI excludes zero.

    Parameters
    ----------
    pair_decomp_df : pd.DataFrame
        Output from decompose_inconsistency(), containing per-pair binary indicators.
    n_bootstrap : int
        Number of bootstrap resamples.

    Returns
    -------
    dict with keys: delta, ci, h1_supported.
    """
    indicators_eo = pair_decomp_df['inconsistent_eo'].values.astype(float)
    indicators_base = pair_decomp_df['inconsistent_baseline'].values.astype(float)
    delta_indicators = indicators_eo - indicators_base

    rate_eo = float(indicators_eo.mean())
    rate_base = float(indicators_base.mean())
    delta = rate_eo - rate_base

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(delta_indicators)
    boot_deltas = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_deltas[b] = delta_indicators[idx].mean()

    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    h1_supported = ci_lower > 0  # CI excludes zero on the positive side

    result = {
        'rate_baseline': rate_base,
        'rate_eo': rate_eo,
        'delta': delta,
        'ci': (ci_lower, ci_upper),
        'h1_supported': h1_supported,
    }

    logger.info(
        "H1: delta=%.4f, 95%% CI=[%.4f, %.4f], supported=%s",
        delta, ci_lower, ci_upper, h1_supported,
    )
    return result


def test_h2(pair_decomp_df: pd.DataFrame) -> dict:
    """
    H2: EO-induced inconsistencies concentrate among legally similar individuals
    near decision thresholds.

    Tests:
    1. Quartile pattern: inconsistency rate decreases as threshold distance increases.
    2. Spearman correlation between min(threshold_distance_i, threshold_distance_j)
       and EO-introduced discordance.

    Parameters
    ----------
    pair_decomp_df : pd.DataFrame
        Output from decompose_inconsistency().

    Returns
    -------
    dict with quartile analysis and Spearman correlation results.
    """
    quartile_df = inconsistency_by_threshold_distance(pair_decomp_df)

    # Check monotone decreasing pattern
    eo_rates = quartile_df['inconsistency_rate_eo'].values
    is_monotone_decreasing = all(eo_rates[i] >= eo_rates[i + 1] for i in range(len(eo_rates) - 1))

    # Spearman correlation: min threshold distance vs EO-introduced indicator
    df = pair_decomp_df.copy()
    df['min_thresh_dist'] = df[['threshold_distance_i', 'threshold_distance_j']].min(axis=1)

    # Remove NaN distances (rule pairs have NaN)
    valid = df['min_thresh_dist'].notna()
    if valid.sum() < 10:
        spearman_r, spearman_p = np.nan, np.nan
    else:
        spearman_r, spearman_p = stats.spearmanr(
            df.loc[valid, 'min_thresh_dist'],
            df.loc[valid, 'eo_introduced'],
        )

    # H2 supported: monotone decreasing quartile pattern AND negative correlation
    # (closer to threshold → more EO-introduced discordance = negative correlation with distance)
    h2_supported = is_monotone_decreasing and (not np.isnan(spearman_r)) and spearman_r < 0

    result = {
        'quartile_analysis': quartile_df.to_dict(orient='records'),
        'monotone_decreasing': is_monotone_decreasing,
        'spearman_r': float(spearman_r) if not np.isnan(spearman_r) else None,
        'spearman_p': float(spearman_p) if not np.isnan(spearman_p) else None,
        'h2_supported': h2_supported,
    }

    logger.info(
        "H2: monotone=%s, Spearman r=%.4f (p=%.4f), supported=%s",
        is_monotone_decreasing, spearman_r or 0, spearman_p or 1, h2_supported,
    )
    return result


def test_h3(
    scores: pd.Series,
    y_test: pd.Series,
    race_test: pd.Series,
    knn_pairs: pd.DataFrame,
    epsilon_grid: list = EPSILON_GRID,
    n_bootstrap: int = BOOTSTRAP_N,
) -> pd.DataFrame:
    """
    H3: Group-level error disparity reductions under EO are negatively associated
    with individual-level consistency. Sweeps across epsilon values.

    For each epsilon, fits the relaxed EO postprocessor, computes group FPR disparity,
    and computes individual inconsistency rate (KNN pairs) with bootstrap CI.

    Parameters
    ----------
    scores : pd.Series
        Calibrated probability scores.
    y_test : pd.Series
        True binary labels.
    race_test : pd.Series
        Binary race indicator.
    knn_pairs : pd.DataFrame
        KNN pair dataframe from similarity module.
    epsilon_grid : list
        Epsilon values to sweep.
    n_bootstrap : int
        Number of bootstrap resamples per epsilon.

    Returns
    -------
    pd.DataFrame
        Columns: epsilon, fpr_disparity, inconsistency_rate, ci_lower, ci_upper.
        Saved to results/h3_pareto.csv.
    """
    from eo_postprocessor import fit_eo_relaxed, apply_eo_decisions
    from baseline_model import _group_metrics

    rows = []
    for eps in epsilon_grid:
        logger.info("H3 sweep: epsilon=%.2f", eps)
        pp = fit_eo_relaxed(scores, y_test, race_test, epsilon=eps, random_seed=RANDOM_SEED)
        decisions = apply_eo_decisions(scores, race_test, pp, random_seed=RANDOM_SEED)

        # Group FPR disparity
        y = y_test.values
        d = decisions.values
        r = race_test.values
        m_black = _group_metrics(y[r == 1], d[r == 1])
        m_white = _group_metrics(y[r == 0], d[r == 0])
        fpr_disp = abs(m_black['fpr'] - m_white['fpr'])

        # Individual inconsistency
        ci_result = inconsistency_with_ci(knn_pairs, decisions, n_bootstrap=n_bootstrap)

        rows.append({
            'epsilon': eps,
            'fpr_disparity': fpr_disp,
            'inconsistency_rate': ci_result['rate'],
            'ci_lower': ci_result['ci_lower'],
            'ci_upper': ci_result['ci_upper'],
        })

    pareto_df = pd.DataFrame(rows)
    pareto_df.to_csv(RESULTS_DIR + 'h3_pareto.csv', index=False)
    logger.info("Saved H3 Pareto frontier to %sh3_pareto.csv", RESULTS_DIR)

    return pareto_df


def robustness_checks(data_split: dict, scores: pd.Series, epsilon_grid: list = EPSILON_GRID) -> dict:
    """
    Re-run analysis pipeline with three robustness variants:
    1. Random forest baseline instead of logistic regression.
    2. Cosine similarity pairs instead of Euclidean.
    3. Cross-race pairs only.

    Parameters
    ----------
    data_split : dict
        Output from data_prep.split_data().
    scores : pd.Series
        Calibrated probability scores from the primary (LR) model.
    epsilon_grid : list
        Epsilon values for H3 sweep.

    Returns
    -------
    dict with H1 delta and H3 Pareto curves for each robustness variant.
        Saved to results/robustness/.
    """
    import os
    from eo_postprocessor import (
        fit_eo_postprocessor, apply_eo_decisions, fit_eo_relaxed,
    )
    from similarity import build_knn_pairs
    from baseline_model import _group_metrics

    robustness_dir = RESULTS_DIR + 'robustness/'
    os.makedirs(robustness_dir, exist_ok=True)

    results = {}

    X_train = data_split['X_train']
    X_test = data_split['X_test']
    y_train = data_split['y_train']
    y_test = data_split['y_test']
    race_test = data_split['race_test']
    race_train = data_split['race_train']
    idx_test = data_split['idx_test']

    # --- Variant 1: Random Forest baseline ---
    logger.info("Robustness: training Random Forest baseline...")
    rf_base = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    rf_cal = CalibratedClassifierCV(rf_base, method='isotonic', cv=5)
    rf_cal.fit(X_train, y_train)
    rf_scores = pd.Series(rf_cal.predict_proba(X_test)[:, 1], index=X_test.index)

    pp_rf = fit_eo_postprocessor(rf_scores, y_test, race_test, random_seed=RANDOM_SEED)
    decisions_rf_base = pd.Series(
        (rf_scores.values >= 0.5).astype(int), index=X_test.index,
    )
    decisions_rf_eo = apply_eo_decisions(rf_scores, race_test, pp_rf, random_seed=RANDOM_SEED)

    knn_pairs_rf = build_knn_pairs(X_test, idx_test, race_test, k=K_NEIGHBORS, metric='euclidean')
    decomp_rf = decompose_inconsistency(knn_pairs_rf, decisions_rf_base, decisions_rf_eo, pp_rf, rf_scores)
    h1_rf = test_h1(decomp_rf, n_bootstrap=1000)

    pareto_rf_rows = []
    for eps in epsilon_grid:
        pp_r = fit_eo_relaxed(rf_scores, y_test, race_test, epsilon=eps, random_seed=RANDOM_SEED)
        d_r = apply_eo_decisions(rf_scores, race_test, pp_r, random_seed=RANDOM_SEED)
        ci_r = inconsistency_with_ci(knn_pairs_rf, d_r, n_bootstrap=1000)
        m_b = _group_metrics(y_test.values[race_test.values == 1], d_r.values[race_test.values == 1])
        m_w = _group_metrics(y_test.values[race_test.values == 0], d_r.values[race_test.values == 0])
        pareto_rf_rows.append({
            'epsilon': eps, 'fpr_disparity': abs(m_b['fpr'] - m_w['fpr']),
            'inconsistency_rate': ci_r['rate'],
        })
    pareto_rf = pd.DataFrame(pareto_rf_rows)
    pareto_rf.to_csv(robustness_dir + 'rf_h3_pareto.csv', index=False)
    results['random_forest'] = {'h1_delta': h1_rf['delta'], 'h3_pareto': pareto_rf}
    logger.info("Robustness (RF) H1 delta: %.4f", h1_rf['delta'])

    # --- Variant 2: Cosine similarity ---
    logger.info("Robustness: cosine KNN pairs...")
    cosine_pairs = build_knn_pairs(X_test, idx_test, race_test, k=K_NEIGHBORS, metric='cosine')
    pp_primary = fit_eo_postprocessor(scores, y_test, race_test, random_seed=RANDOM_SEED)
    decisions_base = pd.Series(
        (scores.values >= 0.5).astype(int), index=X_test.index,
    )
    decisions_eo = apply_eo_decisions(scores, race_test, pp_primary, random_seed=RANDOM_SEED)

    decomp_cos = decompose_inconsistency(cosine_pairs, decisions_base, decisions_eo, pp_primary, scores)
    decomp_cos.to_csv(robustness_dir + 'cosine_pair_decomposition.csv', index=False)
    h1_cos = test_h1(decomp_cos, n_bootstrap=1000)

    pareto_cos_rows = []
    for eps in epsilon_grid:
        pp_c = fit_eo_relaxed(scores, y_test, race_test, epsilon=eps, random_seed=RANDOM_SEED)
        d_c = apply_eo_decisions(scores, race_test, pp_c, random_seed=RANDOM_SEED)
        ci_c = inconsistency_with_ci(cosine_pairs, d_c, n_bootstrap=1000)
        m_b = _group_metrics(y_test.values[race_test.values == 1], d_c.values[race_test.values == 1])
        m_w = _group_metrics(y_test.values[race_test.values == 0], d_c.values[race_test.values == 0])
        pareto_cos_rows.append({
            'epsilon': eps, 'fpr_disparity': abs(m_b['fpr'] - m_w['fpr']),
            'inconsistency_rate': ci_c['rate'],
        })
    pareto_cos = pd.DataFrame(pareto_cos_rows)
    pareto_cos.to_csv(robustness_dir + 'cosine_h3_pareto.csv', index=False)
    results['cosine_similarity'] = {'h1_delta': h1_cos['delta'], 'h3_pareto': pareto_cos}
    logger.info("Robustness (cosine) H1 delta: %.4f", h1_cos['delta'])

    # --- Variant 3: Cross-race pairs only ---
    logger.info("Robustness: cross-race pairs only...")
    # Reuse primary Euclidean pairs, filter to cross-race
    knn_pairs_primary = build_knn_pairs(X_test, idx_test, race_test, k=K_NEIGHBORS, metric='euclidean')
    cross_race_pairs = knn_pairs_primary[knn_pairs_primary['pair_type'] == 'cross_race'].copy()
    logger.info("Cross-race pairs: %d", len(cross_race_pairs))

    if len(cross_race_pairs) > 0:
        decomp_cr = decompose_inconsistency(cross_race_pairs, decisions_base, decisions_eo, pp_primary, scores)
        decomp_cr.to_csv(robustness_dir + 'cross_race_pair_decomposition.csv', index=False)
        h1_cr = test_h1(decomp_cr, n_bootstrap=1000)

        pareto_cr_rows = []
        for eps in epsilon_grid:
            pp_cr = fit_eo_relaxed(scores, y_test, race_test, epsilon=eps, random_seed=RANDOM_SEED)
            d_cr = apply_eo_decisions(scores, race_test, pp_cr, random_seed=RANDOM_SEED)
            ci_cr = inconsistency_with_ci(cross_race_pairs, d_cr, n_bootstrap=1000)
            m_b = _group_metrics(y_test.values[race_test.values == 1], d_cr.values[race_test.values == 1])
            m_w = _group_metrics(y_test.values[race_test.values == 0], d_cr.values[race_test.values == 0])
            pareto_cr_rows.append({
                'epsilon': eps, 'fpr_disparity': abs(m_b['fpr'] - m_w['fpr']),
                'inconsistency_rate': ci_cr['rate'],
            })
        pareto_cr = pd.DataFrame(pareto_cr_rows)
        pareto_cr.to_csv(robustness_dir + 'cross_race_h3_pareto.csv', index=False)
        results['cross_race'] = {'h1_delta': h1_cr['delta'], 'h3_pareto': pareto_cr}
        logger.info("Robustness (cross-race) H1 delta: %.4f", h1_cr['delta'])
    else:
        results['cross_race'] = {'h1_delta': None, 'h3_pareto': None}
        logger.warning("No cross-race pairs found; skipping cross-race robustness check.")

    # Summary table
    summary_rows = []
    for variant, res in results.items():
        summary_rows.append({'variant': variant, 'h1_delta': res['h1_delta']})
    pd.DataFrame(summary_rows).to_csv(robustness_dir + 'robustness_summary.csv', index=False)

    return results
