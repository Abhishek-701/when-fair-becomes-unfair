"""
Module 6: Inconsistency rate computation, bootstrap confidence intervals,
discordance decomposition, and threshold-distance analysis.
"""

import logging

import numpy as np
import pandas as pd

from config import BOOTSTRAP_N, RANDOM_SEED, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def inconsistency_rate(
    pairs_df: pd.DataFrame,
    decisions_col_i: pd.Series,
    decisions_col_j: pd.Series,
) -> float:
    """
    Compute the proportion of similar pairs with discordant decisions.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Pair dataframe with columns id_i, id_j.
    decisions_col_i : pd.Series
        Decision series indexed by individual ID for the 'i' side.
    decisions_col_j : pd.Series
        Decision series indexed by individual ID for the 'j' side.

    Returns
    -------
    float
        Fraction of pairs where decisions differ.
    """
    d_i = decisions_col_i.loc[pairs_df['id_i']].values
    d_j = decisions_col_j.loc[pairs_df['id_j']].values
    return float((d_i != d_j).mean())


def inconsistency_with_ci(
    pairs_df: pd.DataFrame,
    decision_series: pd.Series,
    n_bootstrap: int = BOOTSTRAP_N,
) -> dict:
    """
    Compute inconsistency rate with a 95% percentile bootstrap confidence interval.

    Bootstraps by resampling the pair set with replacement. The statistic is
    the mean of the binary inconsistency indicators (0/1 per pair).

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Pair dataframe with columns id_i, id_j.
    decision_series : pd.Series
        Decision series indexed by individual ID.
    n_bootstrap : int
        Number of bootstrap resamples.

    Returns
    -------
    dict with keys: rate, ci_lower, ci_upper, n_pairs.
    """
    d_i = decision_series.loc[pairs_df['id_i']].values
    d_j = decision_series.loc[pairs_df['id_j']].values
    indicators = (d_i != d_j).astype(float)

    rate = float(indicators.mean())
    n_pairs = len(indicators)

    rng = np.random.default_rng(RANDOM_SEED)
    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(indicators, size=n_pairs, replace=True)
        boot_stats[b] = sample.mean()

    ci_lower = float(np.percentile(boot_stats, 2.5))
    ci_upper = float(np.percentile(boot_stats, 97.5))

    return {'rate': rate, 'ci_lower': ci_lower, 'ci_upper': ci_upper, 'n_pairs': n_pairs}


def decompose_inconsistency(
    pairs_df: pd.DataFrame,
    decisions_baseline: pd.Series,
    decisions_eo: pd.Series,
    postprocessor: dict,
    scores: pd.Series,
) -> pd.DataFrame:
    """
    For each pair, compute baseline/EO discordance and EO-introduced/resolved indicators.
    Also computes threshold distances for each individual.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Pair dataframe with columns id_i, id_j, (and optionally race_i, race_j).
    decisions_baseline : pd.Series
        Baseline binary decisions indexed by individual ID.
    decisions_eo : pd.Series
        EO binary decisions indexed by individual ID.
    postprocessor : dict
        EO postprocessor parameters (for threshold lookup).
    scores : pd.Series
        Calibrated probability scores indexed by individual ID.

    Returns
    -------
    pd.DataFrame
        Full pair-level decomposition, saved to results/pair_decomposition.csv.
    """
    def threshold_distance(score, race_val):
        """Compute min distance from score to either EO threshold for the individual's group."""
        group_key = f'group_{int(race_val)}'
        params = postprocessor[group_key]
        t_low = params['threshold_low']
        t_high = params['threshold_high']
        return min(abs(score - t_low), abs(score - t_high))

    df = pairs_df.copy()

    # Look up decisions
    db_i = decisions_baseline.loc[df['id_i']].values
    db_j = decisions_baseline.loc[df['id_j']].values
    deo_i = decisions_eo.loc[df['id_i']].values
    deo_j = decisions_eo.loc[df['id_j']].values

    df['inconsistent_baseline'] = (db_i != db_j).astype(int)
    df['inconsistent_eo'] = (deo_i != deo_j).astype(int)
    df['eo_introduced'] = ((df['inconsistent_eo'] == 1) & (df['inconsistent_baseline'] == 0)).astype(int)
    df['eo_resolved'] = ((df['inconsistent_eo'] == 0) & (df['inconsistent_baseline'] == 1)).astype(int)

    # Threshold distances
    scores_i = scores.loc[df['id_i']].values
    scores_j = scores.loc[df['id_j']].values

    race_i = df['race_i'].values if 'race_i' in df.columns else np.zeros(len(df))
    race_j = df['race_j'].values if 'race_j' in df.columns else np.zeros(len(df))

    df['threshold_distance_i'] = [
        threshold_distance(s, r) for s, r in zip(scores_i, race_i)
    ]
    df['threshold_distance_j'] = [
        threshold_distance(s, r) for s, r in zip(scores_j, race_j)
    ]

    df.to_csv(RESULTS_DIR + 'pair_decomposition.csv', index=False)
    logger.info("Saved pair decomposition to %spair_decomposition.csv", RESULTS_DIR)

    return df


def inconsistency_by_threshold_distance(
    pair_decomp_df: pd.DataFrame,
    n_quartiles: int = 4,
) -> pd.DataFrame:
    """
    Compute EO inconsistency rate by threshold-distance quartile.

    Bins individuals into quartiles based on their minimum threshold distance.
    For each quartile, computes the EO inconsistency rate for pairs where both
    individuals fall in that quartile.

    Parameters
    ----------
    pair_decomp_df : pd.DataFrame
        Output from decompose_inconsistency().
    n_quartiles : int
        Number of quantile bins (default 4 = quartiles).

    Returns
    -------
    pd.DataFrame
        Columns: quartile, inconsistency_rate_eo, inconsistency_rate_baseline, n_pairs.
    """
    df = pair_decomp_df.copy()

    # Use the minimum threshold distance of the pair (closer individual drives risk)
    df['min_thresh_dist'] = df[['threshold_distance_i', 'threshold_distance_j']].min(axis=1)

    # Compute quartile boundaries from the distribution of min distances
    quantile_labels = [f'Q{i+1}' for i in range(n_quartiles)]
    df['quartile'] = pd.qcut(
        df['min_thresh_dist'],
        q=n_quartiles,
        labels=quantile_labels,
        duplicates='drop',
    )

    rows = []
    for q in df['quartile'].cat.categories:
        mask = df['quartile'] == q
        sub = df[mask]
        if len(sub) == 0:
            continue
        rows.append({
            'quartile': q,
            'inconsistency_rate_eo': sub['inconsistent_eo'].mean(),
            'inconsistency_rate_baseline': sub['inconsistent_baseline'].mean(),
            'n_pairs': len(sub),
            'mean_thresh_dist': sub['min_thresh_dist'].mean(),
        })

    result = pd.DataFrame(rows)
    logger.info("Inconsistency by threshold-distance quartile:\n%s", result.to_string())

    return result
