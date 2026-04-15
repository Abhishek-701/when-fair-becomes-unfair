"""
Module 5: KNN-based and rule-based similarity pair construction for the test set.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from config import K_NEIGHBORS, RANDOM_SEED, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MAX_PAIRS_PER_PROFILE = 500


def build_knn_pairs(
    X_nr_test: pd.DataFrame,
    idx_test: pd.Index,
    race_test: pd.Series,
    k: int = K_NEIGHBORS,
    metric: str = 'euclidean',
) -> pd.DataFrame:
    """
    Construct KNN-based similarity pairs for the test set.

    For each individual, finds their k nearest neighbors in feature space.
    Records pair-level metadata including distance and race combination.

    Parameters
    ----------
    X_nr_test : pd.DataFrame
        Scaled feature matrix for the test set (no race).
    idx_test : pd.Index
        Original dataframe indices for the test set.
    race_test : pd.Series
        Binary race indicator indexed like X_nr_test.
    k : int
        Number of nearest neighbors.
    metric : str
        Distance metric ('euclidean' or 'cosine').

    Returns
    -------
    pd.DataFrame
        Columns: id_i, id_j, distance, race_i, race_j, pair_type.
        Saved to results/knn_pairs_{metric}.csv.
    """
    X = X_nr_test.values
    idx_arr = np.array(idx_test)
    race_arr = race_test.values

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)  # +1 because self is included
    nn.fit(X)
    distances, neighbors = nn.kneighbors(X)

    rows = []
    for i in range(len(X)):
        for j_pos in range(1, k + 1):  # skip self (position 0)
            j = neighbors[i, j_pos]
            dist = distances[i, j_pos]
            id_i = idx_arr[i]
            id_j = idx_arr[j]

            # Only keep each pair once (canonical order id_i < id_j)
            if id_i >= id_j:
                continue

            race_i = int(race_arr[i])
            race_j = int(race_arr[j])
            pair_type = 'same_race' if race_i == race_j else 'cross_race'

            rows.append({
                'id_i': id_i,
                'id_j': id_j,
                'distance': dist,
                'race_i': race_i,
                'race_j': race_j,
                'pair_type': pair_type,
            })

    pairs_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR + f'knn_pairs_{metric}.csv'
    pairs_df.to_csv(out_path, index=False)
    logger.info("KNN pairs (%s): %d pairs saved to %s", metric, len(pairs_df), out_path)

    return pairs_df


def build_rule_pairs(
    df_test: pd.DataFrame,
    race_test: pd.Series,
) -> pd.DataFrame:
    """
    Construct rule-based similarity pairs by grouping individuals with identical
    legal profiles (age bin × prior bin × charge severity).

    Within each profile, forms all pairwise combinations, capped at MAX_PAIRS_PER_PROFILE
    randomly sampled pairs to prevent O(n²) blowup in large profiles.

    Parameters
    ----------
    df_test : pd.DataFrame
        Cleaned test-set dataframe with columns: age, priors_count, charge_degree.
        Must have index matching the test set indices.
    race_test : pd.Series
        Binary race indicator indexed like df_test.

    Returns
    -------
    pd.DataFrame
        Columns: id_i, id_j, distance, race_i, race_j, pair_type, profile_id.
        Saved to results/rule_pairs.csv.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # Compute legal profile bins
    df = df_test.copy()
    df['race'] = race_test

    age_bins = [0, 25, 35, 100]
    age_labels = ['18-25', '26-35', '36+']
    df['age_bin'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)

    prior_bins = [-1, 0, 2, 5, 100]
    prior_labels = ['0', '1-2', '3-5', '6+']
    df['prior_bin'] = pd.cut(df['priors_count'], bins=prior_bins, labels=prior_labels, right=True)

    df['charge_bin'] = df['charge_degree'].map({1: 'felony', 0: 'misdemeanor'})

    df['profile_id'] = (
        df['age_bin'].astype(str) + '_' +
        df['prior_bin'].astype(str) + '_' +
        df['charge_bin'].astype(str)
    )

    profiles = df['profile_id'].unique()
    logger.info("Found %d unique legal profiles in the test set", len(profiles))

    rows = []
    for profile in profiles:
        members = df[df['profile_id'] == profile]
        idx_members = members.index.tolist()
        n = len(idx_members)

        if n < 2:
            continue

        # Generate all pairs
        all_pairs = [(idx_members[i], idx_members[j])
                     for i in range(n) for j in range(i + 1, n)]

        if len(all_pairs) > MAX_PAIRS_PER_PROFILE:
            warnings.warn(
                f"Profile '{profile}' has {len(all_pairs)} pairs; "
                f"capping at {MAX_PAIRS_PER_PROFILE}.",
                stacklevel=2,
            )
            logger.warning("Profile '%s' capped at %d pairs (had %d)",
                           profile, MAX_PAIRS_PER_PROFILE, len(all_pairs))
            chosen_indices = rng.choice(len(all_pairs), size=MAX_PAIRS_PER_PROFILE, replace=False)
            all_pairs = [all_pairs[ci] for ci in chosen_indices]

        for id_i, id_j in all_pairs:
            race_i = int(df.loc[id_i, 'race'])
            race_j = int(df.loc[id_j, 'race'])
            pair_type = 'same_race' if race_i == race_j else 'cross_race'
            rows.append({
                'id_i': id_i,
                'id_j': id_j,
                'distance': np.nan,  # not applicable for rule pairs
                'race_i': race_i,
                'race_j': race_j,
                'pair_type': pair_type,
                'profile_id': profile,
            })

    pairs_df = pd.DataFrame(rows)
    pairs_df.to_csv(RESULTS_DIR + 'rule_pairs.csv', index=False)
    logger.info("Rule-based pairs: %d pairs saved to %srule_pairs.csv",
                len(pairs_df), RESULTS_DIR)

    return pairs_df
