"""
Module 2: Data loading, cleaning, filtering, feature encoding, and train/test split
for the COMPAS recidivism dataset.
"""

import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_RAW_PATH, DATA_CLEAN_PATH, RESULTS_DIR, FEATURES_NR,
    RANDOM_SEED, TEST_SIZE,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_and_clean(path: str = DATA_RAW_PATH) -> pd.DataFrame:
    """
    Load the COMPAS dataset from disk, apply ProPublica standard filters,
    rename and retain only the required columns, and encode protected attributes.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with retained and renamed columns.
    """
    df = pd.read_csv(path)
    logger.info("Loaded raw data: %d rows, %d columns", len(df), df.shape[1])

    # --- ProPublica standard filters ---
    n_before = len(df)
    df = df[df['days_b_screening_arrest'] <= 30]
    df = df[df['days_b_screening_arrest'] >= -30]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != 'O']
    df = df[df['score_text'] != 'N/A']

    n_after_general = len(df)
    n_other_race = len(df[~df['race'].isin(['African-American', 'Caucasian'])])
    logger.info("Removed %d individuals with race other than African-American/Caucasian",
                n_other_race)

    df = df[df['race'].isin(['African-American', 'Caucasian'])]
    n_after = len(df)

    logger.info(
        "Rows before filtering: %d | After general filters: %d | After race filter: %d",
        n_before, n_after_general, n_after,
    )

    # --- Retain and rename columns ---
    rename_map = {
        'id': 'id',
        'age': 'age',
        'race': 'race_raw',
        'sex': 'sex',
        'juv_fel_count': 'juv_fel_count',
        'juv_misd_count': 'juv_misd_count',
        'juv_other_count': 'juv_other_count',
        'priors_count': 'priors_count',
        'c_charge_degree': 'charge_degree',
        'two_year_recid': 'two_year_recid',
    }
    df = df[list(rename_map.keys())].rename(columns=rename_map).copy()

    # --- Encode categorical columns ---
    df['sex'] = (df['sex'] == 'Male').astype(int)
    df['charge_degree'] = (df['charge_degree'] == 'F').astype(int)

    # Protected attribute encoding (documented decision)
    # race = 1 → African-American ("Black"); race = 0 → Caucasian ("White")
    df['race'] = (df['race_raw'] == 'African-American').astype(int)

    logger.info("Final cleaned dataset: %d rows", len(df))
    logger.info("Race distribution — Black: %d (%.1f%%), White: %d (%.1f%%)",
                df['race'].sum(), 100 * df['race'].mean(),
                (df['race'] == 0).sum(), 100 * (1 - df['race'].mean()))
    logger.info("Outcome balance — recidivated: %d (%.1f%%)",
                df['two_year_recid'].sum(), 100 * df['two_year_recid'].mean())

    df.to_csv(DATA_CLEAN_PATH, index=False)
    logger.info("Saved cleaned data to %s", DATA_CLEAN_PATH)

    return df


def encode_features(df: pd.DataFrame):
    """
    Build feature matrices and the target vector from the cleaned dataframe.
    Applies StandardScaler to all numeric features in X_nr (excluding race).
    Saves the fitted scaler to results/scaler.pkl.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe from load_and_clean().

    Returns
    -------
    X : pd.DataFrame
        Full feature matrix including binary 'race' column.
    X_nr : pd.DataFrame
        Feature matrix excluding 'race' (used for modeling and similarity).
    y : pd.Series
        Binary outcome series (two_year_recid).
    """
    # Feature matrix with race
    X = df[FEATURES_NR + ['race']].copy()

    # Feature matrix without race (for modeling)
    X_nr_raw = df[FEATURES_NR].copy()

    # Scale all numeric columns in X_nr
    scaler = StandardScaler()
    X_nr_scaled = scaler.fit_transform(X_nr_raw)
    X_nr = pd.DataFrame(X_nr_scaled, columns=FEATURES_NR, index=df.index)

    # Save scaler
    joblib.dump(scaler, RESULTS_DIR + 'scaler.pkl')
    logger.info("Saved fitted scaler to %sscaler.pkl", RESULTS_DIR)

    y = df['two_year_recid'].copy()

    return X, X_nr, y


def split_data(X_nr: pd.DataFrame, y: pd.Series, race: pd.Series) -> dict:
    """
    Split data into train and test sets, stratifying jointly on race and outcome.

    Parameters
    ----------
    X_nr : pd.DataFrame
        Feature matrix without race.
    y : pd.Series
        Binary outcome.
    race : pd.Series
        Binary race indicator (1=Black, 0=White).

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test, race_train, race_test, idx_test.
    """
    # Create combined stratum label for stratification
    race_y = race.astype(str) + '_' + y.astype(str)

    (X_train, X_test,
     y_train, y_test,
     race_train, race_test) = train_test_split(
        X_nr, y, race,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=race_y,
    )

    idx_test = X_test.index

    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))
    logger.info("Test race — Black: %d, White: %d",
                race_test.sum(), (race_test == 0).sum())
    logger.info("Test outcome — recidivated: %d (%.1f%%)",
                y_test.sum(), 100 * y_test.mean())

    # Save indices
    pd.Series(X_train.index, name='idx').to_csv(RESULTS_DIR + 'train_idx.csv', index=False)
    pd.Series(idx_test, name='idx').to_csv(RESULTS_DIR + 'test_idx.csv', index=False)
    logger.info("Saved train_idx.csv and test_idx.csv")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'race_train': race_train,
        'race_test': race_test,
        'idx_test': idx_test,
    }


if __name__ == '__main__':
    df = load_and_clean()
    X, X_nr, y = encode_features(df)
    race = df['race']
    split = split_data(X_nr, y, race)
    print("Data preparation complete.")
    print(f"  Train: {len(split['X_train'])} samples")
    print(f"  Test:  {len(split['X_test'])} samples")
