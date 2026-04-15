"""
Module 3: Logistic regression baseline with isotonic calibration,
score output, and per-group fairness metrics.
"""

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import RANDOM_SEED, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
    """
    Train a logistic regression model wrapped in isotonic calibration.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (no race).
    y_train : pd.Series
        Training labels.

    Returns
    -------
    CalibratedClassifierCV
        Fitted calibrated classifier saved to results/baseline_model.pkl.
    """
    base_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model = CalibratedClassifierCV(base_lr, method='isotonic', cv=5)
    model.fit(X_train, y_train)

    joblib.dump(model, RESULTS_DIR + 'baseline_model.pkl')
    logger.info("Saved baseline model to %sbaseline_model.pkl", RESULTS_DIR)

    return model


def score_and_decide(
    model: CalibratedClassifierCV,
    X_test: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Generate calibrated probability scores and binary decisions for the test set.

    Parameters
    ----------
    model : CalibratedClassifierCV
        Fitted calibrated model.
    X_test : pd.DataFrame
        Test feature matrix (no race).
    threshold : float
        Decision threshold (default 0.5).

    Returns
    -------
    pd.DataFrame
        Columns: 'score', 'decision_baseline'. Index matches X_test.
    """
    scores = model.predict_proba(X_test)[:, 1]
    decisions = (scores >= threshold).astype(int)

    scores_df = pd.DataFrame(
        {'score': scores, 'decision_baseline': decisions},
        index=X_test.index,
    )

    scores_df.to_csv(RESULTS_DIR + 'baseline_scores.csv')
    logger.info("Saved baseline scores to %sbaseline_scores.csv", RESULTS_DIR)

    return scores_df


def _group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute FPR, FNR, TPR, and accuracy for a single group.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted binary labels.

    Returns
    -------
    dict with keys: fpr, fnr, tpr, accuracy.
    """
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {'fpr': fpr, 'fnr': fnr, 'tpr': tpr, 'accuracy': acc}


def compute_baseline_metrics(
    y_test: pd.Series,
    scores_df: pd.DataFrame,
    race_test: pd.Series,
) -> dict:
    """
    Compute overall and per-group fairness metrics for the baseline model.

    Parameters
    ----------
    y_test : pd.Series
        True binary labels for the test set.
    scores_df : pd.DataFrame
        Output from score_and_decide(); must contain 'score' and 'decision_baseline'.
    race_test : pd.Series
        Binary race indicator for the test set (1=Black, 0=White).

    Returns
    -------
    dict containing overall and per-group metrics plus disparities.
    """
    y = y_test.values
    scores = scores_df['score'].values
    decisions = scores_df['decision_baseline'].values
    race = race_test.values

    overall_acc = float((decisions == y).mean())
    overall_auc = float(roc_auc_score(y, scores))

    mask_black = race == 1
    mask_white = race == 0

    metrics_black = _group_metrics(y[mask_black], decisions[mask_black])
    metrics_white = _group_metrics(y[mask_white], decisions[mask_white])

    fpr_disparity = metrics_black['fpr'] - metrics_white['fpr']
    fnr_disparity = metrics_black['fnr'] - metrics_white['fnr']

    result = {
        'overall': {'accuracy': overall_acc, 'auc_roc': overall_auc},
        'black': metrics_black,
        'white': metrics_white,
        'fpr_disparity': fpr_disparity,
        'fnr_disparity': fnr_disparity,
    }

    # Print formatted table
    print("\n=== Baseline Model Metrics ===")
    print(f"  Overall accuracy: {overall_acc:.1%}  |  AUC-ROC: {overall_auc:.3f}")
    print(f"  {'Group':<10} {'FPR':>7} {'FNR':>7} {'TPR':>7} {'Acc':>7}")
    print(f"  {'Black':<10} {metrics_black['fpr']:>7.1%} {metrics_black['fnr']:>7.1%} "
          f"{metrics_black['tpr']:>7.1%} {metrics_black['accuracy']:>7.1%}")
    print(f"  {'White':<10} {metrics_white['fpr']:>7.1%} {metrics_white['fnr']:>7.1%} "
          f"{metrics_white['tpr']:>7.1%} {metrics_white['accuracy']:>7.1%}")
    print(f"  FPR disparity (Black - White): {fpr_disparity:+.3f}")
    print(f"  FNR disparity (Black - White): {fnr_disparity:+.3f}")

    with open(RESULTS_DIR + 'baseline_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    logger.info("Saved baseline_metrics.json")

    return result


if __name__ == '__main__':
    import data_prep as dp

    df = dp.load_and_clean()
    _, X_nr, y = dp.encode_features(df)
    race = df['race']
    split = dp.split_data(X_nr, y, race)

    model = train_baseline(split['X_train'], split['y_train'])
    scores_df = score_and_decide(model, split['X_test'])
    metrics = compute_baseline_metrics(split['y_test'], scores_df, split['race_test'])
