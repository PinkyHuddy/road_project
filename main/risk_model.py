"""Reusable scoring helpers for the I-80 closure risk artifact."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


RISK_LABELS = np.array(["Low", "Medium", "High", "Extreme"], dtype=object)


def assign_risk_category(probabilities, thresholds):
    """Map closure probabilities to ordered risk categories."""
    probabilities = np.asarray(probabilities, dtype=float)
    cutoffs = np.asarray(
        [thresholds["low_medium"], thresholds["medium_high"], thresholds["high_extreme"]],
        dtype=float,
    )
    if not np.all(np.diff(cutoffs) > 0):
        raise ValueError("Risk thresholds must be strictly increasing.")
    return RISK_LABELS[np.digitize(probabilities, cutoffs, right=False)]


def load_risk_artifact(path):
    """Load a saved model bundle."""
    return joblib.load(Path(path))


def score_feature_frame(feature_frame, artifact):
    """Score a frame that already contains the artifact's model features."""
    required = artifact["metadata"]["model_features"]
    missing = [column for column in required if column not in feature_frame.columns]
    if missing:
        raise KeyError(f"Scoring frame is missing required features: {missing}")

    probabilities = artifact["pipeline"].predict_proba(feature_frame[required])[:, 1]
    categories = assign_risk_category(probabilities, artifact["category_thresholds"])
    return pd.DataFrame(
        {"closure_risk_score": probabilities, "closure_risk_category": categories},
        index=feature_frame.index,
    )
