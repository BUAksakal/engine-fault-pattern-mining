"""
Trains a Random Forest on the raw (continuous) sensor features to validate
that the patterns found by Apriori actually carry predictive signal for
the 3-class engine condition target, then uses SHAP to explain which
sensors drive individual predictions.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data_prep import SENSOR_COLUMNS, TARGET_COLUMN


def train_and_validate(df: pd.DataFrame, random_state: int = 42):
    X = df[SENSOR_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced", random_state=random_state
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc_ovr_macro": roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"),
    }

    return clf, X_test, y_test, metrics


def compute_shap_values(clf: RandomForestClassifier, X_sample: pd.DataFrame, class_index: int = 2):
    """Returns SHAP values for the given class index (default: 2 = Critical)."""
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        return shap_values[class_index]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, class_index]
    return shap_values
