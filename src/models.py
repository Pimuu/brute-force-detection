"""Detectors compared in the study.

* RuleBaseline  - a fail2ban-style threshold: flag a session with >= N failures.
                  Its "score" is the failure count, so it also gets a ROC curve.
* LogisticRegression - linear model on log1p-transformed counts + standardised
                  features, class-balanced loss (interpretable coefficients).
* RandomForest  - 300 trees, class-balanced bootstrap, captures interactions
                  such as "many usernames AND high failure ratio".
* HistGradientBoosting - boosted trees, a strong tabular baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import config as C


class RuleBaseline(BaseEstimator, ClassifierMixin):
    """Flag sessions whose failure count reaches `min_failures` (no learning)."""

    def __init__(self, min_failures: int = C.RULE_MIN_FAILURES):
        self.min_failures = min_failures

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def _failures(self, X):
        return (X["n_attempts"] * X["failure_ratio"]).round().to_numpy()

    def predict_proba(self, X):
        # Map failure count to a monotone pseudo-probability; the 0.5 point is
        # exactly the rule threshold so predict() == (failures >= min_failures).
        f = self._failures(X)
        p = 1.0 / (1.0 + np.exp(-(f - self.min_failures + 0.5)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self._failures(X) >= self.min_failures).astype(int)


def make_models(seed: int = C.SEED) -> dict:
    log_cols = [C.FEATURES.index(c) for c in C.COUNT_FEATURES]
    other_cols = [i for i in range(len(C.FEATURES)) if i not in log_cols]
    lr_pre = ColumnTransformer([
        ("log", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), log_cols),
        ("std", StandardScaler(), other_cols),
    ])
    return {
        "Rule (>=5 failures)": RuleBaseline(),
        "Logistic Regression": Pipeline([
            ("pre", lr_pre),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=seed),
    }
