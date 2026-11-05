"""Metric computation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from . import config as C


def scores(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def binary_metrics(y_true, y_score, threshold: float = C.THRESHOLD) -> dict:
    """Standard detection metrics at a fixed decision threshold."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_score),
        pr_auc=average_precision_score(y_true, y_score),
        fpr=fp / (fp + tn) if (fp + tn) else 0.0,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )
