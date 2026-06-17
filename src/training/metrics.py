"""Evaluation metrics for protein localization classification."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
)


def compute_metrics(preds, labels):
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "mcc": matthews_corrcoef(labels, preds),
    }
