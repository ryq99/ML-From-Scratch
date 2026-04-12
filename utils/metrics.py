"""
Evaluation Metrics
==================
Common metrics for regression and classification tasks.
"""
import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error: (1/n) * sum((y - y_hat)^2)"""
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error: (1/n) * sum(|y - y_hat|)"""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy: fraction of correct predictions."""
    return float(np.mean(y_true == y_pred))


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    """Binary cross-entropy: -(1/n) * sum(y*log(p) + (1-y)*log(1-p))"""
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Confusion matrix C where C[i, j] = number of samples with true label i
    predicted as label j.

    Returns
    -------
    np.ndarray of shape (n_classes, n_classes)
    """
    classes = np.unique(y_true)
    n = len(classes)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    C = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[label_to_idx[t], label_to_idx[p]] += 1
    return C
