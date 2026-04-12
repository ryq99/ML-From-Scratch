"""
K-Nearest Neighbors (KNN)
=========================
Prediction for a query x: find the k training samples nearest in Euclidean
distance and return the majority class (classification) or mean (regression).

No training phase — all computation happens at predict time.
"""
import numpy as np
from collections import Counter


# ── NumPy ─────────────────────────────────────────────────────────────────────

class KNNClassifierNumPy:
    """
    Parameters
    ----------
    k : number of neighbors
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: np.ndarray = None
        self.y_train: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifierNumPy":
        """Store training data. X: (n, d), y: (n,)."""
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X : (m, d)
        Returns predicted labels, shape (m,).
        """
        # Pairwise squared distances: ||x - x_i||^2
        # = ||x||^2 - 2 x·x_i + ||x_i||^2
        sq_dists = (
            (X ** 2).sum(axis=1, keepdims=True)
            - 2 * X @ self.X_train.T
            + (self.X_train ** 2).sum(axis=1)
        )                                              # (m, n)
        nn_idx = np.argpartition(sq_dists, self.k, axis=1)[:, :self.k]
        preds = []
        for idx in nn_idx:
            votes = Counter(self.y_train[idx])
            preds.append(votes.most_common(1)[0][0])
        return np.array(preds)


class KNNRegressorNumPy:
    """
    Parameters
    ----------
    k : number of neighbors
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: np.ndarray = None
        self.y_train: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressorNumPy":
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        sq_dists = (
            (X ** 2).sum(axis=1, keepdims=True)
            - 2 * X @ self.X_train.T
            + (self.X_train ** 2).sum(axis=1)
        )
        nn_idx = np.argpartition(sq_dists, self.k, axis=1)[:, :self.k]
        return self.y_train[nn_idx].mean(axis=1)


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class KNNClassifierTorch:
    """
    KNN classifier with PyTorch tensors.
    Uses torch.cdist for pairwise distances.

    Parameters
    ----------
    k : number of neighbors
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: torch.Tensor = None
        self.y_train: np.ndarray = None

    def fit(self, X, y) -> "KNNClassifierTorch":
        self.X_train = torch.tensor(X, dtype=torch.float32)
        self.y_train = np.array(y)
        return self

    def predict(self, X) -> np.ndarray:
        """Returns predicted labels, shape (m,)."""
        X = torch.tensor(X, dtype=torch.float32)
        # cdist computes pairwise Euclidean distances: (m, n)
        dists = torch.cdist(X, self.X_train)
        # topk with largest=False gives k smallest distances
        _, nn_idx = torch.topk(dists, self.k, dim=1, largest=False)
        nn_idx = nn_idx.numpy()
        preds = []
        for idx in nn_idx:
            votes = Counter(self.y_train[idx])
            preds.append(votes.most_common(1)[0][0])
        return np.array(preds)
