"""
K-Means Clustering (Lloyd's Algorithm)
=======================================
Iterates between two steps until centroids converge:
  E-step: assign each point to nearest centroid
          label_i = argmin_k ||x_i - mu_k||^2
  M-step: update centroids to cluster means
          mu_k = (1/|C_k|) * sum_{i in C_k} x_i
"""
import numpy as np


# ── NumPy ─────────────────────────────────────────────────────────────────────

class KMeansNumPy:
    """
    Parameters
    ----------
    k       : number of clusters
    n_iters : maximum iterations
    tol     : convergence tolerance (centroid movement)
    """

    def __init__(self, k: int = 3, n_iters: int = 300, tol: float = 1e-4):
        self.k = k
        self.n_iters = n_iters
        self.tol = tol
        self.centroids: np.ndarray = None

    def fit(self, X: np.ndarray) -> "KMeansNumPy":
        """
        X : (n_samples, n_features)
        Initializes centroids with random samples (Forgy init).
        """
        n, d = X.shape
        idx = np.random.choice(n, size=self.k, replace=False)
        self.centroids = X[idx].copy()                   # (k, d)

        for _ in range(self.n_iters):
            labels = self._assign(X)
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if (labels == j).any() else self.centroids[j]
                for j in range(self.k)
            ])
            if np.max(np.linalg.norm(new_centroids - self.centroids, axis=1)) < self.tol:
                break
            self.centroids = new_centroids

        return self

    def _assign(self, X: np.ndarray) -> np.ndarray:
        """Returns cluster index for each sample, shape (n,)."""
        # sq distances: (n, k)  = ||x||^2 - 2*X@C^T + ||c||^2
        sq_dists = (
            (X ** 2).sum(axis=1, keepdims=True)
            - 2 * X @ self.centroids.T
            + (self.centroids ** 2).sum(axis=1)
        )
        return np.argmin(sq_dists, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns cluster labels, shape (n,)."""
        return self._assign(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class KMeansTorch:
    """
    K-Means with raw PyTorch tensors.  Uses torch.cdist for pairwise distances.

    Parameters
    ----------
    k       : number of clusters
    n_iters : maximum iterations
    tol     : convergence tolerance
    """

    def __init__(self, k: int = 3, n_iters: int = 300, tol: float = 1e-4):
        self.k = k
        self.n_iters = n_iters
        self.tol = tol
        self.centroids: torch.Tensor = None

    def fit(self, X) -> "KMeansTorch":
        X = torch.tensor(X, dtype=torch.float32)
        n = X.shape[0]
        idx = torch.randperm(n)[:self.k]
        self.centroids = X[idx].clone()

        for _ in range(self.n_iters):
            labels = self._assign(X)
            new_centroids = torch.stack([
                X[labels == j].mean(dim=0) if (labels == j).any() else self.centroids[j]
                for j in range(self.k)
            ])
            shift = (new_centroids - self.centroids).norm(dim=1).max().item()
            self.centroids = new_centroids
            if shift < self.tol:
                break

        return self

    def _assign(self, X: torch.Tensor) -> torch.Tensor:
        """Returns cluster indices, shape (n,)."""
        dists = torch.cdist(X, self.centroids)   # (n, k) Euclidean distances
        return torch.argmin(dists, dim=1)

    def predict(self, X) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32)
        return self._assign(X).numpy()

    def fit_predict(self, X) -> np.ndarray:
        return self.fit(X).predict(X)
