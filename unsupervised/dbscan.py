"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
=====================================================================
Clusters points that are densely packed; marks sparse points as noise (-1).

Key concepts:
  eps         : neighborhood radius
  min_samples : minimum neighbors to be a core point
  Core point  : has >= min_samples points within eps distance
  Border point: within eps of a core point but not itself a core point
  Noise       : neither core nor border

Algorithm:
  For each unvisited point:
    - Find all eps-neighbors.
    - If >= min_samples: start a new cluster and expand recursively.
    - Else: mark as noise (may be reassigned as border later).
"""
import numpy as np
from collections import deque


# ── NumPy ─────────────────────────────────────────────────────────────────────

class DBSCANNumPy:
    """
    Parameters
    ----------
    eps         : neighborhood radius
    min_samples : minimum points to form a dense region
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: np.ndarray = None   # -1 = noise

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        X : (n_samples, n_features)
        Returns cluster labels, shape (n_samples,).  -1 = noise.
        """
        n = len(X)
        # Precompute pairwise squared distances to avoid sqrt
        sq_dists = (
            (X ** 2).sum(axis=1, keepdims=True)
            - 2 * X @ X.T
            + (X ** 2).sum(axis=1)
        )
        neighbors = [np.where(sq_dists[i] <= self.eps ** 2)[0] for i in range(n)]

        labels = np.full(n, -1, dtype=int)
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1:
                continue
            if len(neighbors[i]) < self.min_samples:
                continue   # noise (tentative)

            # Start new cluster with BFS
            labels[i] = cluster_id
            queue = deque(neighbors[i])
            while queue:
                j = queue.popleft()
                if labels[j] == -1:          # noise → border point
                    labels[j] = cluster_id
                if labels[j] != -1 and labels[j] != cluster_id:
                    continue                  # already in another cluster (border)
                labels[j] = cluster_id
                if len(neighbors[j]) >= self.min_samples:
                    for nb in neighbors[j]:
                        if labels[nb] == -1:
                            queue.append(nb)

            cluster_id += 1

        self.labels_ = labels
        return labels


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class DBSCANTorch:
    """
    DBSCAN with PyTorch tensor distance computations.

    Parameters
    ----------
    eps         : neighborhood radius
    min_samples : minimum points to form a dense region
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: np.ndarray = None

    def fit_predict(self, X) -> np.ndarray:
        """
        X : array-like (n_samples, n_features)
        Returns cluster labels, shape (n_samples,).  -1 = noise.
        """
        X = torch.tensor(X, dtype=torch.float32)
        n = len(X)
        # Pairwise Euclidean distance matrix: (n, n)
        dists = torch.cdist(X, X)
        # Boolean neighbor mask
        neighbor_mask = dists <= self.eps   # (n, n)
        neighbor_counts = neighbor_mask.sum(dim=1)   # (n,)

        labels = np.full(n, -1, dtype=int)
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1:
                continue
            if neighbor_counts[i].item() < self.min_samples:
                continue

            labels[i] = cluster_id
            queue = deque(neighbor_mask[i].nonzero(as_tuple=False).squeeze(1).tolist())

            while queue:
                j = queue.popleft()
                if labels[j] == -1:
                    labels[j] = cluster_id
                if labels[j] != cluster_id:
                    continue
                labels[j] = cluster_id
                if neighbor_counts[j].item() >= self.min_samples:
                    for nb in neighbor_mask[j].nonzero(as_tuple=False).squeeze(1).tolist():
                        if labels[nb] == -1:
                            queue.append(nb)

            cluster_id += 1

        self.labels_ = labels
        return labels
