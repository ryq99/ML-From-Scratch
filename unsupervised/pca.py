"""
Principal Component Analysis (PCA)
====================================
Finds the directions of maximum variance in data by computing the eigenvectors
of the covariance matrix.  Implemented via SVD for numerical stability.

Steps:
  1. Center: X_c = X - mean(X)
  2. SVD:    X_c = U S V^T
  3. Top-k principal components: V[:, :k]  (right singular vectors)
  4. Project: Z = X_c @ V[:, :k]
"""
import numpy as np


# ── NumPy ─────────────────────────────────────────────────────────────────────

class PCANumPy:
    """
    Parameters
    ----------
    n_components : number of principal components to keep
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_: np.ndarray = None   # (n_components, n_features)
        self.mean_: np.ndarray = None
        self.explained_variance_: np.ndarray = None

    def fit(self, X: np.ndarray) -> "PCANumPy":
        """
        X : (n_samples, n_features)
        """
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        # SVD: X_c = U S V^T;  columns of V are principal components
        _, s, Vt = np.linalg.svd(X_c, full_matrices=False)
        self.components_ = Vt[:self.n_components]          # (k, d)
        # Explained variance = eigenvalues of covariance matrix = s^2 / (n-1)
        self.explained_variance_ = (s ** 2) / (len(X) - 1)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X onto principal components.
        Returns shape (n_samples, n_components).
        """
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct from reduced representation, shape (n_samples, n_features)."""
        return Z @ self.components_ + self.mean_


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class PCATorch:
    """
    PCA with raw PyTorch tensors using torch.linalg.svd.

    Parameters
    ----------
    n_components : number of principal components to keep
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_: torch.Tensor = None
        self.mean_: torch.Tensor = None
        self.explained_variance_: torch.Tensor = None

    def fit(self, X) -> "PCATorch":
        X = torch.tensor(X, dtype=torch.float32)
        self.mean_ = X.mean(dim=0)
        X_c = X - self.mean_
        # full_matrices=False gives economy SVD: U (n,k), S (k,), Vh (k,d)
        _, s, Vh = torch.linalg.svd(X_c, full_matrices=False)
        self.components_ = Vh[:self.n_components]           # (k, d)
        self.explained_variance_ = (s ** 2) / (len(X) - 1)
        return self

    def transform(self, X) -> np.ndarray:
        """Returns projection, shape (n_samples, n_components)."""
        X = torch.tensor(X, dtype=torch.float32)
        Z = (X - self.mean_) @ self.components_.T
        return Z.numpy()

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z) -> np.ndarray:
        Z = torch.tensor(Z, dtype=torch.float32)
        return (Z @ self.components_ + self.mean_).numpy()
