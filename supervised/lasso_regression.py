"""
Lasso Regression (L1 Regularization)
======================================
Minimize  (1/2n) ||Xw - y||^2 + alpha * ||w||_1

The L1 penalty is non-differentiable at zero.  Coordinate descent applies
the soft-thresholding operator:
    S(z, gamma) = sign(z) * max(|z| - gamma, 0)

Update per coordinate j:
    rho_j = X_j^T (y - X_{-j} w_{-j})
    w_j   = S(rho_j, alpha * n) / (X_j^T X_j)
"""
import numpy as np


# ── NumPy ─────────────────────────────────────────────────────────────────────

class LassoRegressionNumPy:
    """
    Parameters
    ----------
    alpha    : L1 regularization strength
    n_iters  : coordinate descent passes
    tol      : convergence tolerance
    """

    def __init__(self, alpha: float = 1.0, n_iters: int = 1000, tol: float = 1e-4):
        self.alpha = alpha
        self.n_iters = n_iters
        self.tol = tol
        self.w: np.ndarray = None
        self.b: float = 0.0

    @staticmethod
    def _soft_threshold(z: np.ndarray, gamma: float) -> np.ndarray:
        return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegressionNumPy":
        """
        X : (n_samples, n_features)
        y : (n_samples,)
        """
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.n_iters):
            w_old = self.w.copy()
            # Update bias (unregularized)
            self.b = (y - X @ self.w).mean()
            # Coordinate descent over features
            for j in range(d):
                # Partial residual: y - X_{-j} w_{-j} - b
                r_j = y - self.b - X @ self.w + X[:, j] * self.w[j]
                rho_j = X[:, j] @ r_j
                x_j_sq = X[:, j] @ X[:, j]
                if x_j_sq == 0:
                    self.w[j] = 0.0
                else:
                    self.w[j] = self._soft_threshold(rho_j, self.alpha * n) / x_j_sq
            if np.max(np.abs(self.w - w_old)) < self.tol:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predictions, shape (n_samples,)."""
        return X @ self.w + self.b


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class LassoRegressionTorch:
    """
    Lasso regression with raw PyTorch tensors and manual subgradient computation.
    No autograd — gradients derived analytically.

    Forward:  y_hat = X @ w + b
    Loss:     L = (1/n)*||y_hat - y||^2 + alpha*||w||_1
    Gradients (subgradient for L1 — sign(0) = 0 is a valid subgradient):
        dL/dw = (2/n) * X^T @ (y_hat - y)  +  alpha * sign(w)
        dL/db = (2/n) * sum(y_hat - y)

    Parameters
    ----------
    alpha   : L1 regularization strength
    lr      : learning rate
    n_iters : gradient steps
    """

    def __init__(self, alpha: float = 1.0, lr: float = 0.01, n_iters: int = 1000):
        self.alpha = alpha
        self.lr = lr
        self.n_iters = n_iters
        self.w: torch.Tensor = None
        self.b: torch.Tensor = None

    def fit(self, X, y) -> "LassoRegressionTorch":
        """
        X : array-like (n_samples, n_features)
        y : array-like (n_samples,)
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        n, d = X.shape

        self.w = torch.zeros(d)
        self.b = torch.zeros(1)

        for _ in range(self.n_iters):
            y_hat = X @ self.w + self.b            # (n,)
            error = y_hat - y                      # (n,)

            dw = (2 / n) * X.T @ error + self.alpha * torch.sign(self.w)  # (d,)
            db = (2 / n) * error.sum()             # scalar

            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def predict(self, X) -> np.ndarray:
        """Returns predictions as numpy array, shape (n_samples,)."""
        X = torch.tensor(X, dtype=torch.float32)
        return (X @ self.w + self.b).numpy()
