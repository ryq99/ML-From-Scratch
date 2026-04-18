"""
Elastic Net Regression
======================
Combines L1 and L2 penalties:

    Minimize  (1/2n)||Xw - y||^2 + alpha*l1_ratio*||w||_1 + (alpha*(1-l1_ratio)/2)*||w||^2

When l1_ratio=1 -> Lasso; l1_ratio=0 -> Ridge.

Coordinate descent update for feature j:
    rho_j  = X_j^T (y - X_{-j}w_{-j} - b)
    denom  = X_j^T X_j + alpha * (1 - l1_ratio) * n
    w_j    = S(rho_j, alpha * l1_ratio * n) / denom
"""
import numpy as np


# ── NumPy ─────────────────────────────────────────────────────────────────────

class ElasticNetNumPy:
    """
    Parameters
    ----------
    alpha     : overall regularization strength
    l1_ratio  : mixing parameter in [0, 1]; 1 = pure Lasso, 0 = pure Ridge
    n_iters   : coordinate descent passes
    tol       : convergence tolerance
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5,
                 n_iters: int = 1000, tol: float = 1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_iters = n_iters
        self.tol = tol
        self.w: np.ndarray = None
        self.b: float = 0.0

    @staticmethod
    def _soft_threshold(z: float, gamma: float) -> float:
        return np.sign(z) * max(abs(z) - gamma, 0.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetNumPy":
        """
        X : (n_samples, n_features)
        y : (n_samples,)
        """
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        l1_pen = self.alpha * self.l1_ratio * n
        l2_pen = self.alpha * (1 - self.l1_ratio) * n

        for _ in range(self.n_iters):
            w_old = self.w.copy()
            self.b = (y - X @ self.w).mean()
            for j in range(d):
                r_j = y - self.b - X @ self.w + X[:, j] * self.w[j]
                rho_j = X[:, j] @ r_j
                denom = X[:, j] @ X[:, j] + l2_pen
                self.w[j] = self._soft_threshold(rho_j, l1_pen) / denom
            if np.max(np.abs(self.w - w_old)) < self.tol:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predictions, shape (n_samples,)."""
        return X @ self.w + self.b


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class ElasticNetTorch:
    """
    Elastic net with raw PyTorch tensors and manual gradients.
    No autograd — gradients derived analytically.

    Forward:  y_hat = X @ w + b
    Loss:     L = (1/n)*||y_hat-y||^2 + alpha*l1_ratio*||w||_1
                                       + alpha*(1-l1_ratio)*||w||^2
    Gradients:
        dL/dw = (2/n)*X^T@(y_hat-y) + alpha*l1_ratio*sign(w)
                                     + 2*alpha*(1-l1_ratio)*w
        dL/db = (2/n)*sum(y_hat-y)

    Parameters
    ----------
    alpha     : overall regularization strength
    l1_ratio  : mixing parameter in [0, 1]
    lr        : learning rate
    n_iters   : gradient steps
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5,
                 lr: float = 0.01, n_iters: int = 1000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.lr = lr
        self.n_iters = n_iters
        self.w: torch.Tensor = None
        self.b: torch.Tensor = None

    def fit(self, X, y) -> "ElasticNetTorch":
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

            dw = (
                (2 / n) * X.T @ error
                + self.alpha * self.l1_ratio * torch.sign(self.w)
                + 2 * self.alpha * (1 - self.l1_ratio) * self.w
            )                                      # (d,)
            db = (2 / n) * error.sum()             # scalar

            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def predict(self, X) -> np.ndarray:
        """Returns predictions as numpy array, shape (n_samples,)."""
        X = torch.tensor(X, dtype=torch.float32)
        return (X @ self.w + self.b).numpy()
