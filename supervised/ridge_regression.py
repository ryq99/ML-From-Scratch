"""
Ridge Regression (L2 Regularization)
=====================================
Minimize  ||Xw - y||^2 + alpha * ||w||^2

Closed-form: w = (X^T X + alpha * I)^{-1} X^T y
Gradient:    dL/dw = (2/n) X^T (Xw - y) + 2*alpha*w
"""
import numpy as np


# ── NumPy ─────────────────────────────────────────────────────────────────────

class RidgeRegressionNumPy:
    """
    Parameters
    ----------
    alpha   : L2 regularization strength
    method  : 'normal' (closed-form) | 'gd' (gradient descent)
    lr      : learning rate (used when method='gd')
    n_iters : gradient descent steps
    """

    def __init__(self, alpha: float = 1.0, method: str = "normal",
                 lr: float = 0.01, n_iters: int = 1000):
        self.alpha = alpha
        self.method = method
        self.lr = lr
        self.n_iters = n_iters
        self.w: np.ndarray = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressionNumPy":
        """
        X : (n_samples, n_features)
        y : (n_samples,)
        """
        n, d = X.shape
        if self.method == "normal":
            # Bias absorbed into augmented X
            X_b = np.c_[np.ones(n), X]                     # (n, d+1)
            I = np.eye(d + 1)
            I[0, 0] = 0                                     # don't regularize bias
            params = np.linalg.solve(X_b.T @ X_b + self.alpha * I, X_b.T @ y)
            self.b, self.w = params[0], params[1:]
        else:
            self.w = np.zeros(d)
            self.b = 0.0
            for _ in range(self.n_iters):
                y_hat = X @ self.w + self.b
                error = y_hat - y
                self.w -= self.lr * ((2 / n) * X.T @ error + 2 * self.alpha * self.w)
                self.b -= self.lr * (2 / n) * error.sum()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predictions, shape (n_samples,)."""
        return X @ self.w + self.b


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class RidgeRegressionTorch:
    """
    Ridge regression with raw PyTorch tensors.

    Parameters
    ----------
    alpha   : L2 regularization strength
    lr      : learning rate
    n_iters : gradient descent steps
    """

    def __init__(self, alpha: float = 1.0, lr: float = 0.01, n_iters: int = 1000):
        self.alpha = alpha
        self.lr = lr
        self.n_iters = n_iters
        self.w: torch.Tensor = None
        self.b: torch.Tensor = None

    def fit(self, X, y) -> "RidgeRegressionTorch":
        """
        X : array-like (n_samples, n_features)
        y : array-like (n_samples,)
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        n, d = X.shape

        self.w = torch.zeros(d, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

        for _ in range(self.n_iters):
            y_hat = X @ self.w + self.b
            mse = ((y_hat - y) ** 2).mean()
            l2 = self.alpha * (self.w ** 2).sum()
            loss = mse + l2

            loss.backward()
            with torch.no_grad():
                self.w -= self.lr * self.w.grad
                self.b -= self.lr * self.b.grad
            self.w.grad.zero_()
            self.b.grad.zero_()

        return self

    def predict(self, X) -> np.ndarray:
        """Returns predictions as numpy array, shape (n_samples,)."""
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return (X @ self.w + self.b).numpy()
