"""
Linear Regression
=================
Minimize ||Xw + b - y||^2 via gradient descent or the normal equation.

Normal equation (closed-form): w = (X^T X)^{-1} X^T y
Gradient:  dL/dw = (2/n) X^T (Xw - y),  dL/db = (2/n) sum(Xw - y)
"""
import numpy as np


# ── NumPy ─────────────────────────────────────────────────────────────────────

class LinearRegressionNumPy:
    """
    Parameters
    ----------
    method : 'gd' (gradient descent) | 'normal' (closed-form)
    lr     : learning rate (used only when method='gd')
    n_iters: number of gradient descent steps
    """

    def __init__(self, method: str = "normal", lr: float = 0.01, n_iters: int = 1000):
        self.method = method
        self.lr = lr
        self.n_iters = n_iters
        self.w: np.ndarray = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionNumPy":
        """
        X : (n_samples, n_features)
        y : (n_samples,)
        """
        n, d = X.shape
        if self.method == "normal":
            # Augment X with bias column
            X_b = np.c_[np.ones(n), X]                  # (n, d+1)
            params = np.linalg.lstsq(X_b, y, rcond=None)[0]
            self.b, self.w = params[0], params[1:]
        else:  # gradient descent
            self.w = np.zeros(d)
            self.b = 0.0
            for _ in range(self.n_iters):
                y_hat = X @ self.w + self.b
                error = y_hat - y                        # (n,)
                self.w -= self.lr * (2 / n) * X.T @ error
                self.b -= self.lr * (2 / n) * error.sum()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted values, shape (n_samples,)."""
        return X @ self.w + self.b


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class LinearRegressionTorch:
    """
    Linear regression with raw PyTorch tensors and manual autograd.
    Uses gradient descent; no nn.Module.

    Parameters
    ----------
    lr      : learning rate
    n_iters : number of gradient steps
    """

    def __init__(self, lr: float = 0.01, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w: torch.Tensor = None
        self.b: torch.Tensor = None

    def fit(self, X, y) -> "LinearRegressionTorch":
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
            y_hat = X @ self.w + self.b            # (n,)
            loss = ((y_hat - y) ** 2).mean()

            loss.backward()
            with torch.no_grad():
                self.w -= self.lr * self.w.grad
                self.b -= self.lr * self.b.grad
            self.w.grad.zero_()
            self.b.grad.zero_()

        return self

    def predict(self, X) -> np.ndarray:
        """Returns numpy array of predictions, shape (n_samples,)."""
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return (X @ self.w + self.b).numpy()
