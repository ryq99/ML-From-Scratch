"""
Support Vector Machine (SVM)
=============================
Soft-margin linear SVM via subgradient descent on the primal.

Objective:  min_{w,b}  (1/n) * sum_i max(0, 1 - y_i*(w·x_i + b)) + C * ||w||^2

where y_i in {-1, +1}.  This is the hinge loss + L2 regularization.

Subgradient of hinge loss w.r.t. w:
    if 1 - y_i*(w·x_i + b) > 0:  -y_i * x_i
    else:                          0
"""
import numpy as np


# ── NumPy ─────────────────────────────────────────────────────────────────────

class SVMNumPy:
    """
    Linear soft-margin SVM trained with subgradient descent.

    Parameters
    ----------
    C       : regularization parameter (larger C = less regularization)
    lr      : learning rate
    n_iters : number of gradient steps
    """

    def __init__(self, C: float = 1.0, lr: float = 0.001, n_iters: int = 1000):
        self.C = C
        self.lr = lr
        self.n_iters = n_iters
        self.w: np.ndarray = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMNumPy":
        """
        X : (n_samples, n_features)
        y : (n_samples,)  labels in {-1, +1}
        """
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.n_iters):
            margins = y * (X @ self.w + self.b)         # (n,)
            mask = margins < 1                           # support vectors
            # Gradient of hinge loss
            dw = self.w - self.C * (X[mask] * y[mask, None]).mean(axis=0) if mask.any() \
                 else self.w
            db = -self.C * y[mask].mean() if mask.any() else 0.0
            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw score w·x + b, shape (n,)."""
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted labels in {-1, +1}, shape (n,)."""
        return np.sign(self.decision_function(X)).astype(int)


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class SVMTorch:
    """
    Linear SVM with raw PyTorch tensors and manual subgradient computation.
    No autograd — subgradients derived analytically.

    Objective:  L = (1/n)*sum_i max(0, 1 - y_i*(w·x_i+b))  +  C*||w||^2
    Subgradients (for sample i, let m_i = y_i*(w·x_i+b)):
        if m_i < 1:  dL/dw += -y_i*x_i / n,  dL/db += -y_i / n
        else:        no contribution from hinge term
    L2 term always contributes:  dL/dw += 2*C*w  (not regularizing b)

    Parameters
    ----------
    C       : regularization parameter
    lr      : learning rate
    n_iters : gradient steps
    """

    def __init__(self, C: float = 1.0, lr: float = 0.001, n_iters: int = 1000):
        self.C = C
        self.lr = lr
        self.n_iters = n_iters
        self.w: torch.Tensor = None
        self.b: torch.Tensor = None

    def fit(self, X, y) -> "SVMTorch":
        """
        X : array-like (n_samples, n_features)
        y : array-like (n_samples,)  labels in {-1, +1}
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        n, d = X.shape

        self.w = torch.zeros(d)
        self.b = torch.zeros(1)

        for _ in range(self.n_iters):
            margins = y * (X @ self.w + self.b)    # (n,)
            mask = margins < 1                     # support vectors (bool mask)

            # Subgradient of hinge loss
            if mask.any():
                # (X[mask] * y[mask, None]): contributions of misclassified/margin samples
                dw = self.w - self.C * (X[mask] * y[mask, None]).mean(dim=0)
                db = -self.C * y[mask].mean()
            else:
                dw = self.w.clone()
                db = torch.zeros(1)

            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def predict(self, X) -> np.ndarray:
        """Returns predicted labels in {-1, +1}, shape (n,)."""
        X = torch.tensor(X, dtype=torch.float32)
        scores = X @ self.w + self.b
        return np.sign(scores.numpy()).astype(int)
