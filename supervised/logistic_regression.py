"""
Logistic Regression
===================
Binary:     P(y=1|x) = sigma(w^T x + b),  sigma(z) = 1/(1+e^{-z})
Multiclass: P(y=k|x) = softmax(Wx + b)[k]

Loss: Binary cross-entropy  L = -(1/n) sum(y log p + (1-y) log(1-p))
Gradient (binary): dL/dw = (1/n) X^T (p - y),  dL/db = (1/n) sum(p - y)
"""
import numpy as np


# ── NumPy ─────────────────────────────────────────────────────────────────────

class LogisticRegressionNumPy:
    """
    Supports binary classification.

    Parameters
    ----------
    lr      : learning rate
    n_iters : gradient descent steps
    """

    def __init__(self, lr: float = 0.01, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w: np.ndarray = None
        self.b: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionNumPy":
        """
        X : (n_samples, n_features)
        y : (n_samples,)  values in {0, 1}
        """
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.n_iters):
            p = self._sigmoid(X @ self.w + self.b)   # (n,)
            error = p - y                             # (n,)
            self.w -= self.lr * (1 / n) * X.T @ error
            self.b -= self.lr * (1 / n) * error.sum()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns P(y=1|x), shape (n_samples,)."""
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns class labels {0, 1}, shape (n_samples,)."""
        return (self.predict_proba(X) >= threshold).astype(int)


class SoftmaxRegressionNumPy:
    """
    Multinomial logistic regression (softmax) for K-class problems.

    Parameters
    ----------
    lr      : learning rate
    n_iters : gradient descent steps
    """

    def __init__(self, lr: float = 0.01, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.W: np.ndarray = None   # (n_features, n_classes)
        self.b: np.ndarray = None   # (n_classes,)
        self.classes_: np.ndarray = None

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        # Numerically stable: subtract row-wise max
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxRegressionNumPy":
        """
        X : (n_samples, n_features)
        y : (n_samples,)  integer class labels
        """
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n, d = X.shape
        # One-hot encode
        Y = np.zeros((n, K))
        for i, c in enumerate(self.classes_):
            Y[y == c, i] = 1

        self.W = np.zeros((d, K))
        self.b = np.zeros(K)

        for _ in range(self.n_iters):
            P = self._softmax(X @ self.W + self.b)   # (n, K)
            dZ = P - Y                                # (n, K)
            self.W -= self.lr * (1 / n) * X.T @ dZ
            self.b -= self.lr * (1 / n) * dZ.sum(axis=0)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_samples, n_classes) probability matrix."""
        return self._softmax(X @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class labels, shape (n_samples,)."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class LogisticRegressionTorch:
    """
    Binary logistic regression with raw PyTorch tensors and manual gradients.
    No autograd — gradients derived analytically from BCE loss.

    Forward:  p = sigmoid(X @ w + b)
    Loss:     L = -(1/n) * sum(y*log(p) + (1-y)*log(1-p))
    Gradients (dL/dw = dL/dp * dp/dz * dz/dw, simplifies to):
        dL/dw = (1/n) * X^T @ (p - y)
        dL/db = (1/n) * sum(p - y)

    Parameters
    ----------
    lr      : learning rate
    n_iters : gradient descent steps
    """

    def __init__(self, lr: float = 0.01, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w: torch.Tensor = None
        self.b: torch.Tensor = None

    @staticmethod
    def _sigmoid(z: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-torch.clamp(z, -500, 500)))

    def fit(self, X, y) -> "LogisticRegressionTorch":
        """
        X : array-like (n_samples, n_features)
        y : array-like (n_samples,)  values in {0, 1}
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        n, d = X.shape

        self.w = torch.zeros(d)
        self.b = torch.zeros(1)

        for _ in range(self.n_iters):
            p     = self._sigmoid(X @ self.w + self.b)  # (n,)
            error = p - y                                # (n,)  = dL/d(logit)

            dw = (1 / n) * X.T @ error                  # (d,)
            db = (1 / n) * error.sum()                  # scalar

            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Returns P(y=1|x) as numpy array, shape (n_samples,)."""
        X = torch.tensor(X, dtype=torch.float32)
        return self._sigmoid(X @ self.w + self.b).numpy()

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Returns class labels {0, 1}, shape (n_samples,)."""
        return (self.predict_proba(X) >= threshold).astype(int)


class SoftmaxRegressionTorch:
    """
    Multinomial logistic regression with raw PyTorch tensors and manual gradients.
    No autograd — gradients derived analytically from softmax + cross-entropy.

    Forward:  P = softmax(X @ W + b)             (n, K)
    Loss:     L = -(1/n) * sum(Y * log(P))
    Gradients (combined softmax + CE derivative):
        dL/dZ = (1/n) * (P - Y)                  (n, K)
        dL/dW = X^T @ dL/dZ                      (d, K)
        dL/db = sum(dL/dZ, axis=0)               (K,)

    Parameters
    ----------
    lr      : learning rate
    n_iters : gradient descent steps
    """

    def __init__(self, lr: float = 0.01, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.W: torch.Tensor = None
        self.b: torch.Tensor = None
        self.classes_: np.ndarray = None

    @staticmethod
    def _softmax(z: torch.Tensor) -> torch.Tensor:
        z = z - z.max(dim=1, keepdim=True).values   # numerically stable
        exp_z = torch.exp(z)
        return exp_z / exp_z.sum(dim=1, keepdim=True)

    def fit(self, X, y) -> "SoftmaxRegressionTorch":
        """
        X : array-like (n_samples, n_features)
        y : array-like (n_samples,)  integer class labels
        """
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        X = torch.tensor(X, dtype=torch.float32)
        n, d = X.shape

        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_idx = torch.tensor([label_map[yi] for yi in y], dtype=torch.long)
        Y = torch.zeros(n, K)
        Y.scatter_(1, y_idx.unsqueeze(1), 1.0)      # one-hot (n, K)

        self.W = torch.zeros(d, K)
        self.b = torch.zeros(K)

        for _ in range(self.n_iters):
            P  = self._softmax(X @ self.W + self.b) # (n, K)
            dZ = (1 / n) * (P - Y)                  # (n, K)  combined CE+softmax grad

            dW = X.T @ dZ                            # (d, K)
            db = dZ.sum(dim=0)                       # (K,)

            self.W -= self.lr * dW
            self.b -= self.lr * db

        return self

    def predict(self, X) -> np.ndarray:
        """Returns predicted class labels, shape (n_samples,)."""
        X = torch.tensor(X, dtype=torch.float32)
        logits = X @ self.W + self.b
        idx = torch.argmax(logits, dim=1).numpy()
        return self.classes_[idx]
