"""
Gradient Boosting
=================
Builds an ensemble by iteratively fitting shallow trees to the negative gradient
(pseudo-residuals) of the loss function.

Regression (MSE loss):
    F_0(x) = mean(y)
    r_m = y - F_{m-1}(x)          # negative gradient = residuals for MSE
    F_m(x) = F_{m-1}(x) + lr * h_m(x)   where h_m fits r_m

Classification (log-loss / logistic):
    F_0(x) = log(p/(1-p))  with p = mean(y)
    r_m = y - sigmoid(F_{m-1}(x))
    F_m(x) = F_{m-1}(x) + lr * h_m(x)
"""
import numpy as np
from .decision_tree import DecisionTreeNumPy


# ── NumPy ─────────────────────────────────────────────────────────────────────

class GradientBoostingRegressorNumPy:
    """
    Parameters
    ----------
    n_estimators : number of boosting rounds
    lr           : shrinkage / learning rate
    max_depth    : depth of each tree
    """

    def __init__(self, n_estimators: int = 100, lr: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.trees: list = []
        self.F0: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressorNumPy":
        """
        X : (n_samples, n_features)
        y : (n_samples,)
        """
        self.F0 = float(np.mean(y))
        F = np.full(len(y), self.F0)
        self.trees = []

        for _ in range(self.n_estimators):
            residuals = y - F
            tree = DecisionTreeNumPy(task="regression", max_depth=self.max_depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            F += self.lr * update
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = np.full(X.shape[0], self.F0)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return F


class GradientBoostingClassifierNumPy:
    """
    Binary gradient boosting classifier using log-loss.

    Parameters
    ----------
    n_estimators : boosting rounds
    lr           : learning rate
    max_depth    : depth per tree
    """

    def __init__(self, n_estimators: int = 100, lr: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.trees: list = []
        self.F0: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifierNumPy":
        """
        X : (n_samples, n_features)
        y : (n_samples,)  values in {0, 1}
        """
        p0 = np.clip(np.mean(y), 1e-12, 1 - 1e-12)
        self.F0 = float(np.log(p0 / (1 - p0)))
        F = np.full(len(y), self.F0)
        self.trees = []

        for _ in range(self.n_estimators):
            residuals = y - self._sigmoid(F)   # negative gradient of log-loss
            tree = DecisionTreeNumPy(task="regression", max_depth=self.max_depth)
            tree.fit(X, residuals)
            F += self.lr * tree.predict(X)
            self.trees.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns P(y=1|x), shape (n,)."""
        F = np.full(X.shape[0], self.F0)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return self._sigmoid(F)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
from .decision_tree import DecisionTreeTorch


class GradientBoostingRegressorTorch:
    """
    Gradient boosting regressor with PyTorch tensor operations.

    Parameters
    ----------
    n_estimators : boosting rounds
    lr           : learning rate
    max_depth    : depth per tree
    """

    def __init__(self, n_estimators: int = 100, lr: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.trees: list = []
        self.F0: float = 0.0

    def fit(self, X, y) -> "GradientBoostingRegressorTorch":
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        self.F0 = float(y_t.mean().item())
        F = torch.full((len(y_t),), self.F0)
        self.trees = []

        for _ in range(self.n_estimators):
            residuals = y_t - F
            tree = DecisionTreeTorch(task="regression", max_depth=self.max_depth)
            tree.root = tree._build(X_t, residuals, depth=0)
            update = torch.tensor(tree.predict(X), dtype=torch.float32)
            F = F + self.lr * update
            self.trees.append(tree)

        return self

    def predict(self, X) -> np.ndarray:
        F = np.full(len(X), self.F0)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return F
