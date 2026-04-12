"""
Decision Tree
=============
Recursively partitions the feature space by choosing splits that maximize
information gain (classification) or reduce MSE (regression).

Split criterion:
  Classification: Information Gain = H(parent) - weighted_avg H(children)
                  where H(S) = -sum_c p_c * log2(p_c)  (entropy)
  Regression:     Variance Reduction = Var(parent) - weighted_avg Var(children)
"""
import numpy as np
from typing import Optional


# ── NumPy ─────────────────────────────────────────────────────────────────────

class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value   # set only for leaf nodes

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeNumPy:
    """
    Decision tree for classification or regression.

    Parameters
    ----------
    task      : 'classification' | 'regression'
    max_depth : maximum tree depth (None = unlimited)
    min_samples_split : minimum samples required to split a node
    n_features : number of features to consider per split (None = all)
    """

    def __init__(self, task: str = "classification", max_depth: Optional[int] = None,
                 min_samples_split: int = 2, n_features: Optional[int] = None):
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root: Optional[_Node] = None

    # ── impurity helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        counts = np.bincount(y.astype(int))
        probs = counts[counts > 0] / len(y)
        return -float(np.sum(probs * np.log2(probs)))

    @staticmethod
    def _variance(y: np.ndarray) -> float:
        return float(np.var(y))

    def _impurity(self, y: np.ndarray) -> float:
        return self._entropy(y) if self.task == "classification" else self._variance(y)

    def _leaf_value(self, y: np.ndarray):
        if self.task == "classification":
            return int(np.bincount(y.astype(int)).argmax())
        return float(np.mean(y))

    # ── best split ────────────────────────────────────────────────────────────

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        n_features = self.n_features or d
        feature_ids = np.random.choice(d, size=n_features, replace=False)

        best_gain, best_feat, best_thresh = -np.inf, None, None
        parent_imp = self._impurity(y)

        for feat in feature_ids:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left  = y[X[:, feat] <= thresh]
                right = y[X[:, feat] >  thresh]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = parent_imp - (
                    len(left)  / n * self._impurity(left) +
                    len(right) / n * self._impurity(right)
                )
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        return best_feat, best_thresh

    # ── tree building ─────────────────────────────────────────────────────────

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        # Stopping conditions
        if (len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)
                or len(np.unique(y)) == 1):
            return _Node(value=self._leaf_value(y))

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return _Node(value=self._leaf_value(y))

        mask = X[:, feat] <= thresh
        left  = self._build(X[mask],  y[mask],  depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return _Node(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeNumPy":
        """
        X : (n_samples, n_features)
        y : (n_samples,)
        """
        self.root = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x: np.ndarray, node: _Node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predictions, shape (n_samples,)."""
        return np.array([self._predict_one(x, self.root) for x in X])


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class DecisionTreeTorch:
    """
    Decision tree using PyTorch tensors for data storage and comparisons.
    Tree logic is identical to the NumPy version; PyTorch is used for
    tensor-based data handling (no autograd — trees are non-differentiable).

    Parameters
    ----------
    task      : 'classification' | 'regression'
    max_depth : maximum tree depth
    min_samples_split : minimum samples to split
    """

    def __init__(self, task: str = "classification", max_depth: Optional[int] = None,
                 min_samples_split: int = 2):
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[_Node] = None

    @staticmethod
    def _entropy(y: torch.Tensor) -> float:
        _, counts = torch.unique(y, return_counts=True)
        probs = counts.float() / len(y)
        return -float((probs * torch.log2(probs + 1e-12)).sum())

    @staticmethod
    def _variance(y: torch.Tensor) -> float:
        return float(y.float().var())

    def _impurity(self, y: torch.Tensor) -> float:
        return self._entropy(y) if self.task == "classification" else self._variance(y)

    def _leaf_value(self, y: torch.Tensor):
        if self.task == "classification":
            vals, counts = torch.unique(y, return_counts=True)
            return int(vals[counts.argmax()].item())
        return float(y.float().mean().item())

    def _best_split(self, X: torch.Tensor, y: torch.Tensor):
        n, d = X.shape
        best_gain, best_feat, best_thresh = -float("inf"), None, None
        parent_imp = self._impurity(y)

        for feat in range(d):
            thresholds = torch.unique(X[:, feat])
            for thresh in thresholds:
                mask = X[:, feat] <= thresh
                left, right = y[mask], y[~mask]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = parent_imp - (
                    len(left)  / n * self._impurity(left) +
                    len(right) / n * self._impurity(right)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh.item()

        return best_feat, best_thresh

    def _build(self, X: torch.Tensor, y: torch.Tensor, depth: int) -> _Node:
        if (len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)
                or len(torch.unique(y)) == 1):
            return _Node(value=self._leaf_value(y))

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return _Node(value=self._leaf_value(y))

        mask = X[:, feat] <= thresh
        left  = self._build(X[mask],  y[mask],  depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return _Node(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X, y) -> "DecisionTreeTorch":
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y)
        self.root = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x: torch.Tensor, node: _Node):
        if node.is_leaf:
            return node.value
        if x[node.feature].item() <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32)
        return np.array([self._predict_one(x, self.root) for x in X])
