"""
Random Forest
=============
Ensemble of decision trees, each trained on a bootstrap sample with a random
subset of features considered at each split (bagging + feature randomization).

Classification: majority vote across all trees.
Regression:     mean prediction across all trees.

Key hyperparameters:
  n_trees       : number of trees
  max_depth     : depth limit for each tree
  max_features  : features per split ('sqrt' for classification, 'third' for regression)
"""
import numpy as np
from collections import Counter
from .decision_tree import DecisionTreeNumPy


# ── NumPy ─────────────────────────────────────────────────────────────────────

class RandomForestNumPy:
    """
    Parameters
    ----------
    n_trees      : number of trees
    task         : 'classification' | 'regression'
    max_depth    : max depth per tree (None = unlimited)
    max_features : features per split; int or 'sqrt' or 'log2' or None (all)
    min_samples_split : minimum samples to split a node
    """

    def __init__(self, n_trees: int = 100, task: str = "classification",
                 max_depth: int = None, max_features="sqrt",
                 min_samples_split: int = 2):
        self.n_trees = n_trees
        self.task = task
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.trees: list = []

    def _resolve_max_features(self, d: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(d)))
        if self.max_features == "log2":
            return max(1, int(np.log2(d)))
        if self.max_features is None:
            return d
        return int(self.max_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestNumPy":
        """
        X : (n_samples, n_features)
        y : (n_samples,)
        """
        n, d = X.shape
        n_feat = self._resolve_max_features(d)
        self.trees = []

        for _ in range(self.n_trees):
            # Bootstrap sample
            idx = np.random.choice(n, size=n, replace=True)
            X_boot, y_boot = X[idx], y[idx]

            tree = DecisionTreeNumPy(
                task=self.task,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=n_feat,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predictions, shape (n_samples,)."""
        # Matrix of shape (n_trees, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees])

        if self.task == "classification":
            # Majority vote per sample
            return np.array([
                Counter(all_preds[:, i]).most_common(1)[0][0]
                for i in range(X.shape[0])
            ])
        return all_preds.mean(axis=0)


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
from .decision_tree import DecisionTreeTorch


class RandomForestTorch:
    """
    Random forest using PyTorch tensor-based decision trees.

    Parameters
    ----------
    n_trees   : number of trees
    task      : 'classification' | 'regression'
    max_depth : max depth per tree
    """

    def __init__(self, n_trees: int = 100, task: str = "classification",
                 max_depth: int = None, min_samples_split: int = 2):
        self.n_trees = n_trees
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees: list = []

    def fit(self, X, y) -> "RandomForestTorch":
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y)
        n = len(y)
        self.trees = []

        for _ in range(self.n_trees):
            idx = torch.randint(0, n, (n,))
            X_boot, y_boot = X[idx], y[idx]
            tree = DecisionTreeTorch(
                task=self.task,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            # DecisionTreeTorch.fit accepts tensors directly
            tree.root = tree._build(X_boot, y_boot, depth=0)
            self.trees.append(tree)

        return self

    def predict(self, X) -> np.ndarray:
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        if self.task == "classification":
            return np.array([
                Counter(all_preds[:, i]).most_common(1)[0][0]
                for i in range(all_preds.shape[1])
            ])
        return all_preds.mean(axis=0)
