"""
Gaussian Naive Bayes
====================
Assumes feature j in class c is Gaussian:  p(x_j | y=c) = N(mu_{cj}, sigma_{cj}^2)

Prediction:
    log P(y=c | x) ∝ log P(y=c) + sum_j log N(x_j; mu_{cj}, sigma_{cj}^2)

Training: estimate class priors and per-class per-feature mean / variance.
"""
import numpy as np


# ── NumPy ─────────────────────────────────────────────────────────────────────

class GaussianNBNumPy:
    """
    Parameters
    ----------
    var_smoothing : small constant added to variances for numerical stability
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray = None
        self.log_priors_: np.ndarray = None   # (K,)
        self.means_: np.ndarray = None         # (K, d)
        self.vars_: np.ndarray = None          # (K, d)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNBNumPy":
        """
        X : (n_samples, n_features)
        y : (n_samples,)  integer or string class labels
        """
        self.classes_ = np.unique(y)
        K, d = len(self.classes_), X.shape[1]
        self.log_priors_ = np.zeros(K)
        self.means_ = np.zeros((K, d))
        self.vars_  = np.zeros((K, d))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.log_priors_[i] = np.log(len(X_c) / len(X))
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i]  = X_c.var(axis=0) + self.var_smoothing

        return self

    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Returns log P(x | y=c) for each class, shape (n, K)."""
        log_probs = []
        for i in range(len(self.classes_)):
            mu, var = self.means_[i], self.vars_[i]
            # log N(x; mu, var) = -0.5 * sum(log(2π*var) + (x-mu)^2/var)
            log_p = -0.5 * np.sum(np.log(2 * np.pi * var) + (X - mu) ** 2 / var, axis=1)
            log_probs.append(log_p)
        return np.stack(log_probs, axis=1)   # (n, K)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns unnormalized log posteriors, shape (n, K)."""
        return self._log_likelihood(X) + self.log_priors_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class labels, shape (n,)."""
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class GaussianNBTorch:
    """
    Gaussian Naive Bayes with raw PyTorch tensors.
    (Analytical fit — no gradient updates needed.)

    Parameters
    ----------
    var_smoothing : small constant added to variances
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray = None
        self.log_priors_: torch.Tensor = None
        self.means_: torch.Tensor = None
        self.vars_: torch.Tensor = None

    def fit(self, X, y) -> "GaussianNBTorch":
        """
        X : array-like (n_samples, n_features)
        y : array-like (n_samples,)
        """
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        X = torch.tensor(X, dtype=torch.float32)
        n, d = X.shape

        log_priors = torch.zeros(K)
        means = torch.zeros(K, d)
        vars_ = torch.zeros(K, d)

        for i, c in enumerate(self.classes_):
            mask = torch.tensor(y == c)
            X_c = X[mask]
            log_priors[i] = torch.log(torch.tensor(len(X_c) / n))
            means[i] = X_c.mean(dim=0)
            vars_[i] = X_c.var(dim=0, unbiased=False) + self.var_smoothing

        self.log_priors_ = log_priors
        self.means_ = means
        self.vars_ = vars_
        return self

    def predict(self, X) -> np.ndarray:
        """Returns predicted class labels, shape (n,)."""
        X = torch.tensor(X, dtype=torch.float32)
        log_posts = []
        for i in range(len(self.classes_)):
            mu, var = self.means_[i], self.vars_[i]
            log_p = -0.5 * torch.sum(torch.log(2 * torch.pi * var) + (X - mu) ** 2 / var, dim=1)
            log_posts.append(log_p + self.log_priors_[i])
        log_posts = torch.stack(log_posts, dim=1)   # (n, K)
        idx = torch.argmax(log_posts, dim=1).numpy()
        return self.classes_[idx]
