"""
Gaussian Mixture Model (GMM) — EM Algorithm
=============================================
Models data as a mixture of K Gaussians:
    p(x) = sum_k pi_k * N(x; mu_k, Sigma_k)

EM Algorithm:
  E-step: Compute soft assignments (responsibilities)
          r_{ik} = pi_k * N(x_i; mu_k, Sigma_k) / sum_j pi_j * N(x_i; mu_j, Sigma_j)

  M-step: Update parameters using responsibilities
          N_k   = sum_i r_{ik}
          pi_k  = N_k / n
          mu_k  = (1/N_k) * sum_i r_{ik} * x_i
          Sig_k = (1/N_k) * sum_i r_{ik} * (x_i - mu_k)(x_i - mu_k)^T
"""
import numpy as np


# ── NumPy ─────────────────────────────────────────────────────────────────────

class GMMNumPy:
    """
    Parameters
    ----------
    k       : number of Gaussian components
    n_iters : EM iterations
    tol     : convergence tolerance on log-likelihood change
    """

    def __init__(self, k: int = 3, n_iters: int = 100, tol: float = 1e-4):
        self.k = k
        self.n_iters = n_iters
        self.tol = tol
        self.pi_: np.ndarray = None    # (k,)  mixing coefficients
        self.mu_: np.ndarray = None    # (k, d)
        self.sigma_: np.ndarray = None # (k, d, d)

    @staticmethod
    def _gaussian_pdf(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Multivariate Gaussian density for each row of X.
        X : (n, d), mu : (d,), sigma : (d, d)
        Returns : (n,)
        """
        d = X.shape[1]
        diff = X - mu                               # (n, d)
        sign, log_det = np.linalg.slogdet(sigma)
        if sign <= 0:
            # Degenerate covariance — add small jitter
            sigma = sigma + 1e-6 * np.eye(d)
            sign, log_det = np.linalg.slogdet(sigma)
        inv_sigma = np.linalg.inv(sigma)
        maha = np.einsum("nd,dd,nd->n", diff, inv_sigma, diff)  # (n,)
        log_p = -0.5 * (d * np.log(2 * np.pi) + log_det + maha)
        return np.exp(log_p)

    def fit(self, X: np.ndarray) -> "GMMNumPy":
        """
        X : (n_samples, n_features)
        """
        n, d = X.shape
        # Initialize: random assignment
        idx = np.random.choice(n, size=self.k, replace=False)
        self.mu_    = X[idx].copy()                    # (k, d)
        self.sigma_ = np.array([np.eye(d)] * self.k)  # (k, d, d)
        self.pi_    = np.full(self.k, 1.0 / self.k)   # (k,)
        prev_ll = -np.inf

        for _ in range(self.n_iters):
            # ── E-step ────────────────────────────────────────────────────
            R = np.zeros((n, self.k))
            for j in range(self.k):
                R[:, j] = self.pi_[j] * self._gaussian_pdf(X, self.mu_[j], self.sigma_[j])
            R_sum = R.sum(axis=1, keepdims=True) + 1e-300
            R /= R_sum                                 # (n, k) responsibilities

            # Log-likelihood for convergence check
            ll = float(np.log(R_sum).sum())
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

            # ── M-step ────────────────────────────────────────────────────
            N_k = R.sum(axis=0)                        # (k,)
            self.pi_ = N_k / n
            self.mu_ = (R.T @ X) / N_k[:, None]       # (k, d)
            for j in range(self.k):
                diff = X - self.mu_[j]                 # (n, d)
                self.sigma_[j] = (R[:, j, None] * diff).T @ diff / N_k[j]
                self.sigma_[j] += 1e-6 * np.eye(d)    # numerical stability

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns hard cluster assignments, shape (n,)."""
        R = np.zeros((len(X), self.k))
        for j in range(self.k):
            R[:, j] = self.pi_[j] * self._gaussian_pdf(X, self.mu_[j], self.sigma_[j])
        return np.argmax(R, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns soft responsibilities, shape (n, k)."""
        R = np.zeros((len(X), self.k))
        for j in range(self.k):
            R[:, j] = self.pi_[j] * self._gaussian_pdf(X, self.mu_[j], self.sigma_[j])
        R /= R.sum(axis=1, keepdims=True) + 1e-300
        return R


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class GMMTorch:
    """
    GMM with raw PyTorch tensors (diagonal covariance for simplicity).

    Parameters
    ----------
    k       : number of components
    n_iters : EM iterations
    tol     : log-likelihood convergence tolerance
    """

    def __init__(self, k: int = 3, n_iters: int = 100, tol: float = 1e-4):
        self.k = k
        self.n_iters = n_iters
        self.tol = tol
        self.pi_: torch.Tensor = None
        self.mu_: torch.Tensor = None
        self.var_: torch.Tensor = None   # diagonal variances (k, d)

    @staticmethod
    def _log_gaussian_diag(X: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        Log-density under diagonal Gaussian.
        X : (n, d), mu : (d,), var : (d,)
        Returns : (n,)
        """
        d = X.shape[1]
        log_p = -0.5 * (
            d * torch.log(torch.tensor(2 * torch.pi))
            + torch.log(var).sum()
            + ((X - mu) ** 2 / var).sum(dim=1)
        )
        return log_p

    def fit(self, X) -> "GMMTorch":
        X = torch.tensor(X, dtype=torch.float32)
        n, d = X.shape
        idx = torch.randperm(n)[:self.k]
        self.mu_  = X[idx].clone()                    # (k, d)
        self.var_ = torch.ones(self.k, d)             # (k, d)
        self.pi_  = torch.full((self.k,), 1.0 / self.k)
        prev_ll = -float("inf")

        for _ in range(self.n_iters):
            # ── E-step ────────────────────────────────────────────────────
            log_R = torch.zeros(n, self.k)
            for j in range(self.k):
                log_R[:, j] = (
                    torch.log(self.pi_[j])
                    + self._log_gaussian_diag(X, self.mu_[j], self.var_[j])
                )
            # Numerically stable log-sum-exp
            log_sum = torch.logsumexp(log_R, dim=1, keepdim=True)
            ll = float(log_sum.sum().item())
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
            R = torch.exp(log_R - log_sum)           # (n, k)

            # ── M-step ────────────────────────────────────────────────────
            N_k = R.sum(dim=0) + 1e-10               # (k,)
            self.pi_  = N_k / n
            self.mu_  = (R.T @ X) / N_k[:, None]    # (k, d)
            for j in range(self.k):
                diff = X - self.mu_[j]
                self.var_[j] = (R[:, j, None] * diff ** 2).sum(dim=0) / N_k[j] + 1e-6

        return self

    def predict(self, X) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32)
        log_R = torch.zeros(len(X), self.k)
        for j in range(self.k):
            log_R[:, j] = (
                torch.log(self.pi_[j])
                + self._log_gaussian_diag(X, self.mu_[j], self.var_[j])
            )
        return torch.argmax(log_R, dim=1).numpy()
