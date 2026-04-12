"""
Recurrent Neural Network (RNN) and LSTM
=========================================

Vanilla RNN:
    h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
    y_t = W_hy * h_t + b_y

LSTM (4-gate formulation):
    f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)   # forget gate
    i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)   # input gate
    g_t = tanh( W_g * [h_{t-1}, x_t] + b_g)   # cell gate
    o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)   # output gate
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)

Training: Backpropagation through time (BPTT).
"""
import numpy as np
from typing import List, Tuple


# ── NumPy ─────────────────────────────────────────────────────────────────────

class VanillaRNNNumPy:
    """
    Many-to-one vanilla RNN for sequence classification.
    Uses the last hidden state for prediction.

    Parameters
    ----------
    input_size  : x_t dimension
    hidden_size : h_t dimension
    output_size : number of classes
    lr          : learning rate
    n_iters     : training iterations
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 lr: float = 0.01, n_iters: int = 100):
        self.hidden_size = hidden_size
        self.lr = lr
        self.n_iters = n_iters
        scale = 0.01
        self.W_xh = np.random.randn(input_size, hidden_size) * scale
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.b_h  = np.zeros(hidden_size)
        self.W_hy = np.random.randn(hidden_size, output_size) * scale
        self.b_y  = np.zeros(output_size)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _forward(self, X: np.ndarray):
        """
        X : (n, T, input_size)
        Returns: logits (n, output_size), cache for backprop
        """
        n, T, _ = X.shape
        h = np.zeros((n, self.hidden_size))
        hs = [h]
        for t in range(T):
            h = np.tanh(X[:, t, :] @ self.W_xh + h @ self.W_hh + self.b_h)
            hs.append(h)
        logits = hs[-1] @ self.W_hy + self.b_y   # (n, output_size)
        return logits, hs, X

    def _backward(self, logits, hs, X, y):
        n, T, _ = X.shape
        probs = self._softmax(logits)

        # Output layer gradient
        y_oh = np.zeros_like(probs)
        y_oh[np.arange(n), y] = 1
        d_logits = (probs - y_oh) / n             # (n, K)

        dW_hy = hs[-1].T @ d_logits
        db_y  = d_logits.sum(axis=0)
        dh    = d_logits @ self.W_hy.T             # (n, H)

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h  = np.zeros_like(self.b_h)

        for t in reversed(range(T)):
            dtanh = dh * (1 - hs[t + 1] ** 2)     # tanh' = 1 - tanh^2
            dW_xh += X[:, t, :].T @ dtanh
            dW_hh += hs[t].T @ dtanh
            db_h  += dtanh.sum(axis=0)
            dh     = dtanh @ self.W_hh.T           # BPTT

        # Clip to prevent exploding gradients
        for g in [dW_xh, dW_hh, db_h, dW_hy, db_y]:
            np.clip(g, -5, 5, out=g)

        return dW_xh, dW_hh, db_h, dW_hy, db_y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VanillaRNNNumPy":
        """
        X : (n_samples, T, input_size)
        y : (n_samples,)  integer class labels
        """
        for _ in range(self.n_iters):
            logits, hs, X_cache = self._forward(X)
            dW_xh, dW_hh, db_h, dW_hy, db_y = self._backward(logits, hs, X_cache, y)
            self.W_xh -= self.lr * dW_xh
            self.W_hh -= self.lr * dW_hh
            self.b_h  -= self.lr * db_h
            self.W_hy -= self.lr * dW_hy
            self.b_y  -= self.lr * db_y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits, _, _ = self._forward(X)
        return logits.argmax(axis=1)


class LSTMNumPy:
    """
    Many-to-one LSTM for sequence classification.

    Parameters
    ----------
    input_size  : x_t dimension
    hidden_size : h_t / c_t dimension
    output_size : number of classes
    lr          : learning rate
    n_iters     : training iterations
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 lr: float = 0.01, n_iters: int = 100):
        self.hidden_size = hidden_size
        self.lr = lr
        self.n_iters = n_iters
        D = input_size + hidden_size
        scale = 0.01
        # Combined weight matrix for all 4 gates: (D, 4*H)
        self.W   = np.random.randn(D, 4 * hidden_size) * scale
        self.b   = np.zeros(4 * hidden_size)
        self.W_y = np.random.randn(hidden_size, output_size) * scale
        self.b_y = np.zeros(output_size)

    @staticmethod
    def _sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _lstm_step(self, x_t, h_prev, c_prev):
        n, H = h_prev.shape
        xh = np.concatenate([x_t, h_prev], axis=1)   # (n, D)
        gates = xh @ self.W + self.b                  # (n, 4H)
        f = self._sigmoid(gates[:, :H])
        i = self._sigmoid(gates[:, H:2*H])
        g = np.tanh(gates[:, 2*H:3*H])
        o = self._sigmoid(gates[:, 3*H:])
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        cache = (xh, f, i, g, o, c, c_prev, h_prev)
        return h, c, cache

    def _forward(self, X):
        n, T, _ = X.shape
        H = self.hidden_size
        h = np.zeros((n, H))
        c = np.zeros((n, H))
        caches = []
        for t in range(T):
            h, c, cache = self._lstm_step(X[:, t, :], h, c)
            caches.append(cache)
        logits = h @ self.W_y + self.b_y
        return logits, caches, h

    def _backward(self, logits, caches, h_last, y):
        n = len(y)
        H = self.hidden_size
        T = len(caches)
        probs = self._softmax(logits)
        y_oh = np.zeros_like(probs)
        y_oh[np.arange(n), y] = 1
        d_logits = (probs - y_oh) / n

        dW_y = h_last.T @ d_logits
        db_y = d_logits.sum(axis=0)
        dh = d_logits @ self.W_y.T

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dc = np.zeros((n, H))

        for t in reversed(range(T)):
            xh, f, i, g, o, c, c_prev, h_prev = caches[t]
            tc = np.tanh(c)
            do = dh * tc
            dc += dh * o * (1 - tc ** 2)
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc = dc * f

            d_gates = np.concatenate([
                df * f * (1 - f),   # sigmoid'
                di * i * (1 - i),
                dg * (1 - g ** 2),  # tanh'
                do * o * (1 - o),
            ], axis=1)              # (n, 4H)

            dW += xh.T @ d_gates
            db += d_gates.sum(axis=0)
            dxh = d_gates @ self.W.T
            dh = dxh[:, -H:]       # gradient to previous h

        for g in [dW, db, dW_y, db_y]:
            np.clip(g, -5, 5, out=g)

        return dW, db, dW_y, db_y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMNumPy":
        """
        X : (n_samples, T, input_size)
        y : (n_samples,)
        """
        for _ in range(self.n_iters):
            logits, caches, h_last = self._forward(X)
            dW, db, dW_y, db_y = self._backward(logits, caches, h_last, y)
            self.W   -= self.lr * dW
            self.b   -= self.lr * db
            self.W_y -= self.lr * dW_y
            self.b_y -= self.lr * db_y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits, _, _ = self._forward(X)
        return logits.argmax(axis=1)


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class VanillaRNNTorch:
    """
    Vanilla RNN with raw PyTorch tensors + autograd.

    Parameters
    ----------
    input_size  : x_t dimension
    hidden_size : h_t dimension
    output_size : number of classes
    lr          : learning rate
    n_iters     : training iterations
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 lr: float = 0.01, n_iters: int = 100):
        self.hidden_size = hidden_size
        self.lr = lr
        self.n_iters = n_iters
        scale = 0.01
        self.W_xh = (torch.randn(input_size,  hidden_size) * scale).requires_grad_(True)
        self.W_hh = (torch.randn(hidden_size, hidden_size) * scale).requires_grad_(True)
        self.b_h  = torch.zeros(hidden_size, requires_grad=True)
        self.W_hy = (torch.randn(hidden_size, output_size) * scale).requires_grad_(True)
        self.b_y  = torch.zeros(output_size, requires_grad=True)

    def _params(self):
        return [self.W_xh, self.W_hh, self.b_h, self.W_hy, self.b_y]

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        n, T, _ = X.shape
        h = torch.zeros(n, self.hidden_size)
        for t in range(T):
            h = torch.tanh(X[:, t, :] @ self.W_xh + h @ self.W_hh + self.b_h)
        return h @ self.W_hy + self.b_y   # (n, output_size)

    def fit(self, X, y) -> "VanillaRNNTorch":
        """
        X : array-like (n, T, input_size)
        y : array-like (n,)
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        for _ in range(self.n_iters):
            logits = self._forward(X_t)
            log_sum_exp = torch.logsumexp(logits, dim=1)
            loss = (-logits[torch.arange(len(y_t)), y_t] + log_sum_exp).mean()

            loss.backward()
            with torch.no_grad():
                for p in self._params():
                    p -= self.lr * p.grad
                    p.grad.zero_()

        return self

    def predict(self, X) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self._forward(X_t).argmax(dim=1).numpy()


class LSTMTorch:
    """
    LSTM with raw PyTorch tensors + autograd.

    Parameters
    ----------
    input_size  : x_t dimension
    hidden_size : h_t / c_t dimension
    output_size : number of classes
    lr          : learning rate
    n_iters     : training iterations
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 lr: float = 0.01, n_iters: int = 100):
        self.hidden_size = hidden_size
        self.lr = lr
        self.n_iters = n_iters
        D = input_size + hidden_size
        H = hidden_size
        scale = 0.01
        # Single weight matrix for all 4 gates
        self.W   = (torch.randn(D, 4 * H) * scale).requires_grad_(True)
        self.b   = torch.zeros(4 * H, requires_grad=True)
        self.W_y = (torch.randn(H, output_size) * scale).requires_grad_(True)
        self.b_y = torch.zeros(output_size, requires_grad=True)

    def _params(self):
        return [self.W, self.b, self.W_y, self.b_y]

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        n, T, _ = X.shape
        H = self.hidden_size
        h = torch.zeros(n, H)
        c = torch.zeros(n, H)
        for t in range(T):
            xh = torch.cat([X[:, t, :], h], dim=1)   # (n, D)
            gates = xh @ self.W + self.b               # (n, 4H)
            f = torch.sigmoid(gates[:, :H])
            i = torch.sigmoid(gates[:, H:2*H])
            g = torch.tanh(gates[:, 2*H:3*H])
            o = torch.sigmoid(gates[:, 3*H:])
            c = f * c + i * g
            h = o * torch.tanh(c)
        return h @ self.W_y + self.b_y

    def fit(self, X, y) -> "LSTMTorch":
        """
        X : array-like (n, T, input_size)
        y : array-like (n,)
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        for _ in range(self.n_iters):
            logits = self._forward(X_t)
            log_sum_exp = torch.logsumexp(logits, dim=1)
            loss = (-logits[torch.arange(len(y_t)), y_t] + log_sum_exp).mean()

            loss.backward()
            with torch.no_grad():
                for p in self._params():
                    p -= self.lr * p.grad
                    p.grad.zero_()

        return self

    def predict(self, X) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self._forward(X_t).argmax(dim=1).numpy()
