"""
Convolutional Neural Network (CNN)
====================================
A minimal CNN for image classification with:
  Conv2D → ReLU → MaxPool → Flatten → Dense

Forward pass:
  - Conv2D: for each filter k, output[k,i,j] = sum_{c,p,q} W[k,c,p,q] * X[c,i+p,j+q] + b[k]
  - MaxPool: take max over non-overlapping windows
  - Dense:  same as MLP

Backprop:
  - Dense: standard matrix gradient
  - MaxPool: gradient flows only to the max element (max-mask)
  - Conv2D: gradient w.r.t. input is 'full' convolution with flipped filter;
             gradient w.r.t. weights is convolution of input with output gradient
"""
import numpy as np


# ── NumPy ─────────────────────────────────────────────────────────────────────

class CNNNumPy:
    """
    Single conv layer + max pool + fully-connected layer.

    Architecture: [Conv(n_filters, ksize) → ReLU → MaxPool(pool)] → Flatten → Linear

    Parameters
    ----------
    n_filters    : number of conv filters
    kernel_size  : square filter side length
    pool_size    : max pooling window (square)
    n_classes    : output classes
    lr           : learning rate
    n_iters      : training iterations
    """

    def __init__(self, n_filters: int = 8, kernel_size: int = 3, pool_size: int = 2,
                 n_classes: int = 10, lr: float = 0.01, n_iters: int = 100):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.n_classes = n_classes
        self.lr = lr
        self.n_iters = n_iters
        # Params initialized in fit
        self.W_conv: np.ndarray = None   # (n_filters, C_in, ksize, ksize)
        self.b_conv: np.ndarray = None   # (n_filters,)
        self.W_fc:   np.ndarray = None
        self.b_fc:   np.ndarray = None

    # ── Conv helpers ──────────────────────────────────────────────────────────

    def _conv_forward(self, X: np.ndarray):
        """
        X : (n, C, H, W)
        Returns: (n, n_filters, H_out, W_out)
        """
        n, C, H, W = X.shape
        k = self.kernel_size
        H_out = H - k + 1
        W_out = W - k + 1
        out = np.zeros((n, self.n_filters, H_out, W_out))
        for f in range(self.n_filters):
            for i in range(H_out):
                for j in range(W_out):
                    patch = X[:, :, i:i+k, j:j+k]   # (n, C, k, k)
                    out[:, f, i, j] = (patch * self.W_conv[f]).sum(axis=(1, 2, 3)) + self.b_conv[f]
        return out

    def _conv_backward(self, dout: np.ndarray, X: np.ndarray):
        """
        dout : (n, n_filters, H_out, W_out)
        Returns dX, dW, db
        """
        n, C, H, W = X.shape
        k = self.kernel_size
        H_out, W_out = dout.shape[2], dout.shape[3]
        dX = np.zeros_like(X)
        dW = np.zeros_like(self.W_conv)
        db = dout.sum(axis=(0, 2, 3))

        for f in range(self.n_filters):
            for i in range(H_out):
                for j in range(W_out):
                    patch = X[:, :, i:i+k, j:j+k]                     # (n,C,k,k)
                    dW[f] += (dout[:, f, i, j][:, None, None, None] * patch).sum(axis=0)
                    dX[:, :, i:i+k, j:j+k] += (
                        dout[:, f, i, j][:, None, None, None] * self.W_conv[f]
                    )
        return dX, dW, db

    # ── MaxPool helpers ───────────────────────────────────────────────────────

    def _maxpool_forward(self, X: np.ndarray):
        """X : (n, C, H, W) → (n, C, H//p, W//p)"""
        p = self.pool_size
        n, C, H, W = X.shape
        H_p, W_p = H // p, W // p
        out = np.zeros((n, C, H_p, W_p))
        mask = np.zeros_like(X, dtype=bool)
        for i in range(H_p):
            for j in range(W_p):
                window = X[:, :, i*p:(i+1)*p, j*p:(j+1)*p]   # (n,C,p,p)
                max_val = window.max(axis=(2, 3), keepdims=True)
                out[:, :, i, j] = max_val.squeeze((2, 3))
                mask[:, :, i*p:(i+1)*p, j*p:(j+1)*p] = (window == max_val)
        return out, mask

    def _maxpool_backward(self, dout: np.ndarray, mask: np.ndarray):
        """Distribute gradient to the max position."""
        p = self.pool_size
        dX = np.zeros_like(mask, dtype=float)
        H_p, W_p = dout.shape[2], dout.shape[3]
        for i in range(H_p):
            for j in range(W_p):
                dX[:, :, i*p:(i+1)*p, j*p:(j+1)*p] = (
                    mask[:, :, i*p:(i+1)*p, j*p:(j+1)*p]
                    * dout[:, :, i, j][:, :, None, None]
                )
        return dX

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNNumPy":
        """
        X : (n, C, H, W)  — images in channel-first format
        y : (n,)           — integer class labels
        """
        n, C, H, W = X.shape
        k = self.kernel_size
        p = self.pool_size
        # Compute output sizes
        H_c = H - k + 1
        W_c = W - k + 1
        H_p = H_c // p
        W_p = W_c // p
        fc_in = self.n_filters * H_p * W_p

        # Initialize weights
        self.W_conv = np.random.randn(self.n_filters, C, k, k) * 0.1
        self.b_conv = np.zeros(self.n_filters)
        self.W_fc = np.random.randn(fc_in, self.n_classes) * 0.1
        self.b_fc = np.zeros(self.n_classes)

        for _ in range(self.n_iters):
            # ── Forward ───────────────────────────────────────────────────
            conv_out = self._conv_forward(X)               # (n,F,Hc,Wc)
            relu_out = np.maximum(0, conv_out)
            pool_out, pool_mask = self._maxpool_forward(relu_out)
            flat = pool_out.reshape(n, -1)                 # (n, fc_in)
            logits = flat @ self.W_fc + self.b_fc          # (n, K)
            # Softmax + cross-entropy
            logits -= logits.max(axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)

            # ── Backward ──────────────────────────────────────────────────
            y_oh = np.zeros_like(probs)
            y_oh[np.arange(n), y] = 1
            d_logits = (probs - y_oh) / n                  # (n, K)

            dW_fc = flat.T @ d_logits
            db_fc = d_logits.sum(axis=0)
            d_flat = d_logits @ self.W_fc.T

            d_pool = d_flat.reshape(pool_out.shape)
            d_relu = self._maxpool_backward(d_pool, pool_mask)
            d_conv = d_relu * (conv_out > 0)               # ReLU grad
            _, dW_conv, db_conv = self._conv_backward(d_conv, X)

            # ── Update ────────────────────────────────────────────────────
            self.W_fc   -= self.lr * dW_fc
            self.b_fc   -= self.lr * db_fc
            self.W_conv -= self.lr * dW_conv
            self.b_conv -= self.lr * db_conv

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        conv_out = self._conv_forward(X)
        relu_out = np.maximum(0, conv_out)
        pool_out, _ = self._maxpool_forward(relu_out)
        flat = pool_out.reshape(len(X), -1)
        logits = flat @ self.W_fc + self.b_fc
        return logits.argmax(axis=1)


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class CNNTorch:
    """
    CNN with raw PyTorch tensors and full manual backpropagation.
    No autograd, no F.conv2d — every operation (conv, pool, relu, fc) is
    implemented with explicit forward + backward passes, identical math to
    CNNNumPy but using torch tensors instead of numpy arrays.

    Architecture: Conv2D → ReLU → MaxPool → Flatten → Linear

    Backward pass summary:
        FC:      d_logits = (probs - Y_onehot) / n
                 dW_fc = flat.T @ d_logits
                 db_fc = d_logits.sum(dim=0)
                 d_flat = d_logits @ W_fc.T
        MaxPool: gradient flows only to the max position (max-mask)
        ReLU:    d_relu = d_pool * (conv_out > 0)
        Conv:    dW[f] += sum over patches: dout[n,f,i,j] * patch
                 dX at patch += dout[n,f,i,j] * W[f]
                 db = dout.sum over n,i,j

    Parameters
    ----------
    n_filters   : number of conv filters
    kernel_size : square filter size
    pool_size   : max pooling window
    n_classes   : output classes
    lr          : learning rate
    n_iters     : training steps
    """

    def __init__(self, n_filters: int = 8, kernel_size: int = 3, pool_size: int = 2,
                 n_classes: int = 10, lr: float = 0.01, n_iters: int = 100):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.n_classes = n_classes
        self.lr = lr
        self.n_iters = n_iters
        self.W_conv: torch.Tensor = None
        self.b_conv: torch.Tensor = None
        self.W_fc:   torch.Tensor = None
        self.b_fc:   torch.Tensor = None

    # ── Conv helpers ──────────────────────────────────────────────────────────

    def _conv_forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X : (n, C, H, W)
        Returns: (n, n_filters, H_out, W_out)
        """
        n, C, H, W = X.shape
        k = self.kernel_size
        H_out = H - k + 1
        W_out = W - k + 1
        out = torch.zeros(n, self.n_filters, H_out, W_out)
        for f in range(self.n_filters):
            for i in range(H_out):
                for j in range(W_out):
                    patch = X[:, :, i:i+k, j:j+k]           # (n, C, k, k)
                    out[:, f, i, j] = (patch * self.W_conv[f]).sum(dim=(1, 2, 3)) + self.b_conv[f]
        return out

    def _conv_backward(self, dout: torch.Tensor, X: torch.Tensor):
        """
        dout : (n, n_filters, H_out, W_out)
        Returns dX, dW_conv, db_conv
        """
        n, C, H, W = X.shape
        k = self.kernel_size
        H_out, W_out = dout.shape[2], dout.shape[3]
        dX      = torch.zeros_like(X)
        dW_conv = torch.zeros_like(self.W_conv)
        db_conv = dout.sum(dim=(0, 2, 3))             # (n_filters,)

        for f in range(self.n_filters):
            for i in range(H_out):
                for j in range(W_out):
                    patch = X[:, :, i:i+k, j:j+k]           # (n, C, k, k)
                    # dout[:, f, i, j] : (n,)  →  unsqueeze to (n,1,1,1) for broadcast
                    d = dout[:, f, i, j].reshape(n, 1, 1, 1)
                    dW_conv[f] += (d * patch).sum(dim=0)     # (C, k, k)
                    dX[:, :, i:i+k, j:j+k] += d * self.W_conv[f]  # distribute to input
        return dX, dW_conv, db_conv

    # ── MaxPool helpers ───────────────────────────────────────────────────────

    def _maxpool_forward(self, X: torch.Tensor):
        """X : (n, C, H, W) → (n, C, H//p, W//p), also returns max-position mask."""
        p = self.pool_size
        n, C, H, W = X.shape
        H_p, W_p = H // p, W // p
        out  = torch.zeros(n, C, H_p, W_p)
        mask = torch.zeros_like(X, dtype=torch.bool)
        for i in range(H_p):
            for j in range(W_p):
                window = X[:, :, i*p:(i+1)*p, j*p:(j+1)*p]          # (n,C,p,p)
                max_val = window.reshape(n, C, -1).max(dim=2).values  # (n,C)
                out[:, :, i, j] = max_val
                mask[:, :, i*p:(i+1)*p, j*p:(j+1)*p] = (
                    window == max_val.unsqueeze(-1).unsqueeze(-1)
                )
        return out, mask

    def _maxpool_backward(self, dout: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Gradient flows only to the max position in each window."""
        p = self.pool_size
        dX = torch.zeros_like(mask, dtype=torch.float32)
        H_p, W_p = dout.shape[2], dout.shape[3]
        for i in range(H_p):
            for j in range(W_p):
                dX[:, :, i*p:(i+1)*p, j*p:(j+1)*p] = (
                    mask[:, :, i*p:(i+1)*p, j*p:(j+1)*p].float()
                    * dout[:, :, i, j].unsqueeze(-1).unsqueeze(-1)
                )
        return dX

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X, y) -> "CNNTorch":
        """
        X : array-like (n, C, H, W)
        y : array-like (n,)  integer labels
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        n, C, H, W = X_t.shape
        k = self.kernel_size
        p = self.pool_size
        H_c = H - k + 1
        W_c = W - k + 1
        H_p = H_c // p
        W_p = W_c // p
        fc_in = self.n_filters * H_p * W_p

        self.W_conv = torch.randn(self.n_filters, C, k, k) * 0.1
        self.b_conv = torch.zeros(self.n_filters)
        self.W_fc   = torch.randn(fc_in, self.n_classes) * 0.1
        self.b_fc   = torch.zeros(self.n_classes)

        for _ in range(self.n_iters):
            # ── Forward ───────────────────────────────────────────────────
            conv_out = self._conv_forward(X_t)             # (n,F,Hc,Wc)
            relu_out = torch.clamp(conv_out, min=0)        # ReLU
            pool_out, pool_mask = self._maxpool_forward(relu_out)
            flat    = pool_out.reshape(n, -1)              # (n, fc_in)
            logits  = flat @ self.W_fc + self.b_fc         # (n, K)

            # Softmax (numerically stable) + cross-entropy
            logits_s = logits - logits.max(dim=1, keepdim=True).values
            exp_l    = torch.exp(logits_s)
            probs    = exp_l / exp_l.sum(dim=1, keepdim=True)   # (n, K)

            # ── Backward ──────────────────────────────────────────────────
            y_oh = torch.zeros_like(probs)
            y_oh[torch.arange(n), y_t] = 1.0
            d_logits = (probs - y_oh) / n                  # (n, K)

            dW_fc = flat.T @ d_logits                      # (fc_in, K)
            db_fc = d_logits.sum(dim=0)                    # (K,)
            d_flat = d_logits @ self.W_fc.T                # (n, fc_in)

            d_pool = d_flat.reshape(pool_out.shape)        # (n,F,Hp,Wp)
            d_relu = self._maxpool_backward(d_pool, pool_mask)
            d_conv = d_relu * (conv_out > 0).float()       # ReLU grad

            _, dW_conv, db_conv = self._conv_backward(d_conv, X_t)

            # ── Update ────────────────────────────────────────────────────
            self.W_fc   -= self.lr * dW_fc
            self.b_fc   -= self.lr * db_fc
            self.W_conv -= self.lr * dW_conv
            self.b_conv -= self.lr * db_conv

        return self

    def predict(self, X) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32)
        conv_out = self._conv_forward(X_t)
        relu_out = torch.clamp(conv_out, min=0)
        pool_out, _ = self._maxpool_forward(relu_out)
        flat   = pool_out.reshape(len(X_t), -1)
        logits = flat @ self.W_fc + self.b_fc
        return logits.argmax(dim=1).numpy()
