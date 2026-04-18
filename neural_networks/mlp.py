"""
Multilayer Perceptron (MLP) with Backpropagation
=================================================
A fully-connected feed-forward network trained with stochastic gradient descent.

Forward pass:  z^l = W^l a^{l-1} + b^l,   a^l = relu(z^l)  (or sigmoid for output)
Backward pass (chain rule):
    delta^L = dL/da^L * sigma'(z^L)           (output layer)
    delta^l = (W^{l+1})^T delta^{l+1} * relu'(z^l)
    dW^l    = delta^l (a^{l-1})^T  /  n
    db^l    = mean(delta^l, axis=0)
"""
import numpy as np
from typing import List


# ── NumPy ─────────────────────────────────────────────────────────────────────

class MLPNumPy:
    """
    Fully-connected MLP for classification or regression.

    Parameters
    ----------
    layer_sizes : list of hidden layer sizes, e.g. [64, 32]
    n_outputs   : output dimension
    task        : 'classification' (softmax + CE) | 'regression' (linear + MSE)
    lr          : learning rate
    n_iters     : gradient descent steps (full-batch)
    """

    def __init__(self, layer_sizes: List[int], n_outputs: int = 1,
                 task: str = "regression", lr: float = 0.01, n_iters: int = 1000):
        self.layer_sizes = layer_sizes
        self.n_outputs = n_outputs
        self.task = task
        self.lr = lr
        self.n_iters = n_iters
        self.weights: List[np.ndarray] = []
        self.biases:  List[np.ndarray] = []

    @staticmethod
    def _relu(z): return np.maximum(0, z)
    @staticmethod
    def _relu_grad(z): return (z > 0).astype(float)
    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _init_weights(self, n_features: int):
        sizes = [n_features] + self.layer_sizes + [self.n_outputs]
        self.weights = []
        self.biases  = []
        for i in range(len(sizes) - 1):
            # He initialization for ReLU layers
            scale = np.sqrt(2.0 / sizes[i])
            self.weights.append(np.random.randn(sizes[i], sizes[i + 1]) * scale)
            self.biases.append(np.zeros(sizes[i + 1]))

    def _forward(self, X: np.ndarray):
        """Returns (activations, pre-activations) at every layer."""
        activations = [X]
        zs = []
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            zs.append(z)
            is_last = (i == len(self.weights) - 1)
            if is_last and self.task == "classification":
                a = self._softmax(z)
            elif is_last:
                a = z                      # linear output for regression
            else:
                a = self._relu(z)
            activations.append(a)
        return activations, zs

    def _backward(self, activations, zs, y: np.ndarray):
        n = len(y)
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)

        # Output layer delta
        if self.task == "classification":
            # Cross-entropy + softmax: delta = p - y_onehot
            y_oh = np.zeros_like(activations[-1])
            y_oh[np.arange(n), y.astype(int)] = 1
            delta = activations[-1] - y_oh        # (n, n_out)
        else:
            delta = 2 * (activations[-1] - y[:, None]) / n   # MSE grad

        for l in range(len(self.weights) - 1, -1, -1):
            dW[l] = activations[l].T @ delta / n
            db[l] = delta.mean(axis=0)
            if l > 0:
                delta = (delta @ self.weights[l].T) * self._relu_grad(zs[l - 1])

        return dW, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPNumPy":
        """
        X : (n_samples, n_features)
        y : (n_samples,)  class indices for classification, floats for regression
        """
        self._init_weights(X.shape[1])
        for _ in range(self.n_iters):
            activations, zs = self._forward(X)
            dW, db = self._backward(activations, zs, y)
            for l in range(len(self.weights)):
                self.weights[l] -= self.lr * dW[l]
                self.biases[l]  -= self.lr * db[l]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self._forward(X)
        out = activations[-1]
        if self.task == "classification":
            return out.argmax(axis=1)
        return out.squeeze()


# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch


class MLPTorch:
    """
    MLP with raw PyTorch tensors and full manual backpropagation.
    No autograd, no nn.Module — every gradient computed explicitly by chain rule,
    identical math to MLPNumPy.

    Forward (per layer l):
        z^l = a^{l-1} @ W^l + b^l
        a^l = relu(z^l)          for hidden layers
        a^L = softmax(z^L)       for classification output
        a^L = z^L                for regression output

    Backward:
        Output delta (classification):  delta^L = (1/n)*(a^L - Y_onehot)
        Output delta (regression):      delta^L = (2/n)*(a^L - y)
        Hidden delta:  delta^l = (delta^{l+1} @ W^{l+1}.T) * relu'(z^l)
        Weight grad:   dW^l = (a^{l-1}).T @ delta^l
        Bias grad:     db^l = sum(delta^l, dim=0)

    Parameters
    ----------
    layer_sizes : hidden layer widths
    n_outputs   : output dimension
    task        : 'classification' | 'regression'
    lr          : learning rate
    n_iters     : training steps
    """

    def __init__(self, layer_sizes: List[int], n_outputs: int = 1,
                 task: str = "regression", lr: float = 0.01, n_iters: int = 1000):
        self.layer_sizes = layer_sizes
        self.n_outputs = n_outputs
        self.task = task
        self.lr = lr
        self.n_iters = n_iters
        self.weights: List[torch.Tensor] = []
        self.biases:  List[torch.Tensor] = []

    @staticmethod
    def _relu(z: torch.Tensor) -> torch.Tensor:
        return torch.clamp(z, min=0)

    @staticmethod
    def _relu_grad(z: torch.Tensor) -> torch.Tensor:
        return (z > 0).float()

    @staticmethod
    def _softmax(z: torch.Tensor) -> torch.Tensor:
        z = z - z.max(dim=1, keepdim=True).values
        e = torch.exp(z)
        return e / e.sum(dim=1, keepdim=True)

    def _init_weights(self, n_features: int):
        sizes = [n_features] + self.layer_sizes + [self.n_outputs]
        self.weights = []
        self.biases  = []
        for i in range(len(sizes) - 1):
            scale = (2.0 / sizes[i]) ** 0.5     # He initialization
            self.weights.append(torch.randn(sizes[i], sizes[i + 1]) * scale)
            self.biases.append(torch.zeros(sizes[i + 1]))

    def _forward(self, X: torch.Tensor):
        """Returns (activations list, pre-activations list) — same structure as NumPy."""
        activations = [X]
        zs = []
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            zs.append(z)
            is_last = (i == len(self.weights) - 1)
            if is_last and self.task == "classification":
                a = self._softmax(z)
            elif is_last:
                a = z                          # linear output
            else:
                a = self._relu(z)
            activations.append(a)
        return activations, zs

    def _backward(self, activations, zs, y: torch.Tensor):
        """Explicit chain-rule backprop — mirrors MLPNumPy._backward exactly."""
        n = len(y)
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)

        # Output layer delta
        if self.task == "classification":
            y_oh = torch.zeros_like(activations[-1])
            y_oh[torch.arange(n), y.long()] = 1.0
            delta = (activations[-1] - y_oh) / n       # combined softmax+CE grad / n
        else:
            delta = 2 * (activations[-1] - y.unsqueeze(1)) / n

        for l in range(len(self.weights) - 1, -1, -1):
            dW[l] = activations[l].T @ delta / n if self.task == "regression" \
                    else activations[l].T @ delta
            db[l] = delta.sum(dim=0)
            if l > 0:
                delta = (delta @ self.weights[l].T) * self._relu_grad(zs[l - 1])

        return dW, db

    def fit(self, X, y) -> "MLPTorch":
        """
        X : array-like (n, d)
        y : array-like (n,)
        """
        self._init_weights(np.array(X).shape[1])
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        for _ in range(self.n_iters):
            activations, zs = self._forward(X_t)
            dW, db = self._backward(activations, zs, y_t)
            for l in range(len(self.weights)):
                self.weights[l] -= self.lr * dW[l]
                self.biases[l]  -= self.lr * db[l]

        return self

    def predict(self, X) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32)
        activations, _ = self._forward(X_t)
        out = activations[-1]
        if self.task == "classification":
            return out.argmax(dim=1).numpy()
        return out.squeeze().numpy()
