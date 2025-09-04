import torch

class LinearRegression(torch.nn.Module):
    def _init__(self, w, b):
        self.w = torch.tensor(w, dtype=torch.float32, requires_grad=True)
        self.b = torch.tensor(b, dtype=torch.float32, requires_grad=True)
    
    def fit(self, X, y, lr=0.01, epochs=1000):
        
        for epoch in range(epochs):
            # Zero the gradients.
            self.w.grad.zero_()
            self.b.grad.zero_()
            # Forward pass
            y_pred = self.w @ X + self.b
            # Loss
            loss = torch.mean(y_pred - y) ** 2
            # The gradient of the loss with respect to y_pred
            grad_y_pred = torch.mean(2 * (y_pred - y))
            # Gradient of the loss with respect to w
            grad_w = torch.mean(grad_y_pred * X)
            # Gradient of the loss with respect to b (just the mean)
            grad_b = grad_y_pred

            with torch.no_grad():
                self.w -= lr * grad_w
                self.b -= lr * grad_b

            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Weight: {self.w.item():.4f}, Bias: {self.b.item():.4f}')

            