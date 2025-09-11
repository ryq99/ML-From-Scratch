import torch

class Lasso():
    def __init__(self, w, b, alpha=1.0):
        self.w = w
        self.b = b
        self.alpha = alpha

    def predict(self, X):
        return X @ self.w + self.b
    
    def fit(self, X, y, lr=0.01, epochs=1000):
        for epoch in range(epochs):

            y_pred = self.predict(X)
            loss = torch.mean((y_pred - y) ** 2) + self.alpha * torch.sum(self.w)
            grad_loss_y_pred = 2 * (y_pred - y)
            grad_loss_w = torch.mean(torch.transpose(X, 0, 1) @ grad_loss_y_pred) + self.alpha * torch.sum(torch.sign(self.w)) # use SGD soft thresholding operator for L1 regularization
            grad_loss_b = torch.mean(grad_loss_y_pred)

            with torch.no_grad():
                self.w -= lr * grad_loss_w
                self.b -= lr * grad_loss_b

            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Weight: {self.w.squeeze(-1).tolist()}, Bias: {self.b.item():.4f}')

if __name__ == "__main__":
    X = torch.randn(200, 3, dtype=torch.float32)
    y = X @ (3 + torch.randn(3, 1, dtype=torch.float32) * 0.01) + torch.randn(1, dtype=torch.float32) * 0.01

    w = 3 + torch.randn(3, 1, dtype=torch.float32) * 0.001
    b = torch.randn(1, dtype=torch.float32)

    model = Lasso(w=w, b=b, alpha=1.0)
    model.fit(X, y, lr=0.001, epochs=1000)