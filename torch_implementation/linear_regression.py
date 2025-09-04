import torch

class LinearRegression():
    def __init__(self):
        self.w = 3 + torch.randn(1, requires_grad=True)
        self.b = 2 + torch.randn(1, requires_grad=True)
    
    def predict(self, X):

        return X @ self.w + self.b
    
    def fit(self, X, y, lr=0.01, epochs=1000):
        
        for epoch in range(epochs):

            # predict pass
            y_pred = self.predict(X)
            # Loss
            loss = torch.mean((y_pred - y) ** 2)
            # The gradient of the loss with respect to y_pred
            grad_loss = torch.mean(2 * (y_pred - y))
            # Gradient of the loss with respect to w
            grad_w = torch.mean(grad_loss * X)
            # Gradient of the loss with respect to b (just the mean)
            grad_b = grad_loss

            with torch.no_grad():
                self.w -= lr * grad_w
                self.b -= lr * grad_b

            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Weight: {self.w.item():.4f}, Bias: {self.b.item():.4f}')


if __name__ == "__main__":
    # Generate data
    X = torch.randn(200, dtype=torch.float32).unsqueeze(-1)
    y = 3 * X + 2 + torch.randn(200, dtype=torch.float32).unsqueeze(-1) * 0.01
    print(f"Shape X = {X.shape}, Shape y = {y.shape}")

    model = LinearRegression()
    model.fit(X, y, lr=0.001, epochs=500)

    model.predict(X)


            