import torch

class Ridge():
    def __init__(self, w, b, alpha=1.0):
        self.w = w
        self.b = b
        self.alpha = alpha
    
    def predict(self, X):
        return X @ self.w + self.b
    
    def fit(self, X, y, lr=0.01, epochs=1000):

        for epoch in range(epochs):

            y_pred = self.predict(X)
            print(y_pred.shape)
            loss = torch.mean((y_pred - y) ** 2) + torch.mean(self.alpha * self.w ** 2)
            
            grad_loss_y_pred = 2 * (y_pred - y)
            grad_loss_w = (torch.transpose(X, 0, 1) @ grad_loss_y_pred + 2 * self.alpha * self.w) / X.shape[0]
            grad_loss_b = torch.mean(grad_loss_y_pred)

            #print("X:", X.shape, "y:", y_pred.shape , "grad_loss_y_pred:", grad_loss_y_pred.shape, "grad_loss_w:", grad_loss_w.shape, "grad_loss_b:", grad_loss_b.shape)

            with torch.no_grad():
                self.w -= lr * grad_loss_w
                self.b -= lr * grad_loss_b

            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Weight: {self.w.squeeze(-1).tolist()}, Bias: {self.b.item():.4f}')


if __name__ == "__main__":
    
    X = torch.randn(200, 3, dtype=torch.float32) # (n, d)
    y = X @ (3 * torch.randn(3, 1, dtype=torch.float32)) + 2 + torch.randn(200, 1, dtype=torch.float32) * 0.01 # (n, d)
    print(f"Shape X = {X.shape}, Shape y = {y.shape}")

    w_0 = torch.randn(3, 1, dtype=torch.float32) * 0.001 + 3 # (d, 1)
    b_0 = torch.randn(1, dtype=torch.float32) # scalar

    model = Ridge(w=w_0, b=b_0, alpha=1.0)
    model.fit(X, y, lr=0.001, epochs=1000)