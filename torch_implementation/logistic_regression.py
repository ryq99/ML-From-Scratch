import torch

class LogisticRegression():
    def __init__(self, w, b):
        self.w = w
        self.b = b
    
    def predict(self, X):

        return 1 / (1 + torch.exp(-(X @ self.w + self.b)))
    
    def fit(self, X, y, lr=0.01, epoches=1000):

        for epoch in range(epoches):

            y_pred = self.predict(X)
            loss = torch.mean(- y * torch.log(y_pred) - (1 - y) * torch.log(1 - y_pred))
            grad_loss = y_pred - y # The gradient of the loss with respect to the logit is: (y_pred - y), which is the chain rule of the derivation of loss with respect to y_pred times the derivative of y_pred with respect to z, the logit (sigmoid function). 
            grad_w = (torch.transpose(X, 0, 1) @ grad_loss) / X.shape[0] # shape (n, d)
            grad_b = torch.mean(grad_loss)

            #print("X:", X.shape, "y:", y_pred.shape , "grad_loss:", grad_loss.shape, "grad_w:", grad_w.shape)

            with torch.no_grad():
                self.w -= lr * grad_w
                self.b -= lr * grad_b

            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{epoches}], Loss: {loss.item():.4f}, Weight: {self.w.item():.4f}, Bias: {self.b.item():.4f}')


if __name__ == "__main__":

    X = torch.randn(200, 1, dtype=torch.float32) # (n, d)
    y = (3 * X + 2 + torch.randn(200, 1, dtype=torch.float32) * 0.01 > 0.5).float()  # (n, d)

    w_0 = torch.randn(1, 1, dtype=torch.float32) * 0.00001 + 3 # (d, 1)
    b_0 = torch.randn(1, dtype=torch.float32) # scalar

    model = LogisticRegression(w=w_0, b=b_0)
    model.fit(X, y, lr=0.001, epoches=500)