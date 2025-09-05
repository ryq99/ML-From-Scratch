import torch

class LogisticRegression():
    def __init__(self):
        self.w = torch.randn(1, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)
    
    def predict(self, X):

        return 1 / (1 + torch.exp(-(X @ self.w + self.b)))
    
    def fit(self, X, y, lr=0.01, epoches=1000):

        for epoch in epoches:

            y_pred = self.predict(X)
            loss = torch.mean(- y * torch.log(y_pred) - (1 - y) * torch.log(1 - y_pred))
            grad_loss = torch.mean(y_pred - y) # The gradient of the loss with respect to the logit is: (y_pred - y), which is the chain rule of the derivation of loss with respect to y_pred times the derivative of y_pred with respect to z, the logit (sigmoid function). 
            grad_w = torch.mean(grad_loss * X)
            grad_b = torch.mean(grad_loss)

            with torch.no_grad():
                self.w -= lr * grad_w
                self.b -= lr * grad_b

            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{epoches}], Loss: {loss.item():.4f}, Weight: {self.w.item():.4f}, Bias: {self.b.item():.4f}')