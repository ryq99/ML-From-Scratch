import torch

class LogisticRegression():
    def __init__(self, w, b):
        self.w = w
        self.b = b
    
    def predict(self, X):

        return softmax(X @ self.w + self.b)
    
    def fit(self, X, y, lr=0.01, epoches=1000):

        for epoch in range(epoches):

            y_pred = self.predict(X)
            loss = torch.mean(torch.sum(-y * torch.log(y_pred + 1e-8), dim=-1)) # cross entropy loss
            grad_loss_y_pred = y_pred - y # The gradient of the loss with respect to the logit is: (y_pred - y), which is the chain rule of the derivation of loss with respect to y_pred times the derivative of y_pred with respect to z, the logit (sigmoid function). 
            grad_w = torch.transpose(X, 0, 1) @ grad_loss_y_pred / X.shape[0] # shape (n, d)
            grad_b = torch.mean(grad_loss_y_pred)

            #print("X:", X.shape, "y:", y_pred.shape , "grad_loss:", grad_loss.shape, "grad_w:", grad_w.shape)

            with torch.no_grad():
                self.w -= lr * grad_w
                self.b -= lr * grad_b


            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{epoches}], Loss: {loss.item():.4f}, Weight: {self.w.reshape(-1).tolist()}, Bias: {self.b.reshape(-1).tolist()}')

def softmax(z):
    z = z - torch.max(z, dim=-1, keepdim=True)[0]
    z = torch.exp(z)
    z = z / torch.sum(z, dim=-1, keepdim=True)
    return z


if __name__ == "__main__":
    torch.manual_seed(42)

    n, d, k = 200, 3, 2
    X = torch.randn(n, d, dtype=torch.float32)
    y = torch.cat([
        (X @ torch.randn(3, 1, dtype=torch.float32) * 3 + 2 + torch.randn(200, 1, dtype=torch.float32) * 0.01 > 0.5).float(),
        (X @ torch.randn(3, 1, dtype=torch.float32) * 2 + 1 + torch.randn(200, 1, dtype=torch.float32) * 0.01 > 0.3).float()
        ], dim=-1)  # (n, k) k classes

    w_0 = torch.randn(d, k, dtype=torch.float32) * 0.00001 + 3 # (d, k)
    b_0 = torch.randn(k, dtype=torch.float32) # (k,)

    model = LogisticRegression(w=w_0, b=b_0)
    model.fit(X, y, lr=0.001, epoches=500)