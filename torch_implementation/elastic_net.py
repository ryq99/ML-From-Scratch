import torch

class ElasticNet():
    def __init__(self, w, b, alpha=1.0, l1_ratio=0.5):
        self.w = w
        self.b = b
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def predict(self, X):
        return X @ self.w + self.b
    
    def fit(self, X, y, lr=0.01, epochs=1000, solver='gd'):
        if solver == 'gd':
            for epoch in range(epochs):

                y_pred = self.predict(X)
                loss = torch.mean((y_pred - y) ** 2) + self.alpha * ( self.l1_ratio * torch.sum(torch.abs(self.w)) + (1 - self.l1_ratio) * torch.sum(self.w ** 2) )
                grad_loss_y_pred = 2 * (y_pred - y)
                grad_loss_w = (torch.transpose(X, 0, 1) @ grad_loss_y_pred) / X.shape[0] + self.alpha * (self.l1_ratio * torch.sign(self.w) + (1 - self.l1_ratio) * 2 * self.w)
                grad_loss_b = torch.mean(grad_loss_y_pred)

                with torch.no_grad():
                    self.w -= lr * grad_loss_w
                    self.b -= lr * grad_loss_b

                if epoch % 2 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Weight: {self.w.squeeze(-1).tolist()}, Bias: {self.b.item():.4f}')
        
        
        elif solver == 'sgd':
            torch.manual_seed(42)
            grads_w_all_samples = torch.zeros(X.shape[0], self.w.shape[0])
            grads_b_all_samples = torch.zeros(X.shape[0], 1)

            for epoch in range(epochs):
                # stochastic with replacement
                #idx = torch.randint(0, X.shape[0], (1,)).item() 
                #x_sample = X[idx:idx+1, :]
                #y_sample = y[idx:idx+1, :]

                # stochastic without replacement
                epoch_sum_loss = 0
                indices = torch.randperm(X.shape[0])
                for idx in indices:
                    x_sample = X[idx:idx+1, :]
                    y_sample = y[idx:idx+1, :]
                
                    y_pred_sample = self.predict(x_sample)
                    loss = (y_pred_sample - y_sample) ** 2 + self.alpha * (self.l1_ratio * torch.sum(torch.abs(self.w)) + (1 - self.l1_ratio) * torch.sum(self.w ** 2))
                    grad_loss_y_pred = 2 * (y_pred_sample - y_sample)
                    grad_loss_w = torch.transpose(x_sample, 0, 1) @ grad_loss_y_pred + self.alpha * (self.l1_ratio * torch.sign(self.w) + (1 - self.l1_ratio) * 2 * self.w)
                    grad_loss_b = grad_loss_y_pred.squeeze(-1)
                    grads_w_all_samples[idx, :] = grad_loss_w.reshape(-1)
                    grads_b_all_samples[idx, :] = grad_loss_b.reshape(-1)

                    with torch.no_grad():
                        self.w -= lr * grad_loss_w
                        self.b -= lr * grad_loss_b

                    epoch_sum_loss += loss.item()
                
                if epoch % 2 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_sum_loss / X.shape[0]:.4f}, Weight: {self.w.squeeze(-1).tolist()}, Bias: {self.b.item():.4f}')


if __name__ == "__main__":
    X = torch.randn(200, 3, dtype=torch.float32)
    y = X @ (3 + torch.randn(3, 1, dtype=torch.float32) * 0.01) + torch.randn(1, dtype=torch.float32) * 0.01

    w = 3 + torch.randn(3, 1, dtype=torch.float32) * 0.01
    b = torch.randn(1, dtype=torch.float32)

    model = ElasticNet(w=w, b=b, alpha=1.0, l1_ratio=0.5)
    model.fit(X, y, lr=0.001, epochs=1000, solver='sgd')