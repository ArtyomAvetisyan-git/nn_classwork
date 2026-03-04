import numpy as np
class SimpleNN:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = np.random.rand(1, 4)

    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def loss(self, y, y_hat):
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    def fit(self, X, y, tolerance=1e-5, n_epochs=1000, batch_size=1000):
        w1, W2, b1, b2 = self.weights
        pass        
    
    def forward_propagation(self, X, W1, W2, b1, b2):
        z1 = np.dot(X, W1) + b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, W2) + b2 
        a2 = self.sigmoid(z2)
        return z1, a1, z2, a2
    
    def backward_propagation(self, X, y, z1, a1, z2, a2, W1, W2):
        grad_L_z2 = a2 - y
        grad_L_W2 = np.dot(a1.T, grad_L_z2)
        grad_L_b2 = np.sum(grad_L_z2, axis=0, keepdims=True)
        grad_L_a1 = np.dot(grad_L_z2, W2.T)
        grad_L_z1 = grad_L_a1 * self.relu_derivative(z1)
        grad_L_W1 = np.dot(X.T, grad_L_z1)
        grad_L_b1 = np.sum(grad_L_z1, axis=0, keepdims=True)
        return grad_L_W1, grad_L_b1, grad_L_W2, grad_L_b2
    
    def update_parameters(self, W1, b1, W2, b2, grad_L_W1, grad_L_b1, grad_L_W2, grad_L_b2):
        W1 = W1 - self.learning_rate * grad_L_W1
        W2 = W2 - self.learning_rate * grad_L_W2
        b1 = b1 - self.learning_rate * grad_L_b1
        b2 = b2 - self.learning_rate * grad_L_b2
        return W1, b1, W2, b2