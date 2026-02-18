import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
  def __init__(self, num_inputs, learning_rate=0.1):
    self.weights = np.random.rand(num_inputs + 1)
    self.learning_rate = learning_rate

  def weighted_sum(self, inputs):
    Z = np.dot(inputs, self.weights[1:]) + self.weights[0]
    return Z

  def predict(self, X):
    if X.ndim==1:
      Z = np.dot(X, self.weights[1:] + self.weights[0])
      return self.Z
    else:
      Z = np.array([self.weighted_sum(x) for x in X])
      return self.Z

  def loss(self, prediction, target):
    return target - prediction

  def fit(self, X, y, tolerance, n_epochs=100):
    for i in range(n_epochs):
      y_pred = self.predict(X)
      error = self.loss(y_pred, y)
      mse = (error)**2
      if mse < 10**-5:
        return mse
      
      change_w = -2 * X *error
      change_b = -2 * error
      w_new = self.weights[1:] - self.learning_rate * change_w 
      b_new = self.weights[0] - self.learning_rate * change_b 
      self.weights[1:] = w_new 
      self.weights[0] = b_new


    
if __name__ == '__main__':
  
    x = np.linspace(-10, 10, 1000)
    y = 2*x + 8

    error = np.random.randint(-5, 5, 1000)
    y_synt = y + error

    plt.plot(x, y_synt, 'o', c="r")
    plt.plot(x, y)
    plt.show()