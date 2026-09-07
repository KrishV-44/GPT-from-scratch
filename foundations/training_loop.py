import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        n_samples, n_features = X.shape

        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        for i in range(epochs):
            y_hat = np.dot(X,w) + b
            error = y_hat - y
            dL_dw = 2*np.dot(X.T, error) / n_samples
            dL_db = 2*np.mean(y_hat - y)

            w -= lr * dL_dw
            b -= lr * dL_db
        
        return (np.round(w,5), round(b,5))






