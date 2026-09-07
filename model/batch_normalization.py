import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        eps = 1e-5
        
        X = np.array(x, dtype=np.float64)
        g = np.array(gamma, dtype=np.float64)
        b = np.array(beta, dtype=np.float64)
        rm = np.array(running_mean, dtype=np.float64)
        rv = np.array(running_var, dtype=np.float64)

        if training:
            # 1. Compute batch mean and variance along axis 0 (across rows/batch)
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)

            # 2. Update running statistics
            rm = (1 - momentum) * rm + momentum * mean
            rv = (1 - momentum) * rv + momentum * var
        else:
            # During inference, use running statistics
            mean = rm
            var = rv

        # 3. Normalize and apply affine transformation
        X_hat = (X - mean) / np.sqrt(var + eps)
        Y = g * X_hat + b

        # 4. Round to 4 decimal places and convert back to Python lists
        y_list = np.round(Y, 4).tolist()
        rm_list = np.round(rm, 4).tolist()
        rv_list = np.round(rv, 4).tolist()

        return (y_list, rm_list, rv_list)

