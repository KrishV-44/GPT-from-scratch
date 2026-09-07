import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maxVal = np.max(z)
        stable_z = z - maxVal
        exp_z = np.exp(stable_z)
        total = np.sum(exp_z)
        probs = exp_z / total
        return np.round(probs, 4)
