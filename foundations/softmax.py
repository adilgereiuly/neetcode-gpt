import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        new_z = z - np.max(z)
        softmax = pow(np.e, new_z) / np.sum(pow(np.e, new_z))
        return np.round(softmax, 4)
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        pass
