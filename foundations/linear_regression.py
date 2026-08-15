import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        y = np.dot(X, weights)
        return np.round(y, 5)
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        pass

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        res = 1/len(ground_truth) * np.sum(pow((ground_truth - model_prediction), 2))
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        return np.round(res, 5)

        pass
