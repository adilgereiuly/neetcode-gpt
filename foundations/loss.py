import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        small_eps = 1e-7
        new_pred = y_pred + small_eps
        BCE = y_true * np.log(new_pred) + (1 - y_true) * np.log(1 - new_pred)
        res = -1/len(y_true) * np.sum(BCE)
        return round(res, 4)
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        small_eps = 1e-7
        new_pred = y_pred + small_eps
        CCE = y_true * np.log(new_pred)
        res = -1/len(y_true) * np.sum(CCE)
        return round(res, 4)

        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        pass
