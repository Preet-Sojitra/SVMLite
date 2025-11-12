"""Utilities for SVMLite."""

import numpy as np


class StandardScalerLite:
    """
    Simple standard scaler for feature normalization as SVMs are sensitive to feature scaling.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        """
        Compute the mean and std to be used for later scaling.

        Args:
            X (np.ndarray): Input data to compute the mean and std from.
        """

        if isinstance(X, np.ndarray) is False:
            raise TypeError("Input data must be a numpy array.")

        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array.")

        self.mean_ = np.mean(X, axis=0)  # axis=0 because want mean for each feature
        self.std_ = np.std(X, axis=0)
        # To avoid division by zero
        self.std_[self.std_ == 0] = (
            1.0  # replace 0 std with 1.0 because data will be divided by std and division by 1 does not change the value
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.

        Args:
            X (np.ndarray): Input data to be transformed.

        Returns:
            np.ndarray: Transformed data.
        """

        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("The scaler has not been fitted yet.")

        if isinstance(X, np.ndarray) is False:
            raise TypeError("Input data must be a numpy array.")

        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array.")

        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Args:
            X (np.ndarray): Input data to fit and transform.
        Returns:
            np.ndarray: Transformed data.
        """
        self.fit(X)
        return self.transform(X)
