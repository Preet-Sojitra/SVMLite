"""Support Vector Machine (SVM) implementation."""

import numpy as np


class SVCLite:
    def __init__(self, C: float = 1.0):
        """
        Initialize the SVM model.

        Args:
            C (float, optional): Regularization parameter. Defaults to 1.0.

        Raises:
            ValueError: If C is not positive.
        """
        if C <= 0:
            raise ValueError("Regularization parameter C must be positive.")

        self.C = float(C)
        self.weights = None
        self.bias = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        n_iters: int = 1000,
        batch_size: int = 32,
    ) -> None:
        """
        Train the SVM model using mini-batch gradient descent.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).
            learning_rate (float, optional): Learning rate for gradient descent. Defaults to 0.01.
            n_iters (int, optional): Number of iterations for training. Defaults to 1000.
            batch_size (int, optional): Size of mini-batches for gradient descent. Defaults to 32.
        Raises:
            ValueError: If learning_rate is not positive.
            ValueError: If n_iters is not positive.
            ValueError: If batch_size is not positive.
            TypeError: If X or y is not a numpy array.
            ValueError: If the number of samples in X and y are not equal.
            ValueError: If y is not a one-dimensional array.
        """

        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if n_iters <= 0:
            raise ValueError("Number of iterations must be positive.")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")

        if len(X) != len(y):
            raise ValueError("Number of samples in X and y must be equal.")

        if len(y.shape) != 1:
            raise ValueError("y must be a one-dimensional array.")

        learning_rate = float(learning_rate)
        n_iters = n_iters if isinstance(n_iters, int) else int(n_iters)
        batch_size = batch_size if isinstance(batch_size, int) else int(batch_size)

        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0.0

        for i in range(n_iters):
            # shuffle the data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, num_samples, batch_size):
                X_mini_batch = X_shuffled[start : start + batch_size]
                y_mini_batch = y_shuffled[start : start + batch_size]

                dJ_dw_batch = np.zeros_like(self.weights)
                dJ_db_batch = 0.0

                # loop over each sample in mini-batch
                for idx, x_k in enumerate(X_mini_batch):
                    # compute the decision value
                    condition = y_mini_batch[idx] * (
                        np.dot(x_k, self.weights) + self.bias
                    )

                    if condition < 1:
                        # misclassified or within margin
                        # add gradient for only hinge loss part
                        dJ_dw_batch += -y_mini_batch[idx] * x_k
                        dJ_db_batch += -y_mini_batch[idx]

                # perform update for this mini-batch
                dJ_dw_batch = self.weights + self.C * (dJ_dw_batch / len(X_mini_batch))
                dJ_db_batch = self.C * dJ_db_batch / len(X_mini_batch)

                # update weights and bias
                self.weights -= learning_rate * dJ_dw_batch
                self.bias -= learning_rate * dJ_db_batch

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the input data X.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Raises:
            ValueError: If the model is not trained yet.
            TypeError: If the input X is not a numpy array.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model is not trained yet. Please call 'fit' first.")

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")

        linear_output = np.dot(X, self.weights) + self.bias
        # return +1 if linear_output >= 0 else -1
        predictions = np.where(linear_output >= 0, 1, -1)
        return predictions
