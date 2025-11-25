"""Support Vector Machine (SVM) implementation."""

from functools import partial
import numpy as np
import cvxopt as opt
import cvxopt.solvers as solvers
from . import kernels


class SVCLite:
    def __init__(
        self, C: float = 1.0, solver: str = "SGD", kernel: str = "linear", **kwargs
    ):
        """
        Initialize the SVM model.

        Args:
            C (float, optional): Regularization parameter. Defaults to 1.0.
            solver (str, optional): Solver to use for optimization. Defaults to "SGD". Available options: "SGD", "QP".
            kernel (str, optional): Kernel function to use. Defaults to "linear".
            **kwargs: Additional keyword arguments for kernel functions.

        Raises:
            ValueError: If C is not positive.
        """
        self.solver = solver.lower()
        if C <= 0:
            raise ValueError("Regularization parameter C must be positive.")
        if self.solver not in ["sgd", "qp"]:
            raise ValueError("Solver must be either 'SGD' or 'QP'.")
        if self.solver == "sgd" and isinstance(kernel, str) and kernel != "linear":
            raise ValueError("Only 'linear' kernel is supported with 'sgd' solver.")

        self.C = float(C)

        # ---- Kernel setup ----
        self.kernel_params = kwargs
        self._kernel_map = {
            "linear": kernels.linear_kernel,
            "rbf": kernels.rbf_kernel,
            "polynomial": kernels.polynomial_kernel,
            "sigmoid": kernels.sigmoid_kernel,
        }
        if isinstance(kernel, str):
            # user provided kernel name
            if kernel not in self._kernel_map:
                raise ValueError(f"Kernel '{kernel}' is not supported.")
            self.kernel_name = kernel
            if kernel == "linear":
                self.kernel = self._kernel_map[kernel]
            else:
                self.kernel = partial(self._kernel_map[kernel], **self.kernel_params)
        elif callable(kernel):
            # user provided custom kernel function
            self.kernel_name = "custom"
            # we cannot use pre-vectorized kernel gram matrix in dual form
            # we store the function itself
            self.kernel = kernel
        else:
            raise TypeError("Kernel must be either a string or a callable function.")

        # ---- State Variables ---
        # Primal form variables (for "sgd" solver)
        self.weights = None
        self.bias = None

        # Dual from variables (for "qp" solver)
        self.alphas = None
        self.support_vectors = None
        self.support_vector_indices = None
        self.support_vector_labels = None
        self.b = None  # intercept for dual form

    # =======================================================
    # ===== Public methods for fitting and predicting ======
    # =======================================================
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Fit the SVM model to the training data.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.
            **kwargs: Additional keyword arguments for the SGD solver. Supported arguments:
                - learning_rate (float): Learning rate for SGD. Default is 0.01.
                - n_iters (int): Number of iterations for SGD. Default is 1000.
                - batch_size (int): Mini-batch size for SGD. Default is 32.

        Raises:
            TypeError: If X or y are not numpy arrays.
            ValueError: If the number of samples in X and y are not equal.
            ValueError: If y is not a one-dimensional array.
            ValueError: If the solver is not supported.
        """

        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")

        if len(X) != len(y):
            raise ValueError("Number of samples in X and y must be equal.")

        if len(y.shape) != 1:
            raise ValueError("y must be a one-dimensional array.")

        if self.solver == "sgd":
            self._fit_primal(X, y, **kwargs)
        elif self.solver == "qp":
            self._fit_dual(X, y)
        else:
            # This is already checked in __init__, but its good practice
            raise ValueError(f"Solver '{self.solver}' is not supported.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the input data.

        Args:
            X (np.ndarray): Training data features.

        Raises:
            RuntimeError: If the model is not trained yet.
            ValueError: If the solver is not supported.

        Returns:
            np.ndarray: Predicted labels for the input data.
        """
        if self.solver == "sgd":
            if self.weights is None or self.bias is None:
                raise RuntimeError("Model is not trained yet. Please call 'fit' first.")
            return self._predict_primal(X)
        elif self.solver == "qp":
            if self.alphas is None or self.support_vectors is None or self.b is None:
                raise RuntimeError("Model is not trained yet. Please call 'fit' first.")
            return self._predict_dual(X)
        else:
            raise ValueError(f"Solver '{self.solver}' is not supported.")

    # =======================================================
    # ====== Private methods for primal and dual forms ======
    # =======================================================
    def _fit_primal(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the SVM model using the primal formulation with SGD.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.
            **kwargs: Additional keyword arguments:
                - learning_rate (float): Learning rate for SGD. Default is 0.01.
                - n_iters (int): Number of iterations for SGD. Default is 1000.
                - batch_size (int): Mini-batch size for SGD. Default is 32.

        Raises:
            ValueError: If learning_rate is not positive.
            ValueError: If n_iters is not positive.
            ValueError: If batch_size is not positive.
        """
        learning_rate = kwargs.get("learning_rate", 0.01)
        n_iters = kwargs.get("n_iters", 1000)
        batch_size = kwargs.get("batch_size", 32)

        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if n_iters <= 0:
            raise ValueError("Number of iterations must be positive.")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

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

    def _predict_primal(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")

        linear_output = np.dot(X, self.weights) + self.bias
        # return +1 if linear_output >= 0 else -1
        predictions = np.where(linear_output >= 0, 1, -1)
        return predictions

    def _fit_dual(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the SVM model using the dual formulation.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.
        """
        n_samples, n_features = X.shape
        # compute the kernel matrix for all pair of training samples
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        # Setup the matrices for QP solver
        P = opt.matrix(np.outer(y, y) * K)
        q = opt.matrix(-np.ones(n_samples))
        A = opt.matrix(y, (1, n_samples), "d")  # d is for double precision floating
        b = opt.matrix(0.0)
        G = opt.matrix(np.vstack((-np.identity(n_samples), np.identity(n_samples))))
        h = opt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution["x"])

        sv_mask = (
            alphas > 1e-5
        )  # basic thresholding to identify support vectors. Support vectors have non-zero alphas
        sv_indices = np.where(sv_mask)[0]
        self.support_vector_indices = sv_indices
        self.alphas = alphas[sv_mask]
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]

        # calcuate intercept b
        # b = y_K - sum(alpha_i * y_i * K(x_i, x_K)) for any support vector x_K
        first_sv_index = sv_indices[0]
        self.b = y[first_sv_index] - np.sum(
            self.alphas * self.support_vector_labels * K[sv_mask, first_sv_index]
        )

    def _predict_dual(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")

        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        for i, x_sample in enumerate(X):
            kernel_values = np.array(
                [self.kernel(sv, x_sample) for sv in self.support_vectors]
            )
            y_pred[i] = np.sum(self.alphas * self.support_vector_labels * kernel_values)
        y_pred += self.b
        return np.where(y_pred >= 0, 1, -1)
