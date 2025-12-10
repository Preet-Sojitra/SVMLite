"""Support Vector Machine (SVM) implementation."""

from functools import partial
import numpy as np
import cvxopt as opt
import cvxopt.solvers as solvers
from . import kernels


class SVCLite:
    def __init__(
        self,
        C: float = 1.0,
        solver: str = "SGD",
        kernel: str = "linear",
        multiclass_strategy: str = None,
        **kwargs,
    ):
        """
        Initialize the SVM model.

        Args:
            C (float, optional): Regularization parameter. Defaults to 1.0.
            solver (str, optional): Solver to use for optimization. Defaults to "SGD". Available options: "SGD", "QP", "SMO".
            kernel (str, optional): Kernel function to use. Defaults to "linear".
            multiclass_strategy (str, optional): Strategy for multiclass classification. Options: 'ova' (One-vs-All), 'ovo' (One-vs-One), or None for binary. Defaults to None.
            **kwargs: Additional keyword arguments for kernel functions.

        Raises:
            ValueError: If C is not positive.
        """
        self.solver = solver.lower()
        if C <= 0:
            raise ValueError("Regularization parameter C must be positive.")
        if self.solver not in ["sgd", "qp", "smo"]:
            raise ValueError("Solver must be either 'SGD', 'QP', or 'SMO'.")
        if self.solver == "sgd" and isinstance(kernel, str) and kernel != "linear":
            raise ValueError("Only 'linear' kernel is supported with 'sgd' solver.")
        if multiclass_strategy not in [None, "ova", "ovo"]:
            raise ValueError("Multiclass strategy must be 'ova', 'ovo', or None.")

        self.C = float(C)
        self.multiclass_strategy = multiclass_strategy

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

        # For multiclass strategy
        self.classes_ = None # holds the unique class labels
        # for ova it will be list of K models, indexed by class label. For ovo it will be tuple ((class_i, class_j), model)
        self._binary_classifiers = []

    # =======================================================
    # ===== Public methods for fitting and prediction =======
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
            For SMO solver:
                - tol (float): Numerical tolerance. Default is 1e-3.
                - max_passes (int): Maximum number of passes without changes. Default is 5.

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

        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)

        if num_classes <= 2 or self.multiclass_strategy is None:
            # Binary or forced binary
            if num_classes > 2 and self.multiclass_strategy is None:
                raise ValueError(
                    "Multiclass data detected but no multiclass_strategy specified."
                )
            # ensure y is -1/+1 for binary
            if num_classes == 2:
                y = np.where(y == self.classes_[0], -1, 1)
            if self.solver == "sgd":
                self._fit_primal(X, y, **kwargs)
            elif self.solver == "qp":
                self._fit_dual(X, y)
            elif self.solver == "smo":
                self._fit_smo(X, y, **kwargs)
        else:
            # multiclass
            if self.multiclass_strategy == "ova":
                self._fit_ova(X, y, **kwargs)
            elif self.multiclass_strategy == "ovo":
                self._fit_ovo(X, y, **kwargs)
    
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

        if len(self._binary_classifiers) == 0:
            # Binary classification
            if self.solver == "sgd":
                if self.weights is None or self.bias is None:
                    raise RuntimeError(
                        "Model is not trained yet. Please call 'fit' first."
                    )
                return self._predict_primal(X)
            elif self.solver in ["qp", "smo"]:
                if (
                    self.alphas is None
                    or self.support_vectors is None
                    or self.b is None
                ):
                    raise RuntimeError(
                        "Model is not trained yet. Please call 'fit' first."
                    )
                return self._predict_dual(X)
            else:
                raise ValueError(f"Solver '{self.solver}' is not supported.")
        else:
            # Multiclass classification
            if self.multiclass_strategy == "ova":
                return self._predict_ova(X)
            elif self.multiclass_strategy == "ovo":
                return self._predict_ovo(X)

    # ============================================================
    # === Private methods for fitting and prediction multiclass ===
    # ============================================================

    def _fit_ova(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the model using One-vs-All (OvA) strategy.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.
            **kwargs: Additional keyword arguments for the SGD solver or SMO solver.
        """
        self._binary_classifiers = []
        for idx, cls in enumerate(self.classes_):
            # create binary labels for current class vs all
            y_binary = np.where(y == cls, 1, -1)
            sub_model = SVCLite(
                C=self.C,
                solver=self.solver,
                kernel=(
                    self.kernel if isinstance(self.kernel, str) else self.kernel_name
                    # logic is: if custom kernel provided then self.kernel_name will be "custom" else if
                    # it is str then self.kernel holds the kernel function directly
                ),
                **self.kernel_params,
            )
            sub_model.fit(X, y_binary, **kwargs)
            self._binary_classifiers.append((cls, sub_model))

    def _fit_ovo(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the model using One-vs-One (OvO) strategy.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.
            **kwargs: Additional keyword arguments for the SGD solver or SMO solver.
        """
        self._binary_classifiers = []
        # for each pair of classes, create a binary classifier
        for i in range(len(self.classes_)):
            for j in range(i + 1, len(self.classes_)):
                cls_i, cls_j = self.classes_[i], self.classes_[j]
                # mask is just the indices where y is cls_i or cls_j because are creating binary classifier for each pair of classes
                mask = (y == cls_i) | (y == cls_j)
                X_pair = X[mask]
                y_pair = np.where(y[mask] == cls_i, 1, -1)
                sub_model = SVCLite(
                    C=self.C,
                    solver=self.solver,
                    kernel=(
                        self.kernel
                        if isinstance(self.kernel, str)
                        else self.kernel_name
                    ),
                    **self.kernel_params,
                )
                sub_model.fit(X_pair, y_pair, **kwargs)
                self._binary_classifiers.append(((cls_i, cls_j), sub_model))

    def _predict_ova(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the input data using One-vs-All (OvA) strategy.
        
        Prediction strategy:
            1. For each binary classifier, compute the decision function.
            2. The class with the highest decision function value is predicted.

        Args:
            X (np.ndarray): Training data features.

        Returns:
            np.ndarray: Predicted labels for the input data.
        """
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes_)))
        for idx, (cls, sub_model) in enumerate(self._binary_classifiers):
            scores[:, idx] = sub_model._decision_function(X)
        preds = self.classes_[np.argmax(scores, axis=1)]
        return preds

    def _predict_ovo(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the input data using One-vs-One (OvO) strategy.
        
        Prediction strategy:
            1. For each binary classifier, predict the labels for the input data.
            2. Count the number of votes for each class.
            3. The class with the highest number of votes is predicted.

        Args:
            X (np.ndarray): Training data features.

        Returns:
            np.ndarray: Predicted labels for the input data.
        """

        n_samples = X.shape[0]
        votes = np.zeros((n_samples, len(self.classes_)))
        for (cls_i, cls_j), sub_model in self._binary_classifiers:
            preds = sub_model.predict(X)
            # get the indices of the classes in self.classes_
            i_idx = np.where(self.classes_ == cls_i)[0][0]
            j_idx = np.where(self.classes_ == cls_j)[0][0]
            for s in range(n_samples):
                if preds[s] == 1:
                    votes[s, i_idx] += 1
                else:
                    votes[s, j_idx] += 1
        preds = self.classes_[np.argmax(votes, axis=1)]
        return preds

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function values for the input data.

        Args:
            X (np.ndarray): Input data features.
        Returns:
            np.ndarray: returns the raw score for binary classification. For multiclass, raise error, as it is strategy specific, use predict instead.
        """
        if len(self._binary_classifiers) > 0:
            raise RuntimeError(
                "Decision function is for binary classification only. For multiclass, use 'predict' method instead."
            )
        if self.solver == "sgd":
            if self.weights is None or self.bias is None:
                raise RuntimeError("Model is not trained yet. Please call 'fit' first.")
            return np.dot(X, self.weights) + self.bias
        else:
            if self.alphas is None or self.support_vectors is None or self.b is None:
                raise RuntimeError("Model is not trained yet. Please call 'fit' first.")
            n_samples = X.shape[0]
            y_pred = np.zeros(n_samples)

            for i, x_sample in enumerate(X):
                kernel_values = np.array(
                    [self.kernel(sv, x_sample) for sv in self.support_vectors]
                )
                y_pred[i] = np.sum(
                    self.alphas * self.support_vector_labels * kernel_values
                )
            y_pred += self.b
            return y_pred

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
        """
        Predict the labels for the input data using the primal formulation.

        Args:
            X (np.ndarray): Input data features.

        Returns:
            np.ndarray: Predicted labels for the input data.
        """
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
        """
        Predict the labels for the input data using the dual formulation.

        Args:
            X (np.ndarray): Input data features.

        Returns:
            np.ndarray: Predicted labels for the input data.
        """
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

    def _fit_smo(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit the SVM model using the simplified SMO algorithm.

        Args:
            X (np.ndarray): Training data features
            y (np.ndarray): Training data labels
            **kwargs: Additional keyword arguments:
                - tol (float): Numerical tolerance. Default is 1e-3.
                - max_passes (int): Maximum number of passes without changes. Default is 5.
        """
        tol = kwargs.get("tol", 1e-3)
        max_passes = kwargs.get("max_passes", 5)

        if tol <= 0:
            raise ValueError("Tolerance must be positive.")
        if max_passes <= 0:
            raise ValueError("Maximum passes must be positive.")

        n_samples = X.shape[0]
        self.alphas = np.zeros(n_samples)
        self.b = 0.0
        passes = 0

        while passes < max_passes:
            num_changed_alphas = 0
            for i in range(n_samples):
                # Compute f(x_i)
                kernel_vals_i = np.array(
                    [self.kernel(X[k], X[i]) for k in range(n_samples)]
                )
                f_i = np.sum(self.alphas * y * kernel_vals_i) + self.b
                E_i = f_i - y[i]  # Error for sample i

                # Check if alpha_i violates KKT conditions
                if (
                    y[i] * E_i < -tol
                    and self.alphas[i] < self.C
                    or (y[i] * E_i > tol and self.alphas[i] > 0)
                ):
                    # select j!=i randomly
                    j = np.random.choice([k for k in range(n_samples) if k != i])

                    # compute f(x_j)
                    kernel_vals_j = np.array(
                        [self.kernel(X[k], X[j]) for k in range(n_samples)]
                    )
                    f_j = np.sum(self.alphas * y * kernel_vals_j) + self.b
                    E_j = f_j - y[j]  # Error for sample j

                    # save old alphas
                    alpha_i_old = self.alphas[i].copy()
                    alpha_j_old = self.alphas[j].copy()

                    # compute L and H
                    if y[i] == y[j]:
                        L = max(0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)
                    else:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    if L == H:
                        continue

                    # compute eta
                    eta = (
                        2.0 * self.kernel(X[i], X[j])
                        - self.kernel(X[i], X[i])
                        - self.kernel(X[j], X[j])
                    )
                    if eta >= 0:
                        continue

                    # update alpha_j
                    self.alphas[j] = alpha_j_old - y[j] * (E_i - E_j) / eta

                    # clip alpha_j
                    if self.alphas[j] > H:
                        self.alphas[j] = H
                    elif self.alphas[j] < L:
                        self.alphas[j] = L

                    # check if alpha_j changed significantly
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    # update alpha_i
                    self.alphas[i] = alpha_i_old + y[i] * y[j] * (
                        alpha_j_old - self.alphas[j]
                    )

                    # compute b1 and b2
                    b1 = (
                        self.b
                        - E_i
                        - y[i]
                        * (self.alphas[i] - alpha_i_old)
                        * self.kernel(X[i], X[i])
                        - y[j]
                        * (self.alphas[j] - alpha_j_old)
                        * self.kernel(X[i], X[j])
                    )
                    b2 = (
                        self.b
                        - E_j
                        - y[i]
                        * (self.alphas[i] - alpha_i_old)
                        * self.kernel(X[i], X[j])
                        - y[j]
                        * (self.alphas[j] - alpha_j_old)
                        * self.kernel(X[j], X[j])
                    )

                    # update b
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        # extract support vectors
        sv_mask = self.alphas > 1e-5
        self.support_vector_indices = np.where(sv_mask)[0]
        self.alphas = self.alphas[sv_mask]
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]
