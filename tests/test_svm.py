"""Tests for SVM implementation."""

import numpy as np
from svmlite.utils import StandardScalerLite
from svmlite.svm import SVCLite, SVRLite
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


class TestSVCLiteSGD:
    """Test SVCLite implementation against sklearn's SVC using SGD solver."""

    def test_sgd_svm_fit_and_predict_hardmargin(self, linear_separable_data):
        # standardize the data
        X, y = linear_separable_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # train SVCLite model
        svm_lite = SVCLite(
            C=10000, solver="SGD", kernel="linear"
        )  # hard margin for linearly separable data
        svm_lite.fit(
            X_train_scaled, y_train, learning_rate=0.01, n_iters=1000, batch_size=16
        )
        predictions_lite = svm_lite.predict(X_test_scaled)
        accuracy_lite = accuracy_score(y_test, predictions_lite)

        # train sklearn SVC model
        svm_sklearn = SVC(C=10000, kernel="linear")
        svm_sklearn.fit(X_train_scaled, y_train)
        predictions_sklearn = svm_sklearn.predict(X_test_scaled)
        accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

        # compare accuracies
        assert (
            abs(accuracy_lite - accuracy_sklearn) < 0.05
        ), f"SVCLite accuracy {accuracy_lite} differs significantly from sklearn SVC accuracy {accuracy_sklearn}"

    def test_sgd_svm_fit_and_predict_softmargin(self, linear_separable_data):
        # standardize the data
        X, y = linear_separable_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # train SVCLite model
        svm_lite = SVCLite(C=1, solver="SGD", kernel="linear")  # soft margin
        svm_lite.fit(
            X_train_scaled, y_train, learning_rate=0.01, n_iters=1000, batch_size=16
        )
        predictions_lite = svm_lite.predict(X_test_scaled)
        accuracy_lite = accuracy_score(y_test, predictions_lite)

        # train sklearn SVC model
        svm_sklearn = SVC(C=1, kernel="linear")
        svm_sklearn.fit(X_train_scaled, y_train)
        predictions_sklearn = svm_sklearn.predict(X_test_scaled)
        accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

        # compare accuracies
        assert (
            abs(accuracy_lite - accuracy_sklearn) < 0.05
        ), f"SVCLite accuracy {accuracy_lite} differs significantly from sklearn SVC accuracy {accuracy_sklearn}"

    def test_sgd_weights_and_bias(self, linear_separable_data):
        # standardize the data
        X, y = linear_separable_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # train SVCLite model
        svm_lite = SVCLite(C=10)
        svm_lite.fit(
            X_train_scaled, y_train, learning_rate=0.01, n_iters=1000, batch_size=16
        )

        # train sklearn SVC model
        svm_sklearn = SVC(C=10, kernel="linear")
        svm_sklearn.fit(X_train_scaled, y_train)

        # we should not be comparing the weight and basis directly because they can differ by a scaling factor. Morever sklearn's implementation uses different optimization techniques. What matters to us is the direction of the weights vector and the position of the hyperplane defined by the bias. If they are close enough, it indicates that both models have learned similar decision boundaries.

        w_lite = svm_lite.weights
        w_sklearn = svm_sklearn.coef_.flatten()

        # normalize then to get direction (divide by norm)
        w_lite_normalized = w_lite / np.linalg.norm(w_lite)
        w_sklearn_normalized = w_sklearn / np.linalg.norm(w_sklearn)

        # compare weights
        np.testing.assert_allclose(
            w_lite_normalized,
            w_sklearn_normalized,
            rtol=0,
            atol=5e-2,
            err_msg="Normalized weights (direction) from SVCLite do not match sklearn's SVC",
        )

        # similarly for bias
        bias_lite_normalized = svm_lite.bias / np.linalg.norm(w_lite)
        bias_sklearn_normalized = svm_sklearn.intercept_[0] / np.linalg.norm(w_sklearn)

        # compare bias
        np.testing.assert_allclose(
            bias_lite_normalized,
            bias_sklearn_normalized,
            rtol=0,
            atol=0.08,
            err_msg="Normalized bias from SVCLite does not match normalized bias from sklearn's SVC",
        )


class TestSVCLiteQPSolver:
    """Test SVCLite using the QP solver."""

    def test_qp_linear_kernel_accuracy(self, linear_separable_data):
        """Test QP solver with a linear kernel on linearly separable data."""
        X, y = linear_separable_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train SVCLite with QP solver
        svm_lite = SVCLite(C=1.0, solver="QP", kernel="linear")
        svm_lite.fit(X_train_scaled, y_train)
        predictions_lite = svm_lite.predict(X_test_scaled)
        accuracy_lite = accuracy_score(y_test, predictions_lite)

        # Train sklearn SVC
        svm_sklearn = SVC(C=1.0, kernel="linear")
        svm_sklearn.fit(X_train_scaled, y_train)
        predictions_sklearn = svm_sklearn.predict(X_test_scaled)
        accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

        # The results should be very close as both use precise solvers
        assert abs(accuracy_lite - accuracy_sklearn) < 0.05

    def test_qp_support_vectors(self, linear_separable_data):
        """
        Test if the set of support vectors found by SVCLite is similar to sklearn's.
        The exact alpha values might differ slightly due to solver tolerances,
        but the set of points chosen as support vectors should be nearly identical.
        """
        X, y = linear_separable_data
        scaler = StandardScalerLite()
        X_scaled = scaler.fit_transform(X)

        # Train SVCLite
        svm_lite = SVCLite(C=1.0, solver="QP", kernel="linear")
        svm_lite.fit(X_scaled, y)

        # Train sklearn SVC
        svm_sklearn = SVC(C=1.0, kernel="linear")
        svm_sklearn.fit(X_scaled, y)

        # Get the indices of the support vectors from the original training data
        lite_sv_indices = svm_lite.support_vector_indices
        sklearn_sv_indices = svm_sklearn.support_

        # The order might not be the same, so we compare them as sets
        set_lite = set(lite_sv_indices)
        set_sklearn = set(sklearn_sv_indices)

        # Check how similar the sets are using Jaccard similarity (intersection over union)
        # This is robust to minor differences where one model might pick one extra point.
        intersection_size = len(set_lite.intersection(set_sklearn))
        union_size = len(set_lite.union(set_sklearn))

        similarity = intersection_size / union_size if union_size > 0 else 1

        print(f"\nSupport Vector Set Similarity (Jaccard): {similarity:.2f}")
        print(
            f"SVCLite found {len(set_lite)} SVs. Sklearn found {len(set_sklearn)} SVs."
        )

        assert (
            similarity > 0.9
        ), "The set of support vectors should be at least 90% similar."


class TestSVCLiteKernels:
    """Test SVCLite with different kernel functions."""

    def test_qp_rbf_kernel_accuracy(self, nonlinear_separable_data):
        """Test QP solver with RBF kernel on non-linearly separable data."""
        X, y = nonlinear_separable_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train SVCLite with RBF kernel
        svm_lite = SVCLite(C=1.0, solver="QP", kernel="rbf", gamma=0.5)
        svm_lite.fit(X_train_scaled, y_train)
        predictions_lite = svm_lite.predict(X_test_scaled)
        accuracy_lite = accuracy_score(y_test, predictions_lite)

        # Train sklearn SVC with RBF kernel
        svm_sklearn = SVC(C=1.0, kernel="rbf", gamma=0.5)
        svm_sklearn.fit(X_train_scaled, y_train)
        predictions_sklearn = svm_sklearn.predict(X_test_scaled)
        accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

        print(
            f"\nRBF Kernel - SVCLite accuracy: {accuracy_lite:.3f}, sklearn accuracy: {accuracy_sklearn:.3f}"
        )

        # Both should achieve reasonable accuracy on non-linear data
        assert (
            accuracy_lite > 0.7
        ), f"SVCLite RBF kernel accuracy {accuracy_lite} is too low"
        assert abs(accuracy_lite - accuracy_sklearn) < 0.1

    def test_qp_polynomial_kernel_accuracy(self, nonlinear_separable_data):
        """Test QP solver with polynomial kernel on non-linearly separable data."""
        X, y = nonlinear_separable_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train SVCLite with polynomial kernel
        svm_lite = SVCLite(C=1.0, solver="QP", kernel="polynomial", degree=3, coef0=1)
        svm_lite.fit(X_train_scaled, y_train)
        predictions_lite = svm_lite.predict(X_test_scaled)
        accuracy_lite = accuracy_score(y_test, predictions_lite)

        # Train sklearn SVC with polynomial kernel
        svm_sklearn = SVC(C=1.0, kernel="poly", degree=3, coef0=1)
        svm_sklearn.fit(X_train_scaled, y_train)
        predictions_sklearn = svm_sklearn.predict(X_test_scaled)
        accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

        print(
            f"\nPolynomial Kernel - SVCLite accuracy: {accuracy_lite:.3f}, sklearn accuracy: {accuracy_sklearn:.3f}"
        )

        assert (
            accuracy_lite > 0.65
        ), f"SVCLite polynomial kernel accuracy {accuracy_lite} is too low"
        assert abs(accuracy_lite - accuracy_sklearn) < 0.15

    def test_qp_sigmoid_kernel_accuracy(self, nonlinear_separable_data):
        """Test QP solver with sigmoid kernel on non-linearly separable data."""
        X, y = nonlinear_separable_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train SVCLite with sigmoid kernel
        svm_lite = SVCLite(C=1.0, solver="QP", kernel="sigmoid", alpha=0.01, coef0=0)
        svm_lite.fit(X_train_scaled, y_train)
        predictions_lite = svm_lite.predict(X_test_scaled)
        accuracy_lite = accuracy_score(y_test, predictions_lite)

        # Train sklearn SVC with sigmoid kernel
        svm_sklearn = SVC(C=1.0, kernel="sigmoid", gamma=0.01, coef0=0)
        svm_sklearn.fit(X_train_scaled, y_train)
        predictions_sklearn = svm_sklearn.predict(X_test_scaled)
        accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

        print(
            f"\nSigmoid Kernel - SVCLite accuracy: {accuracy_lite:.3f}, sklearn accuracy: {accuracy_sklearn:.3f}"
        )

        # Sigmoid kernel can be less stable, so we're more lenient
        assert (
            accuracy_lite > 0.4
        ), f"SVCLite sigmoid kernel accuracy {accuracy_lite} is too low"

    def test_qp_custom_kernel_rbf(self, circular_separable_data):
        """Test QP solver with a custom RBF kernel function."""
        X, y = circular_separable_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define custom RBF kernel
        def custom_rbf(x1, x2, gamma=0.1):
            distance = np.linalg.norm(x1 - x2) ** 2
            return np.exp(-gamma * distance)

        # Train SVCLite with custom kernel
        svm_lite = SVCLite(C=1.0, solver="QP", kernel=custom_rbf)
        svm_lite.fit(X_train_scaled, y_train)
        predictions_lite = svm_lite.predict(X_test_scaled)
        accuracy_lite = accuracy_score(y_test, predictions_lite)

        # Train with built-in RBF for comparison
        svm_builtin = SVCLite(C=1.0, solver="QP", kernel="rbf", gamma=0.1)
        svm_builtin.fit(X_train_scaled, y_train)
        predictions_builtin = svm_builtin.predict(X_test_scaled)
        accuracy_builtin = accuracy_score(y_test, predictions_builtin)

        print(
            f"\nCustom RBF Kernel - accuracy: {accuracy_lite:.3f}, Built-in RBF accuracy: {accuracy_builtin:.3f}"
        )

        # Custom kernel should perform well on circular data
        assert (
            accuracy_lite > 0.85
        ), f"Custom kernel accuracy {accuracy_lite} is too low"
        # Should match built-in RBF closely
        assert abs(accuracy_lite - accuracy_builtin) < 0.05

    def test_qp_custom_kernel_polynomial(self, nonlinear_separable_data):
        """Test QP solver with a custom polynomial kernel function."""
        X, y = nonlinear_separable_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define custom polynomial kernel
        def custom_poly(x1, x2, degree=2, coef0=1):
            return (np.dot(x1, x2) + coef0) ** degree

        # Train SVCLite with custom kernel
        svm_lite = SVCLite(C=1.0, solver="QP", kernel=custom_poly)
        svm_lite.fit(X_train_scaled, y_train)
        predictions_lite = svm_lite.predict(X_test_scaled)
        accuracy_lite = accuracy_score(y_test, predictions_lite)

        print(f"\nCustom Polynomial Kernel - accuracy: {accuracy_lite:.3f}")

        # Should achieve reasonable accuracy with polynomial kernel
        assert (
            accuracy_lite > 0.65
        ), f"Custom polynomial kernel accuracy {accuracy_lite} is too low"


class TestSVCLiteSMOSolver:
    """Test SVCLite using the SMO solver."""

    def test_smo_linear_kernel_accuracy(self, linear_separable_data):
        """Test SMO solver with a linear kernel on linearly separable data."""
        X, y = linear_separable_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train SVCLite with SMO solver
        svm_lite = SVCLite(C=1.0, solver="SMO", kernel="linear")
        svm_lite.fit(X_train_scaled, y_train)
        predictions_lite = svm_lite.predict(X_test_scaled)
        accuracy_lite = accuracy_score(y_test, predictions_lite)

        # Train sklearn SVC
        svm_sklearn = SVC(C=1.0, kernel="linear")
        svm_sklearn.fit(X_train_scaled, y_train)
        predictions_sklearn = svm_sklearn.predict(X_test_scaled)
        accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

        # The results should be very close as both use precise solvers
        assert abs(accuracy_lite - accuracy_sklearn) < 0.05

    def test_smo_rbf_kernel_accuracy(self, circular_separable_data):
        """Test SMO solver with RBF kernel on non-linearly separable data."""
        X, y = circular_separable_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train SVCLite with RBF kernel
        svm_lite = SVCLite(C=1.0, solver="SMO", kernel="rbf", gamma=0.5)
        svm_lite.fit(X_train_scaled, y_train)
        predictions_lite = svm_lite.predict(X_test_scaled)
        accuracy_lite = accuracy_score(y_test, predictions_lite)

        # Train sklearn SVC with RBF kernel
        svm_sklearn = SVC(C=1.0, kernel="rbf", gamma=0.5)
        svm_sklearn.fit(X_train_scaled, y_train)
        predictions_sklearn = svm_sklearn.predict(X_test_scaled)
        accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

        print(
            f"\nRBF Kernel (SMO) - SVCLite accuracy: {accuracy_lite:.3f}, sklearn accuracy: {accuracy_sklearn:.3f}"
        )

        # Both should achieve reasonable accuracy on non-linear data
        assert (
            accuracy_lite > 0.7
        ), f"SVCLite RBF kernel (SMO) accuracy {accuracy_lite} is too low"
        assert abs(accuracy_lite - accuracy_sklearn) < 0.1

    def test_smo_polynomial_kernel_accuracy(self, circular_separable_data):
        """Test SMO solver with polynomial kernel on non-linearly separable data."""
        X, y = circular_separable_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train SVCLite with polynomial kernel
        svm_lite = SVCLite(C=1.0, solver="SMO", kernel="polynomial", degree=3, coef0=1)
        svm_lite.fit(X_train_scaled, y_train)
        predictions_lite = svm_lite.predict(X_test_scaled)
        accuracy_lite = accuracy_score(y_test, predictions_lite)

        # Train sklearn SVC with polynomial kernel
        svm_sklearn = SVC(C=1.0, kernel="poly", degree=3, coef0=1)
        svm_sklearn.fit(X_train_scaled, y_train)
        predictions_sklearn = svm_sklearn.predict(X_test_scaled)
        accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

        print(
            f"\nPolynomial Kernel (SMO) - SVCLite accuracy: {accuracy_lite:.3f}, sklearn accuracy: {accuracy_sklearn:.3f}"
        )

        assert (
            accuracy_lite > 0.65
        ), f"SVCLite polynomial kernel (SMO) accuracy {accuracy_lite} is too low"
        assert abs(accuracy_lite - accuracy_sklearn) < 0.15


class TestMulticlassSVM:
    """Test SVCLite multiclass strategies (OVA and OVO)."""

    def test_ova_accuracy(self, multiclass_data):
        """Test One-vs-All strategy on multiclass data."""
        X, y = multiclass_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train SVCLite with OVA strategy
        svm_lite = SVCLite(
            C=1.0, solver="SGD", kernel="linear", multiclass_strategy="ova"
        )
        svm_lite.fit(X_train_scaled, y_train, n_iters=1000)
        predictions_lite = svm_lite.predict(X_test_scaled)
        accuracy_lite = accuracy_score(y_test, predictions_lite)

        # Train sklearn SVC (default is ovo, but we can compare accuracy)
        svm_sklearn = SVC(C=1.0, kernel="linear", decision_function_shape="ovr")
        svm_sklearn.fit(X_train_scaled, y_train)
        predictions_sklearn = svm_sklearn.predict(X_test_scaled)
        accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

        print(
            f"\nOVA Strategy - SVCLite accuracy: {accuracy_lite:.3f}, sklearn accuracy: {accuracy_sklearn:.3f}"
        )

        assert (
            accuracy_lite > 0.8
        ), f"SVCLite OVA accuracy {accuracy_lite} is too low"
        assert abs(accuracy_lite - accuracy_sklearn) < 0.1

    def test_ovo_accuracy(self, multiclass_data):
        """Test One-vs-One strategy on multiclass data."""
        X, y = multiclass_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train SVCLite with OVO strategy
        svm_lite = SVCLite(
            C=1.0, solver="SGD", kernel="linear", multiclass_strategy="ovo"
        )
        svm_lite.fit(X_train_scaled, y_train, n_iters=1000)
        predictions_lite = svm_lite.predict(X_test_scaled)
        accuracy_lite = accuracy_score(y_test, predictions_lite)

        # Train sklearn SVC (default is ovo)
        svm_sklearn = SVC(C=1.0, kernel="linear", decision_function_shape="ovo")
        svm_sklearn.fit(X_train_scaled, y_train)
        predictions_sklearn = svm_sklearn.predict(X_test_scaled)
        accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

        print(
            f"\nOVO Strategy - SVCLite accuracy: {accuracy_lite:.3f}, sklearn accuracy: {accuracy_sklearn:.3f}"
        )

        assert (
            accuracy_lite > 0.8
        ), f"SVCLite OVO accuracy {accuracy_lite} is too low"
        assert abs(accuracy_lite - accuracy_sklearn) < 0.1


class TestSVRLiteSGD:
    """Test SVRLite SGD implementation."""

    def test_svrlite_sgd_fit_and_predict(self, regression_data):
        """Test SVRLite SGD fit and predict on simple regression data."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Standardize
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train SVRLite SGD
        svrlite_sgd = SVRLite(epsilon=0.1, C=1.0, learning_rate=0.01, n_iters=10000)
        svrlite_sgd.fit(X_train_scaled, y_train)
        
        # Predict
        predictions = svrlite_sgd.predict(X_test_scaled)
        mse = mean_squared_error(y_test, predictions)
        
        print(f"SVRLite SGD MSE: {mse}")

        # Train sklearn SVR
        svr_sklearn = SVR(epsilon=0.1, C=1.0, max_iter=2000, kernel="linear")
        svr_sklearn.fit(X_train_scaled, y_train)
        predictions_sklearn = svr_sklearn.predict(X_test_scaled)
        mse_sklearn = mean_squared_error(y_test, predictions_sklearn)
        
        print(f"Sklearn SVR MSE: {mse_sklearn}")

        assert mse < 0.5, f"SVRLite SGD MSE {mse} is too high"
        assert abs(mse - mse_sklearn) < 0.05 , f"SVRLite SGD MSE {mse} is too high"

        # assert the parameters are the nearly same with some tolerance
        np.testing.assert_allclose(svrlite_sgd.weights, svr_sklearn.coef_.flatten(), rtol=1e-03, atol=0.1)
        np.testing.assert_allclose(svrlite_sgd.bias, svr_sklearn.intercept_.flatten(), rtol=1e-03, atol=0.1)
        
        
