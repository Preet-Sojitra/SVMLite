"""Tests for SVM implementation."""

import numpy as np
from svmlite.utils import StandardScalerLite
from svmlite.svm import SVCLite
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class TestSVCLite:
    """Test SVCLite implementation against sklearn's SVC."""

    def test_svm_fit_and_predict_hardmargin(self, linear_separable_data):
        # standardize the data
        X, y = linear_separable_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # train SVCLite model
        svm_lite = SVCLite(C=10000)  # hard margin for linearly separable data
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

    def test_svm_fit_and_predict_softmargin(self, linear_separable_data):
        # standardize the data
        X, y = linear_separable_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScalerLite()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # train SVCLite model
        svm_lite = SVCLite(C=1)
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

    def test_weights_and_bias(self, linear_separable_data):
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
