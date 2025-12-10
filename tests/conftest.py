"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from sklearn.datasets import make_blobs, make_classification


@pytest.fixture
def random_features():
    "Random 2D dataset for testing."
    X = np.random.randint(0, 100, size=(100, 5)).astype(float)
    return X


@pytest.fixture
def linear_separable_data():
    "Generate a simple linearly separable dataset to test SVM."
    X, y = make_blobs(
        n_samples=200, centers=2, n_features=2, cluster_std=2, random_state=42
    )
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1
    return X, y


@pytest.fixture
def nonlinear_separable_data():
    """Generate a non-linearly separable dataset (XOR-like pattern)."""
    np.random.seed(42)
    # Create XOR pattern
    X = np.random.randn(200, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    y = np.where(y == 0, -1, 1)  # Convert to -1 and 1
    return X, y


@pytest.fixture
def circular_separable_data():
    """Generate circular/radial separable data (inner and outer circles)."""
    np.random.seed(42)
    n_samples = 200
    # Inner circle
    r_inner = np.random.uniform(0, 2, n_samples // 2)
    theta_inner = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X_inner = np.column_stack(
        [r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)]
    )
    y_inner = np.ones(n_samples // 2) * -1

    # Outer circle
    r_outer = np.random.uniform(3, 5, n_samples // 2)
    theta_outer = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X_outer = np.column_stack(
        [r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)]
    )
    y_outer = np.ones(n_samples // 2)

    X = np.vstack([X_inner, X_outer])
    y = np.hstack([y_inner, y_outer])
    return X, y


@pytest.fixture
def multiclass_data():
    """Generate a multiclass dataset (3 classes)."""
    X, y = make_blobs(
        n_samples=300, centers=3, n_features=2, cluster_std=1.5, random_state=42
    )
    return X, y


@pytest.fixture
def regression_data():
    """Generate a simple regression dataset."""
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2.5 * X.flatten() + 1.0 + np.random.randn(100) * 0.5
    return X, y

@pytest.fixture
def data():
    """Generate synthetic classification data."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
        n_classes=2,
    )
    y_svm = np.where(y == 0, -1, 1)
    return X, y_svm

