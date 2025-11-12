"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from sklearn.datasets import make_blobs


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
