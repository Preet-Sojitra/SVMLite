"""Test for metrics."""

import numpy as np
from svmlite.metrics import accuracy_score
from sklearn.metrics import accuracy_score as sklearn_accuracy_score


def test_accuracy_score() -> None:
    """Test accuracy_score function."""
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1])

    # Test normalized accuracy
    acc = accuracy_score(y_true, y_pred, normalize=True)
    sklearn_acc = sklearn_accuracy_score(y_true, y_pred)
    assert np.isclose(acc, sklearn_acc), f"Expected {sklearn_acc}, got {acc}"

    # Test unnormalized accuracy
    acc_count = accuracy_score(y_true, y_pred, normalize=False)
    sklearn_acc_count = int(sklearn_accuracy_score(y_true, y_pred) * len(y_true))
    assert (
        acc_count == sklearn_acc_count
    ), f"Expected {sklearn_acc_count}, got {acc_count}"
