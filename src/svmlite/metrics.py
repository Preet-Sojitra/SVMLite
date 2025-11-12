"""Module for evaluation metrics."""

import numpy as np


def accuracy_score(
    y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True
) -> float | int:
    """
    Compute the accuracy of predictions.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        normalize (bool, optional): If True, return the fraction of correctly classified samples. Otherwise, return the number of correctly classified samples. Defaults to True.

    Raises:
        TypeError: If y_true or y_pred are not numpy arrays.
        ValueError: If y_true and y_pred do not have the same shape.

    Returns:
        float | int: Accuracy as a fraction if normalize is True, otherwise the count of correct predictions.
    """

    # preliminary type checking
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_true and y_pred must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    correct_predictions = np.sum(y_true == y_pred)
    if normalize:
        return correct_predictions / len(y_true)
    else:
        return correct_predictions
