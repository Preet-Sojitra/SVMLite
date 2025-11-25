"""Kernel functions for SVM."""

import numpy as np


def linear_kernel(x1: float, x2: float) -> float:
    """
    Compute the linear kernel between two vectors.
    Args:
        x1 (float): Data point 1.
        x2 (float): Data point 2.

    Returns:
        float: Kernel value between x1 and x2.
    """

    return np.dot(x1, x2)


def rbf_kernel(x1: float, x2: float, gamma: float = 0.1) -> float:
    """
    Compute the RBF (Gaussian) kernel between two vectors.

    Args:
        x1 (float): Data point 1.
        x2 (float): Data point 2.
        gamma (float, optional): Kernel parameter. Defaults to 0.1. Higher gamma means influence of single support vector is close, lower means far. Smaller gamma leads to smoother decision boundary and larger gamma leads to more complex decision boundary.

    Returns:
        float: Kernel value between x1 and x2.
    """
    distance = np.linalg.norm(x1 - x2) ** 2
    return np.exp(-gamma * distance)


def polynomial_kernel(x1: float, x2: float, degree: int = 3, coef0: float = 1) -> float:
    """
    Compute the polynomial kernel between two vectors.

    Args:
        x1 (float): Data point 1.
        x2 (float): Data point 2.
        degree (int, optional): Degree of the polynomial. Defaults to 3.
        coef0 (float, optional): Independent term in polynomial kernel. Defaults to 1.

    Returns:
        float: Kernel value between x1 and x2.
    """
    return (np.dot(x1, x2) + coef0) ** degree


def sigmoid_kernel(
    x1: float, x2: float, alpha: float = 0.01, coef0: float = 0
) -> float:
    """
    Compute the sigmoid kernel between two vectors.

    Args:
        x1 (float): Data point 1.
        x2 (float): Data point 2.
        alpha (float, optional): Slope parameter. Defaults to 0.01.
        coef0 (float, optional): Independent term in sigmoid kernel. Defaults to 0.

    Returns:
        float: Kernel value between x1 and x2.
    """
    return np.tanh(alpha * np.dot(x1, x2) + coef0)
