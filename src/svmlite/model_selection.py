"""
Module for model selection and hyperparameter tuning.
"""

import copy
import itertools
from typing import Any, Dict, List, Optional, Union
import numpy as np
from .metrics import accuracy_score


# this class is main engine for grid search.
class ParameterGrid:
    """
    Grid of parameters with a discrete number of values for each.

    Args:
        param_grid (dict): Dictionary with parameters names (string) as keys
            and lists of parameter settings to try as values.
    """

    def __init__(self, param_grid: Dict[str, List[Any]]):
        self.param_grid = param_grid

    def __iter__(self):
        """
        Iterate over the points in the grid.

        Yields:
            dict: Params dictionary.
        """
        # Sort the keys of a dictionary, for reproducibility
        items = sorted(self.param_grid.items())
        if not items:
            yield {}
        else:
            keys, values = zip(*items) # unzip the dictionary
            for v in itertools.product(*values): # generate all possible combinations aka cartesian product
                yield dict(zip(keys, v))

    def __len__(self):
        """
        Number of points on the grid.
        """
        product = 1
        for v in self.param_grid.values():
            product *= len(v)
        return product


def cross_val_score(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: Optional[callable] = None,
) -> List[float]:
    """
    Evaluate a score by cross-validation.

    Args:
        estimator: The object to use to fit the data. Must implement fit and predict.
        X (np.ndarray): The data to fit.
        y (np.ndarray): The target variable to try to predict.
        cv (int, optional): Number of folds. Defaults to 5.
        scoring (callable, optional): A scorer callable object / function with signature
            scorer(y_true, y_pred). If None, accuracy_score is used.

    Returns:
        List[float]: Array of scores of the estimator for each run of the cross validation.
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    fold_sizes = np.full(cv, n_samples // cv, dtype=int)
    fold_sizes[: n_samples % cv] += 1
    current = 0
    scores = []

    if scoring is None:
        scoring = accuracy_score

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate((indices[:start], indices[stop:]))

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Clone the estimator to ensure independence between folds
        fold_estimator = copy.deepcopy(estimator)
        
        fold_estimator.fit(X_train, y_train)
        y_pred = fold_estimator.predict(X_test)
        
        score = scoring(y_test, y_pred)
        scores.append(score)
        
        current = stop

    return scores


class GridSearchCVLite:
    """
    Exhaustive search over specified parameter values for an estimator.

    Args:
        estimator: The object to use to fit the data.
        param_grid (dict): Dictionary with parameters names (string) as keys
            and lists of parameter settings to try as values.
        cv (int, optional): Number of folds. Defaults to 5.
        scoring (callable, optional): A scorer callable object / function with signature
            scorer(y_true, y_pred).
        refit (bool, optional): Refit an estimator using the best found parameters on the whole dataset.
            Defaults to True.
    """

    def __init__(
        self,
        estimator: Any,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: Optional[callable] = None,
        refit: bool = True,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        
        self.best_params_ = None # holds the best parameters found
        self.best_score_ = None # holds the best score found
        self.best_estimator_ = None # holds the best estimator found
        self.cv_results_ = [] # holds the results of the cross-validation

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Run fit with all sets of parameters.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.
        """
        grid = ParameterGrid(self.param_grid)
        best_score = -float("inf")
        
        for params in grid:
            # Create a new estimator with the current parameters
            current_estimator = copy.deepcopy(self.estimator)
            
            # Try to set parameters. 
            if hasattr(current_estimator, "set_params"):
                current_estimator.set_params(**params)
            else:
                for key, value in params.items():
                    setattr(current_estimator, key, value)

            scores = cross_val_score(
                current_estimator, X, y, cv=self.cv, scoring=self.scoring
            )
            mean_score = np.mean(scores)
            
            self.cv_results_.append({
                "params": params,
                "mean_test_score": mean_score,
                "std_test_score": np.std(scores)
            })

            if mean_score > best_score:
                best_score = mean_score
                self.best_score_ = best_score
                self.best_params_ = params
                self.best_estimator_ = current_estimator

        if self.refit and self.best_estimator_ is not None:
            self.best_estimator_.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Call predict on the estimator with the best found parameters.

        Args:
            X (np.ndarray): Data to predict.

        Returns:
            np.ndarray: Predictions.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("The model has not been trained yet.")
        return self.best_estimator_.predict(X)
