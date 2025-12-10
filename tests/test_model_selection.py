"""
Tests for model selection module.
"""

import pytest
import numpy as np
from sklearn.model_selection import GridSearchCV as SklearnGridSearchCV
from sklearn.svm import SVC as SklearnSVC
from sklearn.metrics import accuracy_score

from svmlite import SVCLite, GridSearchCVLite, cross_val_score


def test_cross_val_score_shapes(data):
    """Test cross_val_score returns correct number of scores."""
    X, y = data
    clf = SVCLite(kernel="linear", C=1.0)
    
    cv = 5
    scores = cross_val_score(clf, X, y, cv=cv)
    
    assert len(scores) == cv
    assert all(isinstance(s, float) for s in scores)
    assert all(0 <= s <= 1 for s in scores)


def test_grid_search_cv_comparison(data):
    """Compare SVCLite GridSearchCV with Scikit-learn."""
    X, y = data
    
    # Define parameter grid
    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"]
    }
    
    # --- SVCLite ---
    lite_clf = SVCLite()
    lite_gs = GridSearchCVLite(lite_clf, param_grid, cv=3)
    lite_gs.fit(X, y)
    
    # --- Scikit-learn ---
    sklearn_clf = SklearnSVC(gamma='auto') # gamma='auto' to silence warnings
    sklearn_gs = SklearnGridSearchCV(sklearn_clf, param_grid, cv=3)
    sklearn_gs.fit(X, y)
    
    # Checks
    print(f"SVCLite Best Params: {lite_gs.best_params_}")
    print(f"Sklearn Best Params: {sklearn_gs.best_params_}")
    print(f"SVCLite Best Score: {lite_gs.best_score_}")
    print(f"Sklearn Best Score: {sklearn_gs.best_score_}")
    
    # 1. Check that best_params_ is a dict and contains expected keys
    assert isinstance(lite_gs.best_params_, dict)
    assert "C" in lite_gs.best_params_
    assert "kernel" in lite_gs.best_params_
    
    # 2. Check that best_score_ is a float
    assert isinstance(lite_gs.best_score_, float)
    
    # 3. Check that best_estimator_ is an instance of SVCLite
    assert isinstance(lite_gs.best_estimator_, SVCLite)
    
    # 4. Check that we can predict with the fitted grid
    y_pred = lite_gs.predict(X)
    assert y_pred.shape == y.shape
    
    # 5. Check refit works (best_estimator_ should be fitted)
    # If we call predict, it shouldn't raise error
    try:
        lite_gs.best_estimator_.predict(X)
    except Exception as e:
        pytest.fail(f"best_estimator_ should be fitted after GridSearchCV.fit: {e}")


def test_grid_search_cv_results_structure(data):
    """Test the structure of cv_results_."""
    X, y = data
    param_grid = {"C": [1.0], "kernel": ["linear"]}
    
    clf = SVCLite()
    gs = GridSearchCVLite(clf, param_grid, cv=3)
    gs.fit(X, y)
    
    results = gs.cv_results_
    assert isinstance(results, list)
    assert len(results) == 1 # only 1 combination
    
    res = results[0]
    assert "params" in res
    assert "mean_test_score" in res
    assert "std_test_score" in res
    assert res["params"] == {"C": 1.0, "kernel": "linear"}
