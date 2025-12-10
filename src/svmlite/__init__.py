"""SVMLite: A lightweight SVM implementation from scratch."""

__version__ = "0.4.0"

from .svm import SVCLite
from .utils import StandardScalerLite
from .model_selection import GridSearchCVLite, cross_val_score

__all__ = ["SVCLite", "StandardScalerLite", "GridSearchCVLite", "cross_val_score"]
