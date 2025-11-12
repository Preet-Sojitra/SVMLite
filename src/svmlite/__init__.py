"""SVMLite: A lightweight SVM implementation from scratch."""

__version__ = "0.1.0"

from .svm import SVCLite
from .utils import StandardScalerLite

__all__ = ["SVCLite", "StandardScalerLite"]
