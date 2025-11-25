"""SVMLite: A lightweight SVM implementation from scratch."""

__version__ = "0.2.0"

from .svm import SVCLite
from .utils import StandardScalerLite

__all__ = ["SVCLite", "StandardScalerLite"]
