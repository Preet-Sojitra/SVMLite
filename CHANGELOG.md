# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-11-24

### Added
- QP (Quadratic Programming) based SVM implementation using cvxopt
- Kernel Support: Linear, Polynomial, RBF kernel and Sigmoid kernel
- Custom Kernel support 
- Added cvxopt dependency for convex optimization

### Changed
- Updated SVCLite class `fit` and `predict` method to be dispathchers based on the selected solver (SGD or QP)

### Fixed
- No bug fixes in this release

## [0.1.0] - 2025-11-12

### Added
- Initial release
- Hard margin SVM implementation using Stochastic Gradient Descent (SGD)
- Soft margin SVM implementation with regularization parameter C
- Standard Scaler for feature scaling
- Accuracy metric function for model evaluation
- Comprehensive tests comparing SVMLite with scikit-learn's SVM implementation