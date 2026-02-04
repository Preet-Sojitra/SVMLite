# SVMLite

This project is part of CS6375 Machine learning course at University of Texas at Dallas.

A lightweight Python library implementing Support  Vector Machines from scratch for educational and experimental use.

## Project Report

You can access the detailed project report here at this [link](https://drive.google.com/file/d/1mEAilQojR0T8nGhrNu0BiGJ8jjtfr7Fg/view?usp=sharing)

# Features
- Implemenation of primal form (hard margin and soft margin) of SVM Classification using Stochastic Gradient Descent (SGD). 
- QP (Quadratic Programming) based SVM implementation using cvxopt
- Kernel Support: Linear, Polynomial, RBF kernel, Sigmoid kernel and Custom Kernel support
- SMO (Sequential Minimal Optimization) algorithm for optimization (simplified heuristic for selecting alpha pairs)
- Multiclass classification using One-vs-One (OvO) and One-vs-All (OvA) strategies
- Grid Search for hyperparameter tuning
- Cross Validation for model selection

## Todo

- [ ] Add more example of the features that were implemented in quick starts.
- [ ] Add documentation
- [ ] Add more comprehensive usage guide
- [ ] Set CI/CD for automatic deployment to pip

## Installation

```bash
pip install svmlite
```

## Quick Start

```python
from svmlite.svm import SVCLite
from svmlite.utils import StandardScalerLite
from svmlite.metrics import accuracy_score
import numpy as np

# prepare data
X = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 8], [8, 7]])
y = np.array([-1, -1, -1, 1, 1, 1])

# scale features
scaler = StandardScalerLite()
X_scaled = scaler.fit_transform(X)

# train SVM
model = SVCLite(C=1.0)
model.fit(X_scaled, y, learning_rate=0.01, n_iters=1000)

# predict
predictions = model.predict(X_scaled)
print("Predictions:", predictions)

# evaluate
acc = accuracy_score(y, predictions)
print("Accuracy:", acc)
```
