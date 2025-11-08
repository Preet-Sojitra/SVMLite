# Week 1: Primal Focus and Core Library Structure (11/8 - 11/15)
- Theory to study:
  - Max margin classifier intuition
  - hard margin svm (primal form)
  - soft margin svm (primal form)
  - optimization using gradient descent or sub-gradient descent
- Implementation tasks:
  - Set up project structure and version control
  - SVM Class structure: init (with regularization parameter) , fit, predict
  - Fit method using primal form with gradient descent
  - Simple standard scaler as svm is sensitive to feature scaling
  - Basic test with linearly separable data. Compare with sklearn's SVM
- Report:
  - Section 1: Introduction
  - Section 2: The maximal margin classifier: core intution of SVM
  - Section 3: primal svm formulation: detailed math for hard margin svm and soft margin svm. role of regularization
  - Section 4: optimzation with gradient descent and sub-gradient descent

# Week 2: Duality and Kernel Trick (11/16 - 11/22)
- Theory to study:
  - Lagrangian duality
  - SVM Dual problem
  - Kernel trick
  - common kernels
- Implementation tasks:
  - use black box QP solver (not to implment SMO yet). To isolate the problem, first use a standard library like `cvxopt` to solve the dual problem. Task is to correctly formulate the matrices (P, q, G, h, etc.) that cvxopt needs, based on the SVM dual formulation.
  - Kernel functions
  - Update SVM class to handle dual formulation and kernel trick
    - compute the kernel matrix (gram matrix)
    - setup matrices for cvxopt based on dual formulation
    - call `cvxopt.solvers.qp` to get the alphas
  - update predict method for dual formulation. It will be bsed on support vectors and kernel function
  - Test with non-linearly separable data. Compare with sklearn's SVM
- Report:
  - Section 5: Dual Formulation
  - Section 6: Kernel Trick and Common Kernels

# Week 3: Advanced Optimzation (SMO) and Extensions (11/23 - 11/29)
> This week includes fall break and thanksgiving. Try to utilize this week to its fullest.
- Theory to study:
  - SMO Algorithm: understand its working. KKT conditions for optimality in SVMs.
  - Multiclass : OVR and OVO
  - Support Vector Regression
- Implementation tasks:
  - Implement SMO (replace cvxopt with your own SMO implementation)
    - main loop until convergence
    - heuristic to select first alpha and second alpha
    - function to compute new alpha values and clip them
    - update b (intercept) using the KKT conditions
  - Integrate SMO into SVM class
  - Multiclass wappers: OVR and OVO. Manages multiple binary SVM instances internally
  - Support Vector Regression (SVR) class (similar to SVM class just loss function will be different and SMO updates will be different)
  - Test
- Report:
  - Section 7: Optimization - Sequential Minimal Optimization (SMO)
  - Section 8: Extensions 
    - Subsection 8.1: Multiclass SVM (OVR and OVO)
    - Subsection 8.2: Support Vector Regression (SVR)
# Week 4 Evaluation, Polish and Final Submission (11/30 - 12/6)
- Theory to study:
  - Cross Validation and Grid Search
  - Evaluation Metrics for classification and regression
- Implementation tasks:
  - Parameter selection utilities: simple GridSearchCV
  - Evaluation tools
  - Visulization: helper functions to plot decision boundaries, support vectors, etc. on 2D data
  - Comparison with Scikit-learn (Pick some dataset from kaggle or any open source dataset repository and compare your implementation with sklearn's SVM implementation in terms of accuracy and training time)
  - Finalize libary: Clean code, add docstrings, comments and Readme. Publish to pypi if possible
- Report and Presentation:
  - Section 9: Experiments and Evaluations (it will be results of my implementation on various datasets, visualizations, evaluation metrics, etc.)
  - Section 10:Comparison with Scikit-learn (table or plot comparing my implementation with sklearn's SVM in terms of accuracy and training time on selected datasets). Discuss any discrepencies and possible reasons.
  - Section 11: Conclusion and Future Work
  - Final polish: Review the entire report for clarity, grammer and formatting. Add references (Don't forget to cite original paper and the AI lol)
  - Create presentation slides

# Final deadline for submission: 12/10

I will try to stick to this plan as closely as possible, but some adjustments may be necessary based on progress and unforeseen challenges. Remaining days after 12/6 will be used mainly for report polishing. I need to record video explaining the PPT as well. That's why last few days are for this miscellaneous tasks.