# Art-2a

This is an implementation of **ART-2A** in Python.

Adaptive resonance theory is developed by Stephen Grossberg and Gail Carpenter.

Art-2a is an **unsupervised** machine learning model. Basicaly it is a clustering method.

Reference: Du, K.-L & Swamy, M.N.s. (2006). Neural Networks in a Softcomputing Framework (pp. 210-212).

# Usage:

import art2a_test file

Use train method to get weights for future cluster predictions
### Ex:
    cluster, weights = art2a_train(vigilance_value, X, X.shape[1], cycle=10000)
    
Use predict method to get labels for new data. If data cannot be clustered, label would be -1.
### Ex:
    labels = art2a_predict(vigilance_value, X, X.shape[1], weights)