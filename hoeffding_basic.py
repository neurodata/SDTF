"""
Main Author: Haoyin Xu
Corresponding Email: haoyinxu@gmail.com
"""

# import the necessary packages
import math
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def hoeffding_batch(X, y, confidence):
    """
    Create a basic hoeffding tree trained by data X with labels y.

    Parameters
    ----------
    X : ndarray
        Input data matrix.
    y : ndarray
        Output (i.e. response data matrix).
    confidence: int
                Probability of Hoeffding bound's correctness
    """
    n_samples, n_features = X.shape
    range = 1

    # calculate the Hoeffding bound
    ht_bound = math.sqrt((range**2) * math.log(1/(1-confidence)) / (2*n_samples))

    # create an empty decision tree
    ht = DecisionTreeClassifier(max_features=2, min_impurity_decrease=ht_bound)
    print(ht_bound)

    # fit the Hoeffding batch tree
    ht.fit(X, y)

    # return the fitted decision tree
    return ht
