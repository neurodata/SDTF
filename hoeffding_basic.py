"""
Main Author: Haoyin Xu
Corresponding Email: haoyinxu@gmail.com
"""

# import the necessary packages
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree


def hoeffding_tree(X, y):
    """
    Create a basic hoeffding tree trained by data X with labels y.

    Parameters
    ----------
    X : ndarray
        Input data matrix.
    y : ndarray
        Output (i.e. response data matrix).
    """

    # create an empty decision tree
    ht = DecisionTreeClassifier()

    # return the fitted decision tree
    return ht
