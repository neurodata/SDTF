"""
Main Author: Haoyin Xu
Corresponding Email: haoyinxu@gmail.com
"""
# import the necessary packages
from scipy import stats
from numpy.random import permutation

# NOTE: the sklearn dependence is based on
# personal fork and not corresponding to
# the official scikit-learn repository
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import (
    check_X_y,
    check_array,
)
from joblib import Parallel, delayed


def _partial_fit(tree, X, y, classes=None):
    """
    Internal function to partially fit a tree.

    Parameters
    ----------
    tree : DecisionTreeClassifier
        Tree to be partially fitted.
    X : ndarray
        Input data matrix.
    y : ndarray
        Output (i.e. response data matrix).

    Returns
    -------
    tree : DecisionTreeClassifier
        The fitted decision tree.
    """
    p = permutation(X.shape[0])
    X_r = X[p]
    y_r = y[p]
    tree.partial_fit(X_r, y_r, classes=classes)

    return tree


class StreamDecisionForest:
    """
    A class used to represent a naive ensemble of
    random stream decision trees.

    Attributes
    ----------
    n_estimators : int, default=100
        An integer that represents the number of stream decision trees.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : {"sqrt", "log2"}, int or float, default="sqrt"
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

    n_jobs : int, default=None
        The number of jobs to run in parallel.

    forest_ : list of sklearn.tree.DecisionTreeClassifier
        An internal list that contains random
        sklearn.tree.DecisionTreeClassifier.
    """

    def __init__(
        self, n_estimators=100, splitter="best", max_features="sqrt", n_jobs=None
    ):
        self.forest_ = []
        self.n_jobs = n_jobs

        for i in range(n_estimators):
            tree = DecisionTreeClassifier(max_features=max_features, splitter=splitter)
            self.forest_.append(tree)

    def partial_fit(self, X, y, classes=None):
        """
        Partially fits the forest to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response data matrix).

        Returns
        -------
        self : StreamDecisionForest
            The object itself.
        """
        X, y = check_X_y(X, y)

        # Update stream decision trees with random inputs
        trees = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit)(tree, X, y, classes=classes) for tree in self.forest_
        )
        self.forest_ = trees

        return self

    def predict(self, X):
        """
        Performs inference using the forest.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        Returns
        -------
        major_result : ndarray
            The majority predictions.
        """
        X = check_array(X)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.forest_
        )

        major_result = stats.mode(results)[0][0]

        return major_result


class CascadeStreamForest:
    """
    A class used to represent a cascading ensemble of
    stream decision trees.

    Attributes
    ----------
    n_estimators : int, default=100
        An integer that represents the max number of stream decision trees.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : {"sqrt", "log2"}, int or float, default="sqrt"
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

    n_jobs : int, default=None
        The number of jobs to run in parallel.

    forest_ : list of sklearn.tree.DecisionTreeClassifier
        An internal list that contains cascading
        sklearn.tree.DecisionTreeClassifier.
    """

    def __init__(
        self, n_estimators=100, splitter="best", max_features="sqrt", n_jobs=None
    ):
        self.forest_ = []
        self.n_estimators = n_estimators
        self.splitter = splitter
        self.n_jobs = n_jobs
        self.max_features = max_features

    def partial_fit(self, X, y, classes=None):
        """
        Partially fits the forest to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response data matrix).

        Returns
        -------
        self : CascadeStreamForest
            The object itself.
        """
        X, y = check_X_y(X, y)

        # Update existing stream decision trees
        trees = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit)(tree, X, y, classes=classes) for tree in self.forest_
        )
        self.forest_ = trees

        # Before the maximum number of trees
        if len(self.forest_) < self.n_estimators:
            # Add a new decision tree based on new data
            sdt = DecisionTreeClassifier(
                splitter=self.splitter, max_features=self.max_features
            )
            _partial_fit(sdt, X, y, classes=classes)
            self.forest_.append(sdt)

        return self

    def predict(self, X):
        """
        Performs inference using the forest.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        Returns
        -------
        major_result : ndarray
            The majority predictions.
        """
        X = check_array(X)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.forest_
        )

        major_result = stats.mode(results)[0][0]

        return major_result
