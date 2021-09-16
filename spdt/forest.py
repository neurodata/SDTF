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


class StreamForest:
    """
    A class used to represent a naive ensemble of
    random stream decision trees.

    Attributes
    ----------
    n_estimators : int
        An integer that represents the number of stream decision trees.

    forest_ : list of sklearn.tree.DecisionTreeClassifier
        An internal list that contains random
        sklearn.tree.DecisionTreeClassifier.
    """

    def __init__(self, n_estimators=100, splitter="best"):
        self.forest_ = []

        for i in range(n_estimators):
            tree = DecisionTreeClassifier(max_features="auto", splitter=splitter)
            self.forest_.append(tree)

    def fit(self, X, y, classes=None):
        """
        Fits the forest to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response data matrix).

        Returns
        -------
        self : TreeClassificationTransformer
            The object itself.
        """
        X, y = check_X_y(X, y)

        # Update stream decision trees with random inputs
        for tree in self.forest_:
            p = permutation(X.shape[0])
            X_r = X[p]
            y_r = y[p]
            tree.partial_fit(X_r, y_r, classes=classes)

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

        results = []
        for tree in self.forest_:
            result = tree.predict(X)
            results.append(result)

        major_result = stats.mode(results)[0][0]

        return major_result


class CascadeStreamForest:
    """
    A class used to represent a cascading ensemble of
    stream decision trees.

    Attributes
    ----------
    n_estimators : int
        An integer that represents the max number of stream decision trees.

    splitter : str
        A choice of decision tree splitter

    forest_ : list of sklearn.tree.DecisionTreeClassifier
        An internal list that contains cascading
        sklearn.tree.DecisionTreeClassifier.
    """

    def __init__(self, n_estimators=100, splitter="best"):
        self.forest_ = []
        self.n_estimators = n_estimators
        self.splitter = splitter

    def fit(self, X, y, classes=None):
        """
        Fits the forest to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response data matrix).

        Returns
        -------
        self : TreeClassificationTransformer
            The object itself.
        """
        X, y = check_X_y(X, y)

        # Update existing stream decision trees
        for tree in self.forest_:
            tree.partial_fit(X, y, classes=classes)

        # Before the maximum number of trees
        if len(self.forest_) < self.n_estimators:
            # Add a new decision tree based on new data
            sdt = DecisionTreeClassifier(splitter=self.splitter)
            sdt.partial_fit(X, y, classes=classes)
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

        results = []
        for tree in self.forest_:
            result = tree.predict(X)
            results.append(result)

        major_result = stats.mode(results)[0][0]

        return major_result
