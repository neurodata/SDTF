"""
Main Author: Haoyin Xu
Corresponding Email: haoyinxu@gmail.com
"""
# import the necessary packages
from scipy import stats

# NOTE: the sklearn dependence is based on
# personal fork and not corresponding to
# the official scikit-learn repository
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import (
    check_X_y,
    check_array,
)


class CascadeStreamForest:
    """
    A class used to represent a cascading ensemble of
    stream decision trees.

    Attributes
    ----------
    forest_ : list of sklearn.tree.DecisionTreeClassifier
        An internal list that contains cascading
        sklearn.tree.DecisionTreeClassifier.
    """

    def __init__(self):
        self.forest_ = []

    def fit(self, X, y):
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
            tree.fit(X, y, update_tree=True)

        # Add a new decision tree based on new data
        sdt = DecisionTreeClassifier()
        sdt.fit(X, y)
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
