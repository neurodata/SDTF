"""
Main Author: Haoyin Xu
Corresponding Email: haoyinxu@gmail.com
"""
# import the necessary packages
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)
import matplotlib.pyplot as plt

from proglearn.base import BaseTransformer

# define the Hoeffding Decision Tree
class HoeffdingTreeTransformer(BaseTransformer):
    """
    A class used to represent an hoeffding decision tree.

    Attributes
    ---

    Methods
    ---

    """
    def __init__(self, kwargs={}):

        self.kwargs = kwargs

        self._is_fitted = False

    def fit(self, X, y):
        """
        Fit the transformer to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response data matrix).
        """

        X, y = check_X_y(X, y)

        # define the ensemble
        self.transformer = DecisionTreeClassifier(**self.kwargs).fit(X, y)

        self._is_fitted = True

        return self

    def transform(self, X):
        """
        Perform inference using the transformer.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """

        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)
        return self.transformer.apply(X)

    def plot(self):
        """
        Plot the fitted tree.

        Parameters
        ----------
        None
        """

        if (self.is_fitted()):
            fig, ax = plt.subplots(figsize=(20, 20))
            plot_tree(self.transformer, filled=True, fontsize=15)
            plt.show()

    def is_fitted(self):
        """
        Indicate whether the transformer is fitted.

        Parameters
        ----------
        None
        """

        return self._is_fitted
