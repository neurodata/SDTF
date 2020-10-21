"""
Main Author: Haoyin Xu
Corresponding Email: haoyinxu@gmail.com
"""
# import the necessary packages
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils import check_random_state
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

    Attributes (object)
    ---
    kwargs : dict
        A dictionary to contain parameters of the tree.
    _is_fitted_ : bool
        A boolean to identify if the model is currently fitted.

    Methods
    ---
    fit(X, y)
        Fits the transformer to data X with labels y.
    transform(X)
        Performs inference using the transformer.
    is_fitted()
        Indicates whether the transformer is fitted.
    plot()
        Plot the fitted tree.
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

    def is_fitted(self):
        """
        Indicate whether the transformer is fitted.

        Parameters
        ----------
        None
        """

        return self._is_fitted

    def plot(self):
        """
        Plot the fitted tree.

        Parameters
        ----------
        None
        """

        if self.is_fitted():
            fig, ax = plt.subplots(figsize=(20, 20))
            plot_tree(self.transformer, filled=True, fontsize=15)
            plt.show()

    def chunk_fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted="deprecated"):
        """
        Fit the tree with data chunks.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response data matrix).
        """

        if not self.is_fitted():
            self.fit(X, y)
        else:
            # TODO: Implement the chunk fitting function

            return self
