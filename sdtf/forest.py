"""
Main Author: Haoyin Xu
Corresponding Email: haoyinxu@gmail.com
"""
# import the necessary packages
import numpy as np
from scipy import stats
from numpy.random import permutation

# NOTE: the sklearn dependence is based on
# personal fork and not corresponding to
# the official scikit-learn repository
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble._forest import (
    _get_n_samples_bootstrap,
    _generate_sample_indices,
    _generate_unsampled_indices,
)
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.validation import (
    check_X_y,
    check_array,
)
from joblib import Parallel, delayed


def _partial_fit(tree, X, y, n_samples_bootstrap, classes=None):
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
    indices = _generate_sample_indices(
        tree.random_state, X.shape[0], n_samples_bootstrap
    )
    tree.partial_fit(X[indices, :], y[indices], classes=classes)

    return tree


class StreamDecisionForest:
    """
    A class used to represent a naive ensemble of
    random stream decision trees.

    Parameters
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

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    n_jobs : int, default=None
        The number of jobs to run in parallel.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.
        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

    Attributes
    ----------
    forest_ : list of sklearn.tree.DecisionTreeClassifier
        An internal list that contains random
        sklearn.tree.DecisionTreeClassifier.
    """

    def __init__(
        self,
        n_estimators=100,
        splitter="best",
        max_features="sqrt",
        bootstrap=True,
        n_jobs=None,
        max_samples=None,
    ):
        self.forest_ = []
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.max_samples = max_samples

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
        if self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(X.shape[0], self.max_samples)
        else:
            n_samples_bootstrap = X.shape[0]
        trees = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit)(
                tree, X, y, n_samples_bootstrap=n_samples_bootstrap, classes=classes
            )
            for tree in self.forest_
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

    Parameters
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

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    n_jobs : int, default=None
        The number of jobs to run in parallel.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.
        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

    Attributes
    ----------
    forest_ : list of sklearn.tree.DecisionTreeClassifier
        An internal list that contains cascading
        sklearn.tree.DecisionTreeClassifier.
    """

    def __init__(
        self,
        n_estimators=100,
        splitter="best",
        max_features="sqrt",
        bootstrap=True,
        n_jobs=None,
        max_samples=None,
    ):
        self.forest_ = []
        self.n_estimators = n_estimators
        self.splitter = splitter
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.max_samples = max_samples

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

        if self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(X.shape[0], self.max_samples)
        else:
            n_samples_bootstrap = X.shape[0]
        # Update existing stream decision trees
        trees = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit)(
                tree, X, y, n_samples_bootstrap=n_samples_bootstrap, classes=classes
            )
            for tree in self.forest_
        )
        self.forest_ = trees

        # Before the maximum number of trees
        if len(self.forest_) < self.n_estimators:
            # Add a new decision tree based on new data
            sdt = DecisionTreeClassifier(
                splitter=self.splitter, max_features=self.max_features
            )
            _partial_fit(
                sdt, X, y, n_samples_bootstrap=n_samples_bootstrap, classes=classes
            )
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
