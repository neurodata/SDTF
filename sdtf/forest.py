"""
Main Author: Haoyin Xu
Corresponding Email: haoyinxu@gmail.com
"""
# import the necessary packages
import numpy as np
from scipy import stats

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
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed


def _partial_fit(tree, X, y, n_samples_bootstrap, classes):
    """Internal function to partially fit a tree."""
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

    n_swaps : int, default=1
        The number of trees to swap at each partial fitting. The actual
        swaps occur with `1/n_batches_` probability.

    Attributes
    ----------
    estimators_ : list of sklearn.tree.DecisionTreeClassifier
        An internal list that contains all
        sklearn.tree.DecisionTreeClassifier.

    classes_ : list of all unique class labels
        An internal list that stores class labels after the first call
        to `partial_fit`.

    n_batches_ : int
        The number of batches seen with `partial_fit`.
    """

    def __init__(
        self,
        n_estimators=100,
        splitter="best",
        max_features="sqrt",
        bootstrap=True,
        n_jobs=None,
        max_samples=None,
        n_swaps=1,
    ):
        self.estimators_ = []
        self.n_batches_ = 0
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.splitter = splitter
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.max_samples = max_samples
        self.n_swaps = n_swaps

        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_features=self.max_features, splitter=self.splitter
            )
            self.estimators_.append(tree)

    def fit(self, X, y, classes=None):
        """
        Partially fits the forest to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        y : ndarray
            Output (i.e. response data matrix).

        classes : ndarray, default=None
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        Returns
        -------
        self : StreamDecisionForest
            The object itself.
        """
        if classes is None:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = classes

        if self.n_batches_ != 0:
            self.estimators_ = []
            for i in range(self.n_estimators):
                tree = DecisionTreeClassifier(
                    max_features=self.max_features, splitter=self.splitter
                )
                self.estimators_.append(tree)
            self.n_batches_ = 0

        return self.partial_fit(X, y, classes=self.classes_)

    def partial_fit(self, X, y, classes=None):
        """
        Partially fits the forest to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        y : ndarray
            Output (i.e. response data matrix).

        classes : ndarray, default=None
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        Returns
        -------
        self : StreamDecisionForest
            The object itself.
        """
        X, y = check_X_y(X, y)
        if classes is not None:
            self.classes_ = classes

        # Update stream decision trees with random inputs
        if self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(X.shape[0], self.max_samples)
        else:
            n_samples_bootstrap = X.shape[0]

        self.n_batches_ += 1

        # Calculate probability of swaps
        swap_prob = 1 / self.n_batches_
        if self.n_swaps > 0 and self.n_batches_ > 2 and np.random.random() <= swap_prob:
            # Evaluate forest performance
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(tree.predict)(X) for tree in self.estimators_
            )

            # Sort predictions by accuracy
            acc_l = []
            for idx, result in enumerate(results):
                acc_l.append([accuracy_score(result, y), idx])
            acc_l = sorted(acc_l, key=lambda x: x[0])

            # Generate new trees
            new_trees = Parallel(n_jobs=self.n_jobs)(
                delayed(_partial_fit)(
                    DecisionTreeClassifier(
                        max_features=self.max_features, splitter=self.splitter
                    ),
                    X,
                    y,
                    n_samples_bootstrap=n_samples_bootstrap,
                    classes=self.classes_,
                )
                for i in range(self.n_swaps)
            )

            # Swap worst performing trees with new trees
            for i in range(self.n_swaps):
                self.estimators_[acc_l[i][1]] = new_trees[i]

        # Update existing stream decision trees
        trees = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit)(
                tree,
                X,
                y,
                n_samples_bootstrap=n_samples_bootstrap,
                classes=self.classes_,
            )
            for tree in self.estimators_
        )
        self.estimators_ = trees

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
        check_is_fitted(self)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.estimators_
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
    estimators_ : list of sklearn.tree.DecisionTreeClassifier
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
        self.estimators_ = []
        self.n_estimators = n_estimators
        self.splitter = splitter
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.max_samples = max_samples

    def fit(self, X, y, classes=None):
        """
        Partially fits the forest to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        y : ndarray
            Output (i.e. response data matrix).

        classes : ndarray, default=None
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        Returns
        -------
        self : CascadeStreamForest
            The object itself.
        """

        if classes is None:
            classes = np.unique(y)

        return self.partial_fit(X, y, classes=classes)

    def partial_fit(self, X, y, classes=None):
        """
        Partially fits the forest to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        y : ndarray
            Output (i.e. response data matrix).

        classes : ndarray, default=None
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

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
            for tree in self.estimators_
        )
        self.estimators_ = trees

        # Before the maximum number of trees
        if len(self.estimators_) < self.n_estimators:
            # Add a new decision tree based on new data
            sdt = DecisionTreeClassifier(
                splitter=self.splitter, max_features=self.max_features
            )
            _partial_fit(
                sdt, X, y, n_samples_bootstrap=n_samples_bootstrap, classes=classes
            )
            self.estimators_.append(sdt)

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
        check_is_fitted(self)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.estimators_
        )

        major_result = stats.mode(results)[0][0]

        return major_result
