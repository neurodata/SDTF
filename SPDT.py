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
class SPDTClassificationTransformer(BaseTransformer):
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

    # def chunk_fit(self, X, y):
    #     """
    #     Fit the tree with data chunks.
    #
    #     Parameters
    #     ----------
    #     X : ndarray
    #         Input data matrix.
    #     y : ndarray
    #         Output (i.e. response data matrix).
    #     """
    #
    #     if not self.is_fitted():
    #         self.fit(X, y)
    #     else:
    #         # TODO: Implement the chunk fitting function
    #
    #         if check_input:
    #             # Need to validate separately here.
    #             # We can't pass multi_ouput=True because that would allow y to be
    #             # csr.
    #             check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
    #             check_y_params = dict(ensure_2d=False, dtype=None)
    #             X, y = self.transformer._validate_data(X, y,
    #                                        validate_separately=(check_X_params,
    #                                                             check_y_params))
    #             if issparse(X):
    #                 X.sort_indices()
    #
    #         # Determine output settings
    #         n_samples = X.shape[0]
    #         is_classification = is_classifier(self.transformer)
    #
    #         y = np.atleast_1d(y)
    #         expanded_class_weight = None
    #
    #         if y.ndim == 1:
    #             # reshape is necessary to preserve the data contiguity against vs
    #             # [:, np.newaxis] that does not.
    #             y = np.reshape(y, (-1, 1))
    #
    #         self.transformer.n_outputs_ += y.shape[1]
    #
    #         if is_classification:
    #             y = np.copy(y)
    #
    #             if self.transformer.class_weight is not None:
    #                 y_original = np.copy(y)
    #
    #             y_encoded = np.zeros(y.shape, dtype=int)
    #             for k in range(y.shape[1]):
    #                 classes_k, y_encoded[:, k] = np.unique(y[:, k],
    #                                                        return_inverse=True)
    #                 self.transformer.classes_.append(classes_k)
    #                 self.transformer.n_classes_.append(classes_k.shape[0])
    #             y = y_encoded
    #
    #             if self.transformer.class_weight is not None:
    #                 expanded_class_weight = compute_sample_weight(
    #                     self.transformer.class_weight, y_original)
    #
    #             self.transformer.n_classes_ = np.concatenate((self.transformer.n_classes_, np.array(self.transformer.n_classes_, dtype=np.intp)), axis=0)
    #
    #         if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
    #             y = np.ascontiguousarray(y, dtype=DOUBLE)
    #
    #         # Check parameters
    #         max_depth = (np.iinfo(np.int32).max if self.transformer.max_depth is None
    #                      else self.transformer.max_depth)
    #         max_leaf_nodes = (-1 if self.transformer.max_leaf_nodes is None
    #                           else self.transformer.max_leaf_nodes)
    #
    #         min_samples_split = max(min_samples_split, 2 * min_samples_leaf)
    #
    #         if isinstance(self.transformer.max_features, str):
    #             if self.transformer.max_features == "auto":
    #                 if is_classification:
    #                     max_features = max(1, int(np.sqrt(self.transformer.n_features_)))
    #                 else:
    #                     max_features = self.transformer.n_features_
    #             elif self.transformer.max_features == "sqrt":
    #                 max_features = max(1, int(np.sqrt(self.transformer.n_features_)))
    #             elif self.transformer.max_features == "log2":
    #                 max_features = max(1, int(np.log2(self.transformer.n_features_)))
    #             else:
    #                 raise ValueError("Invalid value for max_features. "
    #                                  "Allowed string values are 'auto', "
    #                                  "'sqrt' or 'log2'.")
    #         elif self.transformer.max_features is None:
    #             max_features = self.transformer.n_features_
    #         elif isinstance(self.transformer.max_features, numbers.Integral):
    #             max_features = self.transformer.max_features
    #         else:  # float
    #             if self.transformer.max_features > 0.0:
    #                 max_features = max(1,
    #                                    int(self.transformer.max_features * self.transformer.n_features_))
    #             else:
    #                 max_features = 0
    #
    #         self.transformer.max_features_ = max_features
    #
    #         if len(y) != n_samples:
    #             raise ValueError("Number of labels=%d does not match "
    #                              "number of samples=%d" % (len(y), n_samples))
    #         if not 0 <= self.transformer.min_weight_fraction_leaf <= 0.5:
    #             raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
    #         if max_depth <= 0:
    #             raise ValueError("max_depth must be greater than zero. ")
    #         if not (0 < max_features <= self.transformer.n_features_):
    #             raise ValueError("max_features must be in (0, n_features]")
    #         if not isinstance(max_leaf_nodes, numbers.Integral):
    #             raise ValueError("max_leaf_nodes must be integral number but was "
    #                              "%r" % max_leaf_nodes)
    #         if -1 < max_leaf_nodes < 2:
    #             raise ValueError(("max_leaf_nodes {0} must be either None "
    #                               "or larger than 1").format(max_leaf_nodes))
    #
    #         # Set min_weight_leaf from min_weight_fraction_leaf
    #         min_weight_leaf = (self.transformer.min_weight_fraction_leaf *
    #                                np.sum(sample_weight))
    #
    #         min_impurity_split = self.transformer.min_impurity_split
    #         if min_impurity_split is not None:
    #             warnings.warn("The min_impurity_split parameter is deprecated. "
    #                           "Its default value has changed from 1e-7 to 0 in "
    #                           "version 0.23, and it will be removed in 0.25. "
    #                           "Use the min_impurity_decrease parameter instead.",
    #                           FutureWarning)
    #
    #             if min_impurity_split < 0.:
    #                 raise ValueError("min_impurity_split must be greater than "
    #                                  "or equal to 0")
    #         else:
    #             min_impurity_split = 0
    #
    #         if self.transformer.min_impurity_decrease < 0.:
    #             raise ValueError("min_impurity_decrease must be greater than "
    #                              "or equal to 0")
    #
    #         # Build tree
    #         criterion = self.transformer.criterion
    #         if not isinstance(criterion, Criterion):
    #             if is_classification:
    #                 criterion = CRITERIA_CLF[self.transformer.criterion](self.transformer.n_outputs_,
    #                                                          self.transformer.n_classes_)
    #             else:
    #                 criterion = CRITERIA_REG[self.transformer.criterion](self.transformer.n_outputs_,
    #                                                          n_samples)
    #
    #         SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS
    #
    #         splitter = self.transformer.splitter
    #         if not isinstance(self.transformer.splitter, Splitter):
    #             splitter = SPLITTERS[self.transformer.splitter](criterion,
    #                                                 self.transformer.max_features_,
    #                                                 min_samples_leaf,
    #                                                 min_weight_leaf,
    #                                                 random_state)
    #
    #         if is_classifier(self.transformer):
    #             self.transformer.tree_ = Tree(self.transformer.n_features_,
    #                               self.transformer.n_classes_, self.transformer.n_outputs_)
    #         else:
    #             self.transformer.tree_ = Tree(self.transformer.n_features_,
    #                               # TODO: tree should't need this in this case
    #                               np.array([1] * self.transformer.n_outputs_, dtype=np.intp),
    #                               self.transformer.n_outputs_)
    #
    #         # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
    #         if max_leaf_nodes < 0:
    #             builder = DepthFirstTreeBuilder(splitter, min_samples_split,
    #                                             min_samples_leaf,
    #                                             min_weight_leaf,
    #                                             max_depth,
    #                                             self.transformer.min_impurity_decrease,
    #                                             min_impurity_split)
    #         else:
    #             builder = BestFirstTreeBuilder(splitter, min_samples_split,
    #                                            min_samples_leaf,
    #                                            min_weight_leaf,
    #                                            max_depth,
    #                                            max_leaf_nodes,
    #                                            self.transformer.min_impurity_decrease,
    #                                            min_impurity_split)
    #
    #         builder.build(self.transformer.tree_, X, y, None)
    #
    #         if self.transformer.n_outputs_ == 1 and is_classifier(self.transformer):
    #             self.transformer.n_classes_ = self.transformer.n_classes_[0]
    #             self.transformer.classes_ = self.transformer.classes_[0]
    #
    #         self.transformer._prune_tree()
    #
    #         return self
