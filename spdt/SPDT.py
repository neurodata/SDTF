"""
Main Author: Haoyin Xu
Corresponding Email: haoyinxu@gmail.com
"""
# import the necessary packages
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)
import matplotlib.pyplot as plt

from proglearn.base import BaseTransformer

# define the Hoeffding Decision Tree
class ClassificationTransformer(BaseTransformer):
    """
    A class used to represent an hoeffding decision tree.

    Parameters
    ----------
    kwargs : dict, default={}
        A dictionary to contain parameters of the tree.

    Attributes
    ----------
    transformer_ : sklearn.tree.DecisionTreeClassifier
        An internal sklearn DecisionTreeClassifier.
    histograms_ : list of dict
        A list to contain summary histograms for each feature.
    n_features_ : int
        An integer to specify the number of features.
    n_bin_ : int
        An integer to specify the number of histogram bins.
    """

    def __init__(self, kwargs={}):
        self.kwargs = kwargs

    def fit(self, X, y):
        """
        Fits the transformer to data X with labels y.

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

        self.n_bin_ = len(X) / 5
        self.n_features_ = len(X[0])
        self.histograms_ = [{} for idx in range(self.n_features_)]

        for i in range(len(X)):
            for j in range(self.n_features_):
                self.histograms_[j] = self._update(self.histograms_[j], X[i][j], y[i])

        self.transformer_ = DecisionTreeClassifier(**self.kwargs).fit(X, y)
        return self

    def transform(self, X):
        """
        Performs inference using the transformer.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        Returns
        -------
        X_transformed : ndarray
            The transformed input.

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self.transformer_.apply(X)

    def plot(self):
        """
        Plot the fitted tree.

        Parameters
        ----------
        None
        """
        fig, ax = plt.subplots(figsize=(20, 20))
        plot_tree(self.transformer_, filled=True, fontsize=15)
        plt.show()

    def _update(self, histogram, p, label):
        """
        Update the histogram with feature and label

        Parameters
        ----------
        histogram : dict
            Histgram to update.
        p : dtype
            Input feature value.
        label : dtype
            Output class label.

        Returns
        -------
        histogram : dict
            Updated histogram.
        """
        if p in histogram:
            histogram[p][0] += 1

        elif len(histogram) < self.n_bin_:
            histogram[p] = [1, label]

        else:
            histogram[p] = [1, label]
            histogram = self._trim(histogram)

        return histogram

    def _order(self, histogram):
        """
        Order the histogram

        Parameters
        ----------
        histogram : dict
            Histgram to order.

        Returns
        -------
        histogram: dict
            Ordered histogram.
        """
        histogram = dict(sorted(histogram.items()))
        return histogram

    def _trim(self, histogram):
        """
        Trim the histogram

        Parameters
        ----------
        histogram : dict
            Histgram to trim.

        Returns
        -------
        histogram: dict
            Trimmed histogram.
        """
        histogram = self._order(histogram)

        max_diff = float("inf")
        first = True
        for key, value in histogram.items():
            # Skip the first comparison
            if first:
                first = False
                prev_key = key
                prev_value = value
                continue

            # Find the most similar point values
            diff = value[0] - prev_value[0]
            if diff < max_diff:
                max_diff = diff
                key_0 = prev_key
                key_1 = key
                value_0 = prev_value
                value_1 = value

            prev_key = key
            prev_value = value

        # Remove old keys
        del histogram[key_0]
        del histogram[key_1]

        # Insert new key
        new_key = (key_0 * value_0[0] + key_1 * value_1[0]) / (value_0[0] + value_1[0])
        new_label = value_0[1]
        if value_0[0] < value_1[0]:
            new_label = value_1[1]
        new_value = [(value_0[0] + value_1[0]), new_label]
        histogram[new_key] = new_value

        return histogram
