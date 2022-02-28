"""
Coauthors: Nick Hahn
           Will LeVine
"""
import numpy as np
from sklearn.datasets import make_blobs
from skgarden import MondrianForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sdtf import StreamDecisionForest
from sklearn.tree import DecisionTreeClassifier
from proglearn.forest import LifelongClassificationForest
from river import tree
import ast

from toolbox import *


def load_result(filename):
    """Loads results from specified file"""
    inputs = open(filename, "r")
    lines = inputs.readlines()
    ls = []
    for line in lines:
        ls.append(ast.literal_eval(line))
    return ls


def run(exp_type, classifiers, mc_rep, n_test):
    """Runs XOR, R-XOR(XNOR), XOR experiment mc_rep times"""
    n_xor = 750
    n_rxor = 750
    if exp_type == "XNOR":
        angle = np.pi / 2
    else:  # R-XOR
        angle = np.pi / 4
    mean_error = np.zeros((10, 90))
    for i in range(mc_rep):
        errors = experiment(angle, classifiers, n_xor, n_rxor, n_test)
        mean_error += errors
    return mean_error / mc_rep


def ht_partial_fit(ht, X, y):
    """Helper function to partially fit Hoeffding Tree"""
    for j in range(X.shape[0]):
        X_t = X[j]
        y_t = y[j]
        idx = range(25)
        X_t = dict(zip(idx, X_t))
        ht.learn_one(X_t, y_t)


def ht_predict(ht, test_x_xor, test_x_rxor):
    """Helper function to generate Hoeffding Tree predictions"""
    xor_y_hat = np.zeros(test_x_xor.shape[0])
    rxor_y_hat = np.zeros(test_x_rxor.shape[0])
    for j in range(test_x_xor.shape[0]):
        xor_y_hat[j] = ht.predict_one(test_x_xor[j])
        rxor_y_hat[j] = ht.predict_one(test_x_rxor[j])
    return xor_y_hat, rxor_y_hat


def experiment(angle, classifiers, n_xor, n_rxor, n_test):
    """Perform XOR RXOR(XNOR) XOR experiment"""
    X_xor, y_xor = generate_gaussian_parity(n_xor)
    X_rxor, y_rxor = generate_gaussian_parity(n_rxor, angle_params=angle)
    X_xor_2, y_xor_2 = generate_gaussian_parity(n_xor)
    test_x_xor, test_y_xor = generate_gaussian_parity(n_test)
    test_x_rxor, test_y_rxor = generate_gaussian_parity(n_test, angle_params=angle)
    X_stream = np.concatenate((X_xor, X_rxor, X_xor_2), axis=0)
    y_stream = np.concatenate((y_xor, y_rxor, y_xor_2), axis=0)

    # Instantiate classifiers
    if classifiers[0] == 1:
        ht = tree.HoeffdingTreeClassifier(grace_period=2, split_confidence=1e-01)
    if classifiers[1] == 1:
        mf = MondrianForestClassifier(n_estimators=10)
    if classifiers[2] == 1:
        sdt = DecisionTreeClassifier()
    if classifiers[3] == 1:
        sdf = StreamDecisionForest()
    if classifiers[4] == 1:
        synf = LifelongClassificationForest(default_n_estimators=10)

    errors = np.zeros((10, int(X_stream.shape[0] / 25)))

    for i in range(int(X_stream.shape[0] / 25)):
        X = X_stream[i * 25 : (i + 1) * 25]
        y = y_stream[i * 25 : (i + 1) * 25]

        # Hoeffding Tree Classifier
        if classifiers[0] == 1:
            ht_partial_fit(ht, X, y)
            ht_xor_y_hat, ht_rxor_y_hat = ht_predict(ht, test_x_xor, test_x_rxor)
            errors[0, i] = 1 - np.mean(ht_xor_y_hat == test_y_xor)
            errors[1, i] = 1 - np.mean(ht_rxor_y_hat == test_y_rxor)

        # Mondrian Forest Classifier
        if classifiers[1] == 1:
            mf.partial_fit(X, y)
            mf_xor_y_hat = mf.predict(test_x_xor)
            mf_rxor_y_hat = mf.predict(test_x_rxor)
            errors[2, i] = 1 - np.mean(mf_xor_y_hat == test_y_xor)
            errors[3, i] = 1 - np.mean(mf_rxor_y_hat == test_y_rxor)

        # Stream Decision Tree Classifier
        if classifiers[2] == 1:
            sdt.partial_fit(X, y, classes=[0, 1])
            sdt_xor_y_hat = sdt.predict(test_x_xor)
            sdt_rxor_y_hat = sdt.predict(test_x_rxor)
            errors[4, i] = 1 - np.mean(sdt_xor_y_hat == test_y_xor)
            errors[5, i] = 1 - np.mean(sdt_rxor_y_hat == test_y_rxor)

        # Stream Decision Forest Classifier
        if classifiers[3] == 1:
            sdf.partial_fit(X, y, classes=[0, 1])
            sdf_xor_y_hat = sdf.predict(test_x_xor)
            sdf_rxor_y_hat = sdf.predict(test_x_rxor)
            errors[6, i] = 1 - np.mean(sdf_xor_y_hat == test_y_xor)
            errors[7, i] = 1 - np.mean(sdf_rxor_y_hat == test_y_rxor)

        # Synergistic Forest Classifier
        if classifiers[4] == 1:
            if i == 0:
                synf.add_task(X, y, n_estimators=10, task_id=0)
                synf_xor_y_hat = synf.predict(test_x_xor, task_id=0)
                synf_rxor_y_hat = synf.predict(test_x_rxor, task_id=0)
            elif i < (n_xor / 25):
                synf.update_task(X, y, task_id=0)
                synf_xor_y_hat = synf.predict(test_x_xor, task_id=0)
                synf_rxor_y_hat = synf.predict(test_x_rxor, task_id=0)
            elif i == (n_xor / 25):
                synf.add_task(X, y, n_estimators=10, task_id=1)
                synf_xor_y_hat = synf.predict(test_x_xor, task_id=1)
                synf_rxor_y_hat = synf.predict(test_x_rxor, task_id=1)
            elif i < (n_xor + n_rxor) / 25:
                synf.update_task(X, y, task_id=1)
                synf_xor_y_hat = synf.predict(test_x_xor, task_id=1)
                synf_rxor_y_hat = synf.predict(test_x_rxor, task_id=1)
            elif i < (2 * n_xor + n_rxor) / 25:
                synf.update_task(X, y, task_id=0)
                synf_xor_y_hat = synf.predict(test_x_xor, task_id=0)
                synf_rxor_y_hat = synf.predict(test_x_rxor, task_id=0)

            errors[8, i] = 1 - np.mean(synf_xor_y_hat == test_y_xor)
            errors[9, i] = 1 - np.mean(synf_rxor_y_hat == test_y_rxor)

    return errors


def r_xor_plot_error(mean_error):
    """Plot Generalization Errors"""
    algorithms = [
        "Hoeffding Tree ",
        "Mondrian Forest",
        "Stream Decision Tree",
        "Stream Decision Forest",
        "Synergistic Forest",
    ]
    fontsize = 30
    labelsize = 28
    ls = ["-", "--"]
    colors = sns.color_palette("bright")
    fig = plt.figure(figsize=(21, 14))
    gs = fig.add_gridspec(14, 21)
    ax1 = fig.add_subplot(gs[7:, :6])
    # Hoeffding Tree XOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[0],
        label=algorithms[0],
        c=colors[4],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Mondrian Forest XOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[2],
        label=algorithms[1],
        c=colors[5],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Stream Decision Tree XOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[4],
        label=algorithms[2],
        c=colors[2],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Stream Decision Forest XOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[6],
        label=algorithms[3],
        c=colors[3],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Synergistic Forest XOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[8],
        label=algorithms[4],
        c=colors[9],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    ax1.set_ylabel("Generalization Error (XOR)", fontsize=fontsize)
    ax1.set_xlabel("Total Sample Size", fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.set_yticks([0.1, 0.3, 0.5, 0.9])
    ax1.set_xticks([750, 1500, 2250])
    ax1.axvline(x=750, c="gray", linewidth=1.5, linestyle="dashed")
    ax1.axvline(x=1500, c="gray", linewidth=1.5, linestyle="dashed")

    right_side = ax1.spines["right"]
    right_side.set_visible(False)
    top_side = ax1.spines["top"]
    top_side.set_visible(False)

    ax1.text(200, np.mean(ax1.get_ylim()) + 0.5, "XOR", fontsize=26)
    ax1.text(850, np.mean(ax1.get_ylim()) + 0.5, "RXOR", fontsize=26)
    ax1.text(1700, np.mean(ax1.get_ylim()) + 0.5, "XOR", fontsize=26)

    ######## RXOR
    ax1 = fig.add_subplot(gs[7:, 8:14])
    # Hoeffding Tree R-XOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[1],
        label=algorithms[0],
        c=colors[4],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Mondrian Forest R-XOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[3],
        label=algorithms[1],
        c=colors[5],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Stream Decision Tree R-XOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[5],
        label=algorithms[2],
        c=colors[2],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Stream Decision Forest R-XOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[7],
        label=algorithms[3],
        c=colors[3],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Synergistic Forest R-XOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[9],
        label=algorithms[4],
        c=colors[9],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )

    ax1.set_ylabel("Generalization Error (%s)" % "RXOR", fontsize=fontsize)
    ax1.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize=20, frameon=False)
    ax1.set_xlabel("Total Sample Size", fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.set_yticks([0.1, 0.3, 0.5, 0.9])
    ax1.set_xticks([750, 1500, 2250])
    ax1.axvline(x=750, c="gray", linewidth=1.5, linestyle="dashed")
    ax1.axvline(x=1500, c="gray", linewidth=1.5, linestyle="dashed")
    right_side = ax1.spines["right"]
    right_side.set_visible(False)
    top_side = ax1.spines["top"]
    top_side.set_visible(False)

    ax1.text(200, np.mean(ax1.get_ylim()) + 0.5, "XOR", fontsize=26)
    ax1.text(850, np.mean(ax1.get_ylim()) + 0.5, "RXOR", fontsize=26)
    ax1.text(1700, np.mean(ax1.get_ylim()) + 0.5, "XOR", fontsize=26)


def xnor_plot_error(mean_error):
    """Plot Generalization Errors"""
    algorithms = [
        "Hoeffding Tree ",
        "Mondrian Forest",
        "Stream Decision Tree",
        "Stream Decision Forest",
        "Synergistic Forest",
    ]
    fontsize = 30
    labelsize = 28
    ls = ["-", "--"]
    colors = sns.color_palette("bright")
    fig = plt.figure(figsize=(21, 14))
    gs = fig.add_gridspec(14, 21)
    ax1 = fig.add_subplot(gs[7:, :6])
    # Hoeffding Tree XOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[0],
        label=algorithms[0],
        c=colors[4],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Mondrian Forest XOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[2],
        label=algorithms[1],
        c=colors[5],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Stream Decision Tree XOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[4],
        label=algorithms[2],
        c=colors[2],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Stream Decision Forest XOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[6],
        label=algorithms[3],
        c=colors[3],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Synergistic Forest XOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[8],
        label=algorithms[4],
        c=colors[9],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    ax1.set_ylabel("Generalization Error (XOR)", fontsize=fontsize)
    ax1.set_xlabel("Total Sample Size", fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.set_yticks([0.1, 0.3, 0.5, 0.9])
    ax1.set_xticks([750, 1500, 2250])
    ax1.axvline(x=750, c="gray", linewidth=1.5, linestyle="dashed")
    ax1.axvline(x=1500, c="gray", linewidth=1.5, linestyle="dashed")

    right_side = ax1.spines["right"]
    right_side.set_visible(False)
    top_side = ax1.spines["top"]
    top_side.set_visible(False)

    ax1.text(200, np.mean(ax1.get_ylim()) + 0.5, "XOR", fontsize=26)
    ax1.text(850, np.mean(ax1.get_ylim()) + 0.5, "XNOR", fontsize=26)
    ax1.text(1700, np.mean(ax1.get_ylim()) + 0.5, "XOR", fontsize=26)

    ######## XNOR
    ax1 = fig.add_subplot(gs[7:, 8:14])
    # Hoeffding Tree XNOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[1],
        label=algorithms[0],
        c=colors[4],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Mondrian Forest XNOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[3],
        label=algorithms[1],
        c=colors[5],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Stream Decision Tree XNOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[5],
        label=algorithms[2],
        c=colors[2],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Stream Decision Forest XNOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[7],
        label=algorithms[3],
        c=colors[3],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )
    # Synergistic Forest XNOR
    ax1.plot(
        (100 * np.arange(0.25, 22.75, step=0.25)).astype(int),
        mean_error[9],
        label=algorithms[4],
        c=colors[9],
        ls=ls[np.sum(1 > 1).astype(int)],
        lw=3,
    )

    ax1.set_ylabel("Generalization Error (%s)" % "XNOR", fontsize=fontsize)
    ax1.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize=20, frameon=False)
    ax1.set_xlabel("Total Sample Size", fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.set_yticks([0.1, 0.3, 0.5, 0.9])
    ax1.set_xticks([750, 1500, 2250])
    ax1.axvline(x=750, c="gray", linewidth=1.5, linestyle="dashed")
    ax1.axvline(x=1500, c="gray", linewidth=1.5, linestyle="dashed")
    right_side = ax1.spines["right"]
    right_side.set_visible(False)
    top_side = ax1.spines["top"]
    top_side.set_visible(False)

    ax1.text(200, np.mean(ax1.get_ylim()) + 0.5, "XOR", fontsize=26)
    ax1.text(850, np.mean(ax1.get_ylim()) + 0.5, "XNOR", fontsize=26)
    ax1.text(1700, np.mean(ax1.get_ylim()) + 0.5, "XOR", fontsize=26)


def plot_xor_rxor_xor(num_data_points):
    """Visualize Gaussian XOR and Gaussian R-XOR Data"""
    colors = sns.color_palette("Dark2", n_colors=2)
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    xor_1, y_xor_1 = generate_gaussian_parity(num_data_points)
    xor_2, y_xor_2 = generate_gaussian_parity(num_data_points)
    r_xor, y_rxor = generate_gaussian_parity(num_data_points, angle_params=np.pi / 4)
    ax[0].scatter(xor_1[:, 0], xor_1[:, 1], c=get_colors(colors, y_xor_1), s=50)
    ax[1].scatter(r_xor[:, 0], r_xor[:, 1], c=get_colors(colors, y_rxor), s=50)
    ax[2].scatter(xor_2[:, 0], xor_2[:, 1], c=get_colors(colors, y_xor_2), s=50)

    ax[0].set_title("Gaussian XOR", fontsize=30)
    ax[1].set_title("Gaussian R-XOR", fontsize=30)
    ax[2].set_title("Gaussian XOR", fontsize=30)
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[2].set_yticks([])

    plt.show()


def plot_xor_xnor_xor(num_data_points):
    """Visualize Gaussian XOR and Gaussian XNOR Data"""
    colors = sns.color_palette("Dark2", n_colors=2)
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    xor_1, y_xor_1 = generate_gaussian_parity(num_data_points)
    xor_2, y_xor_2 = generate_gaussian_parity(num_data_points)
    r_xor, y_rxor = generate_gaussian_parity(num_data_points, angle_params=np.pi / 2)
    ax[0].scatter(xor_1[:, 0], xor_1[:, 1], c=get_colors(colors, y_xor_1), s=50)
    ax[1].scatter(r_xor[:, 0], r_xor[:, 1], c=get_colors(colors, y_rxor), s=50)
    ax[2].scatter(xor_2[:, 0], xor_2[:, 1], c=get_colors(colors, y_xor_2), s=50)

    ax[0].set_title("Gaussian XOR", fontsize=30)
    ax[1].set_title("Gaussian XNOR", fontsize=30)
    ax[2].set_title("Gaussian XOR", fontsize=30)
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[2].set_yticks([])

    plt.show()


# Additional plotting functions from proglearn/sims/gaussian_sim.py
def _generate_2d_rotation(theta=0):
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    return R


def generate_gaussian_parity(
    n_samples,
    centers=None,
    class_label=None,
    cluster_std=0.25,
    angle_params=None,
    random_state=None,
):
    """
    Generate 2-dimensional Gaussian XOR distribution.
    (Classic XOR problem but each point is the
    center of a Gaussian blob distribution)
    Parameters
    ----------
    n_samples : int
        Total number of points divided among the four
        clusters with equal probability.
    centers : array of shape [n_centers,2], optional (default=None)
        The coordinates of the ceneter of total n_centers blobs.
    class_label : array of shape [n_centers], optional (default=None)
        class label for each blob.
    cluster_std : float, optional (default=1)
        The standard deviation of the blobs.
    angle_params: float, optional (default=None)
        Number of radians to rotate the distribution by.
    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """

    if random_state != None:
        np.random.seed(random_state)

    if centers == None:
        centers = np.array([(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)])

    if class_label == None:
        class_label = [0, 1, 1, 0]

    blob_num = len(class_label)

    # get the number of samples in each blob with equal probability
    samples_per_blob = np.random.multinomial(
        n_samples, 1 / blob_num * np.ones(blob_num)
    )

    X, y = make_blobs(
        n_samples=samples_per_blob,
        n_features=2,
        centers=centers,
        cluster_std=cluster_std,
    )

    for blob in range(blob_num):
        y[np.where(y == blob)] = class_label[blob]

    if angle_params != None:
        R = _generate_2d_rotation(angle_params)
        X = X @ R

    return X, y


def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c
