"""
Author: Haoyin Xu
"""
import os
import pickle
import gzip
import shutil
import numpy as np
from sklearn.metrics import accuracy_score


def write_result(filename, acc_ls, tuple=False):
    """Writes results to specified text file"""
    if not tuple:
        output = open(filename + ".txt", "a")
        for acc in acc_ls:
            output.write(str(acc) + "\n")
    else:
        first = open(filename + "_first.txt", "a")
        second = open(filename + "_second.txt", "a")
        for acc in acc_ls:
            first.write(str(np.array(acc)[:, 0].tolist()) + "\n")
            second.write(str(np.array(acc)[:, 1].tolist()) + "\n")


def prediction(classifier, X_test, y_test):
    """Generates predictions from model"""
    y_preds = classifier.predict(X_test)

    return accuracy_score(y_preds, y_test)


def node_count(classifier, forest=False):
    """Records the number of nodes"""
    num_nodes = 0
    if forest:
        for tree in classifier.estimators_:
            num_nodes += tree.tree_.node_count
    else:
        num_nodes = classifier.tree_.node_count

    return num_nodes


def clf_size(classifier, file_name):
    """Records the classifier size"""
    p = pickle.dump(classifier, open(file_name, "wb"))
    file_size = os.path.getsize(file_name)

    with open(file_name, "rb") as f_in:
        with gzip.open(file_name + ".gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    zip_size = os.path.getsize(file_name)

    return file_size, zip_size
