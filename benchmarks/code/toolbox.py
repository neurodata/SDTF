"""
Author: Haoyin Xu
"""
import os
import pickle
from sklearn.metrics import accuracy_score


def write_result(filename, acc_ls):
    """Writes results to specified text file"""
    output = open(filename, "a")
    for acc in acc_ls:
        output.write(str(acc) + "\n")


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
    p = pickle.dump(classifier, open(file_name, "w"))
    file_size = os.path.getsize(file_name)

    return file_size
