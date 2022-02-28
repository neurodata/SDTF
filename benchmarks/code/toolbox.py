"""
Author: Haoyin Xu
"""
from sklearn.metrics import accuracy_score


def write_result(filename, acc_ls):
    """Writes results to specified text file"""
    output = open(filename, "w")
    for acc in acc_ls:
        output.write(str(acc) + "\n")


def prediction(classifier, X_test, y_test):
    """Generates predictions from model"""
    y_preds = classifier.predict(X_test)

    return accuracy_score(y_preds, y_test)
