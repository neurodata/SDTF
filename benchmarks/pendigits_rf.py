"""
Author: Haoyin Xu
"""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def write_result(filename, acc_ls):
    """Writes results to specified text file"""
    output = open(filename, "w")
    for acc in acc_ls:
        output.write(str(acc) + "\n")


def prediction(classifier):
    """Generates predictions from model"""
    predictions = classifier.predict(X_test)

    p_t = 0
    for i in range(X_test.shape[0]):
        if predictions[i] == y_test[i]:
            p_t += 1

    return p_t / X_test.shape[0]


def experiment_rf():
    """Runs experiments for Random Forest"""
    rf_l = []
    train_time_l = []
    test_time_l = []

    rf = RandomForestClassifier()

    for i in range(74):
        X_t = X_r[: (i + 1) * 100]
        y_t = y_r[: (i + 1) * 100]

        # Train the model
        start_time = time.perf_counter()
        rf.fit(X_t, y_t)
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Test the model
        start_time = time.perf_counter()
        rf_l.append(prediction(rf))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    return rf_l, train_time_l, test_time_l


# prepare pendigits data
pendigits = pd.read_csv("pendigits.tra", header=None)
pendigits_test = pd.read_csv("pendigits.tes", header=None)
X_test = pendigits_test.iloc[:, :-1]
y_test = pendigits_test.iloc[:, -1]

# Perform experiments
rf_acc_l = []
rf_train_t_l = []
rf_test_t_l = []
for i in range(100):
    p = pendigits.sample(frac=1)
    X_r = p.iloc[:, :-1]
    y_r = p.iloc[:, -1]

    rf_acc, rf_train_t, rf_test_t = experiment_rf()
    rf_acc_l.append(rf_acc)
    rf_train_t_l.append(rf_train_t)
    rf_test_t_l.append(rf_test_t)

    write_result("rf/pendigits_acc.txt", rf_acc_l)
    write_result("rf/pendigits_train_t.txt", rf_train_t_l)
    write_result("rf/pendigits_test_t.txt", rf_test_t_l)
