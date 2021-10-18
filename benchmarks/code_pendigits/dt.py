"""
Author: Haoyin Xu
"""
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


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


def experiment_dt():
    """Runs experiments for Batch Decision Tree"""
    dt_l = []
    train_time_l = []
    test_time_l = []

    dt = DecisionTreeClassifier()

    for i in range(74):
        X_t = X_r[: (i + 1) * 100]
        y_t = y_r[: (i + 1) * 100]

        # Train the model
        start_time = time.perf_counter()
        dt.fit(X_t, y_t)
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Test the model
        start_time = time.perf_counter()
        dt_l.append(prediction(dt))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    return dt_l, train_time_l, test_time_l


# prepare pendigits data
pendigits = pd.read_csv("../pendigits.tra", header=None)
pendigits_test = pd.read_csv("../pendigits.tes", header=None)
X_test = pendigits_test.iloc[:, :-1]
y_test = pendigits_test.iloc[:, -1]

# Perform experiments
dt_acc_l = []
dt_train_t_l = []
dt_test_t_l = []
for i in range(10):
    p = pendigits.sample(frac=1)
    X_r = p.iloc[:, :-1]
    y_r = p.iloc[:, -1]

    dt_acc, dt_train_t, dt_test_t = experiment_dt()
    dt_acc_l.append(dt_acc)
    dt_train_t_l.append(dt_train_t)
    dt_test_t_l.append(dt_test_t)

    write_result("../dt/pendigits_acc.txt", dt_acc_l)
    write_result("../dt/pendigits_train_t.txt", dt_train_t_l)
    write_result("../dt/pendigits_test_t.txt", dt_test_t_l)
