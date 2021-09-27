"""
Author: Haoyin Xu
"""
import time
import numpy as np
import pandas as pd
from spdt import StreamDecisionForest


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


def experiment_sdf():
    """Runs experiments for Stream Decision Forest"""
    sdf_l = []
    train_time_l = []
    test_time_l = []

    sdf = StreamDecisionForest()

    for i in range(74):
        X_t = X_r[i * 100 : (i + 1) * 100]
        y_t = y_r[i * 100 : (i + 1) * 100]

        # Train the model
        start_time = time.perf_counter()
        sdf.partial_fit(X_t, y_t, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Test the model
        start_time = time.perf_counter()
        sdf_l.append(prediction(sdf))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    # Reformat the train times
    for i in range(1, 74):
        train_time_l[i] += train_time_l[i - 1]

    return sdf_l, train_time_l, test_time_l


# prepare pendigits data
pendigits = pd.read_csv("../pendigits.tra", header=None)
pendigits_test = pd.read_csv("../pendigits.tes", header=None)
X_test = pendigits_test.iloc[:, :-1]
y_test = pendigits_test.iloc[:, -1]

# Perform experiments
sdf_acc_l = []
sdf_train_t_l = []
sdf_test_t_l = []
for i in range(10):
    p = pendigits.sample(frac=1)
    X_r = p.iloc[:, :-1]
    y_r = p.iloc[:, -1]

    sdf_acc, sdf_train_t, sdf_test_t = experiment_sdf()
    sdf_acc_l.append(sdf_acc)
    sdf_train_t_l.append(sdf_train_t)
    sdf_test_t_l.append(sdf_test_t)

    write_result("../sdf/pendigits_acc.txt", sdf_acc_l)
    write_result("../sdf/pendigits_train_t.txt", sdf_train_t_l)
    write_result("../sdf/pendigits_test_t.txt", sdf_test_t_l)
