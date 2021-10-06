"""
Author: Haoyin Xu
"""
import time
import numpy as np
import pandas as pd
from numpy.random import permutation
from sklearn.model_selection import train_test_split
from sdtf import CascadeStreamForest


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


def experiment_csf():
    """Runs experiments for Cascade Stream Forest"""
    csf_l = []
    train_time_l = []
    test_time_l = []

    csf = CascadeStreamForest()

    for i in range(23):
        X_t = X_r[i * 100 : (i + 1) * 100]
        y_t = y_r[i * 100 : (i + 1) * 100]

        # Train the model
        start_time = time.perf_counter()
        csf.partial_fit(X_t, y_t, classes=[0, 1, 2])
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Test the model
        start_time = time.perf_counter()
        csf_l.append(prediction(csf))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    # Reformat the train times
    for i in range(1, 23):
        train_time_l[i] += train_time_l[i - 1]

    return csf_l, train_time_l, test_time_l


# prepare splice DNA data
df = pd.read_csv("../dna.csv")
X = df.drop(["Label"], axis=1).values
y = df["Label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Perform experiments
csf_acc_l = []
csf_train_t_l = []
csf_test_t_l = []
for i in range(10):
    p = permutation(X_train.shape[0])

    X_r = X_train[p]
    y_r = y_train[p]

    csf_acc, csf_train_t, csf_test_t = experiment_csf()
    csf_acc_l.append(csf_acc)
    csf_train_t_l.append(csf_train_t)
    csf_test_t_l.append(csf_test_t)

    write_result("../csf/splice_acc.txt", csf_acc_l)
    write_result("../csf/splice_train_t.txt", csf_train_t_l)
    write_result("../csf/splice_test_t.txt", csf_test_t_l)
