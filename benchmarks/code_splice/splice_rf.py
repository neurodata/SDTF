"""
Author: Haoyin Xu
"""
import time
import numpy as np
import pandas as pd
from numpy.random import permutation
from sklearn.model_selection import train_test_split
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

    for i in range(23):
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


# prepare splice DNA data
df = pd.read_csv("dna.csv")
X = df.drop(["Label"], axis=1).values
y = df["Label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Perform experiments
rf_acc_l = []
rf_train_t_l = []
rf_test_t_l = []
for i in range(100):
    p = permutation(X_train.shape[0])

    X_r = X_train[p]
    y_r = y_train[p]

    rf_acc, rf_train_t, rf_test_t = experiment_rf()
    rf_acc_l.append(rf_acc)
    rf_train_t_l.append(rf_train_t)
    rf_test_t_l.append(rf_test_t)

    write_result("../rf/splice_acc.txt", rf_acc_l)
    write_result("../rf/splice_train_t.txt", rf_train_t_l)
    write_result("../rf/splice_test_t.txt", rf_test_t_l)
