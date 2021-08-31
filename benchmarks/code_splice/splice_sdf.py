"""
Author: Haoyin Xu
"""
import time
import numpy as np
import pandas as pd
from numpy.random import permutation
from sklearn.model_selection import train_test_split
from spdt import StreamForest


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

    sdf = StreamForest()

    for i in range(23):
        X_t = X_r[i * 100 : (i + 1) * 100]
        y_t = y_r[i * 100 : (i + 1) * 100]

        # Train the model
        start_time = time.perf_counter()
        sdf.fit(X_t, y_t)
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Test the model
        start_time = time.perf_counter()
        sdf_l.append(prediction(sdf))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    # Reformat the train times
    for i in range(1, 23):
        train_time_l[i] += train_time_l[i - 1]

    return sdf_l, train_time_l, test_time_l


# prepare splice DNA data
df = pd.read_csv("dna.csv")
X = df.drop(["Label"], axis=1).values
y = df["Label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Perform experiments
sdf_acc_l = []
sdf_train_t_l = []
sdf_test_t_l = []
for i in range(100):
    p = permutation(X_train.shape[0])

    X_r = X_train[p]
    y_r = y_train[p]

    sdf_acc, sdf_train_t, sdf_test_t = experiment_sdf()
    sdf_acc_l.append(sdf_acc)
    sdf_train_t_l.append(sdf_train_t)
    sdf_test_t_l.append(sdf_test_t)

    write_result("../sdf/splice_acc.txt", sdf_acc_l)
    write_result("../sdf/splice_train_t.txt", sdf_train_t_l)
    write_result("../sdf/splice_test_t.txt", sdf_test_t_l)
