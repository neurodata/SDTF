"""
Author: Haoyin Xu
"""
import time
import numpy as np
import pandas as pd
from river import tree


def write_result(filename, acc_ls):
    """Writes results to specified text file"""
    output = open(filename, "w")
    for acc in acc_ls:
        output.write(str(acc) + "\n")


def experiment_ht():
    """Runs experiments for Hoeffding Tree"""
    ht_l = []
    train_time_l = []
    test_time_l = []

    ht = tree.HoeffdingTreeClassifier()

    for i in range(7400):
        X_t = X_r.iloc[i]
        y_t = y_r.iloc[i]

        idx = range(16)
        X_t = dict(zip(idx, X_t))

        start_time = time.perf_counter()
        ht.learn_one(X_t, y_t)
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        if i > 0 and (i + 1) % 100 == 0:
            p_t = 0.0
            start_time = time.perf_counter()
            for j in range(X_test.shape[0]):
                y_pred = ht.predict_one(X_test.iloc[j])
                if y_pred == y_test.iloc[j]:
                    p_t += 1
            ht_l.append(p_t / X_test.shape[0])
            end_time = time.perf_counter()
            test_time_l.append(end_time - start_time)

    # Reformat the train times
    new_train_time_l = []
    for i in range(1, 7400):
        train_time_l[i] += train_time_l[i - 1]
        if i > 0 and (i + 1) % 100 == 0:
            new_train_time_l.append(train_time_l[i])
    train_time_l = new_train_time_l

    return ht_l, train_time_l, test_time_l


# prepare pendigits data
pendigits = pd.read_csv("pendigits.tra", header=None)
pendigits_test = pd.read_csv("pendigits.tes", header=None)
X_test = pendigits_test.iloc[:, :-1]
y_test = pendigits_test.iloc[:, -1]

# Perform experiments
ht_acc_l = []
ht_train_t_l = []
ht_test_t_l = []
for i in range(100):
    p = pendigits.sample(frac=1)
    X_r = p.iloc[:, :-1]
    y_r = p.iloc[:, -1]

    ht_acc, ht_train_t, ht_test_t = experiment_ht()
    ht_acc_l.append(ht_acc)
    ht_train_t_l.append(ht_train_t)
    ht_test_t_l.append(ht_test_t)

    write_result("ht/pendigits_acc.txt", ht_acc_l)
    write_result("ht/pendigits_train_t.txt", ht_train_t_l)
    write_result("ht/pendigits_test_t.txt", ht_test_t_l)
