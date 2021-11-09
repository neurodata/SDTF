"""
Author: Haoyin Xu
"""
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from river import tree
from skgarden import MondrianForestClassifier
from sdtf import StreamDecisionForest, CascadeStreamForest


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


def experiment_ht():
    """Runs experiments for Hoeffding Tree"""
    ht_l = []
    train_time_l = []
    test_time_l = []

    ht = tree.HoeffdingTreeClassifier(grace_period=2)

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


def experiment_mf():
    """Runs experiments for Mondrian Forest"""
    mf_l = []
    train_time_l = []
    test_time_l = []

    mf = MondrianForestClassifier(n_estimators=10)

    for i in range(74):
        X_t = X_r[i * 100 : (i + 1) * 100]
        y_t = y_r[i * 100 : (i + 1) * 100]

        # Train the model
        start_time = time.perf_counter()
        mf.partial_fit(X_t, y_t)
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Test the model
        start_time = time.perf_counter()
        mf_l.append(prediction(mf))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    # Reformat the train times
    for i in range(1, 74):
        train_time_l[i] += train_time_l[i - 1]

    return mf_l, train_time_l, test_time_l


def experiment_sdt():
    """Runs experiments for Stream Decision Tree"""
    sdt_l = []
    train_time_l = []
    test_time_l = []

    sdt = DecisionTreeClassifier()

    for i in range(74):
        X_t = X_r[i * 100 : (i + 1) * 100]
        y_t = y_r[i * 100 : (i + 1) * 100]

        # Train the model
        start_time = time.perf_counter()
        sdt.partial_fit(X_t, y_t, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Test the model
        start_time = time.perf_counter()
        sdt_l.append(prediction(sdt))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    # Reformat the train times
    for i in range(1, 74):
        train_time_l[i] += train_time_l[i - 1]

    return sdt_l, train_time_l, test_time_l


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


def experiment_csf():
    """Runs experiments for Cascade Stream Forest"""
    csf_l = []
    train_time_l = []
    test_time_l = []

    csf = CascadeStreamForest()

    for i in range(74):
        X_t = X_r[i * 100 : (i + 1) * 100]
        y_t = y_r[i * 100 : (i + 1) * 100]

        # Train the model
        start_time = time.perf_counter()
        csf.partial_fit(X_t, y_t, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Test the model
        start_time = time.perf_counter()
        csf_l.append(prediction(csf))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    # Reformat the train times
    for i in range(1, 74):
        train_time_l[i] += train_time_l[i - 1]

    return csf_l, train_time_l, test_time_l


# Prepare pendigits data
pendigits = pd.read_csv("../pendigits.tra", header=None)
pendigits_test = pd.read_csv("../pendigits.tes", header=None)
X_test = pendigits_test.iloc[:, :-1]
y_test = pendigits_test.iloc[:, -1]

# Parse classifier choices
parser = argparse.ArgumentParser()
parser.add_argument("-all", help="all classifiers", required=False, action="store_true")
parser.add_argument("-dt", help="decision forests", required=False, action="store_true")
parser.add_argument("-rf", help="random forests", required=False, action="store_true")
parser.add_argument("-ht", help="hoeffding trees", required=False, action="store_true")
parser.add_argument("-mf", help="mondrian forests", required=False, action="store_true")
parser.add_argument(
    "-sdt", help="stream decision trees", required=False, action="store_true"
)
parser.add_argument(
    "-sdf", help="stream decision forests", required=False, action="store_true"
)
parser.add_argument(
    "-csf", help="cascade stream forests", required=False, action="store_true"
)
args = parser.parse_args()

# Perform experiments
if args.all or args.dt:
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

        write_result("../results/dt/pendigits_acc.txt", dt_acc_l)
        write_result("../results/dt/pendigits_train_t.txt", dt_train_t_l)
        write_result("../results/dt/pendigits_test_t.txt", dt_test_t_l)

if args.all or args.rf:
    rf_acc_l = []
    rf_train_t_l = []
    rf_test_t_l = []
    for i in range(10):
        p = pendigits.sample(frac=1)
        X_r = p.iloc[:, :-1]
        y_r = p.iloc[:, -1]

        rf_acc, rf_train_t, rf_test_t = experiment_rf()
        rf_acc_l.append(rf_acc)
        rf_train_t_l.append(rf_train_t)
        rf_test_t_l.append(rf_test_t)

        write_result("../results/rf/pendigits_acc.txt", rf_acc_l)
        write_result("../results/rf/pendigits_train_t.txt", rf_train_t_l)
        write_result("../results/rf/pendigits_test_t.txt", rf_test_t_l)

if args.all or args.ht:
    ht_acc_l = []
    ht_train_t_l = []
    ht_test_t_l = []
    for i in range(10):
        p = pendigits.sample(frac=1)
        X_r = p.iloc[:, :-1]
        y_r = p.iloc[:, -1]

        ht_acc, ht_train_t, ht_test_t = experiment_ht()
        ht_acc_l.append(ht_acc)
        ht_train_t_l.append(ht_train_t)
        ht_test_t_l.append(ht_test_t)

        write_result("../results/ht/pendigits_acc.txt", ht_acc_l)
        write_result("../results/ht/pendigits_train_t.txt", ht_train_t_l)
        write_result("../results/ht/pendigits_test_t.txt", ht_test_t_l)

if args.all or args.mf:
    mf_acc_l = []
    mf_train_t_l = []
    mf_test_t_l = []
    for i in range(10):
        p = pendigits.sample(frac=1)
        X_r = p.iloc[:, :-1]
        y_r = p.iloc[:, -1]

        mf_acc, mf_train_t, mf_test_t = experiment_mf()
        mf_acc_l.append(mf_acc)
        mf_train_t_l.append(mf_train_t)
        mf_test_t_l.append(mf_test_t)

        write_result("../results/mf/pendigits_acc.txt", mf_acc_l)
        write_result("../results/mf/pendigits_train_t.txt", mf_train_t_l)
        write_result("../results/mf/pendigits_test_t.txt", mf_test_t_l)

if args.all or args.sdt:
    sdt_acc_l = []
    sdt_train_t_l = []
    sdt_test_t_l = []
    for i in range(10):
        p = pendigits.sample(frac=1)
        X_r = p.iloc[:, :-1]
        y_r = p.iloc[:, -1]

        sdt_acc, sdt_train_t, sdt_test_t = experiment_sdt()
        sdt_acc_l.append(sdt_acc)
        sdt_train_t_l.append(sdt_train_t)
        sdt_test_t_l.append(sdt_test_t)

        write_result("../results/sdt/pendigits_acc.txt", sdt_acc_l)
        write_result("../results/sdt/pendigits_train_t.txt", sdt_train_t_l)
        write_result("../results/sdt/pendigits_test_t.txt", sdt_test_t_l)

if args.all or args.sdf:
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

        write_result("../results/sdf/pendigits_acc.txt", sdf_acc_l)
        write_result("../results/sdf/pendigits_train_t.txt", sdf_train_t_l)
        write_result("../results/sdf/pendigits_test_t.txt", sdf_test_t_l)

if args.all or args.csf:
    csf_acc_l = []
    csf_train_t_l = []
    csf_test_t_l = []
    for i in range(10):
        p = pendigits.sample(frac=1)
        X_r = p.iloc[:, :-1]
        y_r = p.iloc[:, -1]

        csf_acc, csf_train_t, csf_test_t = experiment_csf()
        csf_acc_l.append(csf_acc)
        csf_train_t_l.append(csf_train_t)
        csf_test_t_l.append(csf_test_t)

        write_result("../results/csf/pendigits_acc.txt", csf_acc_l)
        write_result("../results/csf/pendigits_train_t.txt", csf_train_t_l)
        write_result("../results/csf/pendigits_test_t.txt", csf_test_t_l)
