"""
Author: Haoyin Xu
"""
import time
import psutil
import argparse
import numpy as np
from numpy.random import permutation
import openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from river import tree
from skgarden import MondrianForestClassifier
from sdtf import StreamDecisionForest

from toolbox import *


def experiment_dt():
    """Runs experiments for Batch Decision Tree"""
    dt_l = []
    train_time_l = []
    test_time_l = []
    v_m_l = []
    s_m_l = []

    dt = DecisionTreeClassifier()

    for i in range(23):
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

        # Check memory
        v_m = psutil.virtual_memory()[2]
        v_m_l.append(v_m)
        s_m = psutil.swap_memory()[3]
        s_m_l.append(s_m)

    return dt_l, train_time_l, test_time_l, v_m_l, s_m_l


def experiment_rf():
    """Runs experiments for Random Forest"""
    rf_l = []
    train_time_l = []
    test_time_l = []
    v_m_l = []
    s_m_l = []

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

        # Check memory
        v_m = psutil.virtual_memory()[2]
        v_m_l.append(v_m)
        s_m = psutil.swap_memory()[3]
        s_m_l.append(s_m)

    return rf_l, train_time_l, test_time_l, v_m_l, s_m_l


def experiment_ht():
    """Runs experiments for Hoeffding Tree"""
    ht_l = []
    train_time_l = []
    test_time_l = []
    v_m_l = []
    s_m_l = []

    ht = tree.HoeffdingTreeClassifier(max_size=1000, grace_period=2)

    for i in range(X_train.shape[0]):
        X_t = X_r[i]
        y_t = y_r[i]

        idx = range(60)
        X_t = dict(zip(idx, X_t))

        start_time = time.perf_counter()
        ht.learn_one(X_t, y_t)
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        if i > 0 and (i + 1) % 100 == 0:
            p_t = 0.0
            start_time = time.perf_counter()
            for j in range(X_test.shape[0]):
                y_pred = ht.predict_one(X_test[j])
                if y_pred == y_test[j]:
                    p_t += 1
            ht_l.append(p_t / X_test.shape[0])
            end_time = time.perf_counter()
            test_time_l.append(end_time - start_time)

            # Check memory
            v_m = psutil.virtual_memory()[2]
            v_m_l.append(v_m)
            s_m = psutil.swap_memory()[3]
            s_m_l.append(s_m)

    # Reformat the train times
    new_train_time_l = []
    for i in range(1, X_train.shape[0]):
        train_time_l[i] += train_time_l[i - 1]
        if i > 0 and (i + 1) % 100 == 0:
            new_train_time_l.append(train_time_l[i])
    train_time_l = new_train_time_l

    return ht_l, train_time_l, test_time_l, v_m_l, s_m_l


def experiment_mf():
    """Runs experiments for Mondrian Forest"""
    mf_l = []
    train_time_l = []
    test_time_l = []
    v_m_l = []
    s_m_l = []

    mf = MondrianForestClassifier(n_estimators=10)

    for i in range(23):
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

        # Check memory
        v_m = psutil.virtual_memory()[2]
        v_m_l.append(v_m)
        s_m = psutil.swap_memory()[3]
        s_m_l.append(s_m)

    # Reformat the train times
    for i in range(1, 23):
        train_time_l[i] += train_time_l[i - 1]

    return mf_l, train_time_l, test_time_l, v_m_l, s_m_l


def experiment_sdt():
    """Runs experiments for Stream Decision Tree"""
    sdt_l = []
    train_time_l = []
    test_time_l = []
    v_m_l = []
    s_m_l = []

    sdt = DecisionTreeClassifier()

    for i in range(23):
        X_t = X_r[i * 100 : (i + 1) * 100]
        y_t = y_r[i * 100 : (i + 1) * 100]

        # Train the model
        start_time = time.perf_counter()
        sdt.partial_fit(X_t, y_t, classes=[0, 1, 2])
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Test the model
        start_time = time.perf_counter()
        sdt_l.append(prediction(sdt))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

        # Check memory
        v_m = psutil.virtual_memory()[2]
        v_m_l.append(v_m)
        s_m = psutil.swap_memory()[3]
        s_m_l.append(s_m)

    # Reformat the train times
    for i in range(1, 23):
        train_time_l[i] += train_time_l[i - 1]

    return sdt_l, train_time_l, test_time_l, v_m_l, s_m_l


def experiment_sdf():
    """Runs experiments for Stream Decision Forest"""
    sdf_l = []
    train_time_l = []
    test_time_l = []
    v_m_l = []
    s_m_l = []

    sdf = StreamDecisionForest()

    for i in range(23):
        X_t = X_r[i * 100 : (i + 1) * 100]
        y_t = y_r[i * 100 : (i + 1) * 100]

        # Train the model
        start_time = time.perf_counter()
        sdf.partial_fit(X_t, y_t, classes=[0, 1, 2])
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Test the model
        start_time = time.perf_counter()
        sdf_l.append(prediction(sdf))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

        # Check memory
        v_m = psutil.virtual_memory()[2]
        v_m_l.append(v_m)
        s_m = psutil.swap_memory()[3]
        s_m_l.append(s_m)

    # Reformat the train times
    for i in range(1, 23):
        train_time_l[i] += train_time_l[i - 1]

    return sdf_l, train_time_l, test_time_l, v_m_l, s_m_l


# Prepare splice DNA data
dataset = openml.datasets.get_dataset(46)
X, y, is_categorical, _ = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)
X_train, X_test, y_train, y_test = train_test_split(X, y)

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
args = parser.parse_args()

# Perform experiments
if args.all or args.dt:
    dt_acc_l = []
    dt_train_t_l = []
    dt_test_t_l = []
    dt_v_m_l = []
    dt_s_m_l = []
    for i in range(10):
        p = permutation(X_train.shape[0])

        X_r = X_train[p]
        y_r = y_train[p]

        dt_acc, dt_train_t, dt_test_t, dt_v_m, dt_s_m = experiment_dt()
        dt_acc_l.append(dt_acc)
        dt_train_t_l.append(dt_train_t)
        dt_test_t_l.append(dt_test_t)
        dt_v_m_l.append(dt_v_m)
        dt_s_m_l.append(dt_s_m)

        write_result("../results/dt/splice_acc.txt", dt_acc_l)
        write_result("../results/dt/splice_train_t.txt", dt_train_t_l)
        write_result("../results/dt/splice_test_t.txt", dt_test_t_l)
        write_result("../results/dt/splice_v_m.txt", dt_v_m_l)
        write_result("../results/dt/splice_s_m.txt", dt_s_m_l)

if args.all or args.rf:
    rf_acc_l = []
    rf_train_t_l = []
    rf_test_t_l = []
    rf_v_m_l = []
    rf_s_m_l = []
    for i in range(10):
        p = permutation(X_train.shape[0])

        X_r = X_train[p]
        y_r = y_train[p]

        rf_acc, rf_train_t, rf_test_t, rf_v_m, rf_s_m = experiment_rf()
        rf_acc_l.append(rf_acc)
        rf_train_t_l.append(rf_train_t)
        rf_test_t_l.append(rf_test_t)
        rf_v_m_l.append(rf_v_m)
        rf_s_m_l.append(rf_s_m)

        write_result("../results/rf/splice_acc.txt", rf_acc_l)
        write_result("../results/rf/splice_train_t.txt", rf_train_t_l)
        write_result("../results/rf/splice_test_t.txt", rf_test_t_l)
        write_result("../results/rf/splice_v_m.txt", rf_v_m_l)
        write_result("../results/rf/splice_s_m.txt", rf_s_m_l)

if args.all or args.ht:
    ht_acc_l = []
    ht_train_t_l = []
    ht_test_t_l = []
    ht_v_m_l = []
    ht_s_m_l = []
    for i in range(10):
        p = permutation(X_train.shape[0])

        X_r = X_train[p]
        y_r = y_train[p]

        ht_acc, ht_train_t, ht_test_t, ht_v_m, ht_s_m = experiment_ht()
        ht_acc_l.append(ht_acc)
        ht_train_t_l.append(ht_train_t)
        ht_test_t_l.append(ht_test_t)
        ht_v_m_l.append(ht_v_m)
        ht_s_m_l.append(ht_s_m)

        write_result("../results/ht/splice_acc.txt", ht_acc_l)
        write_result("../results/ht/splice_train_t.txt", ht_train_t_l)
        write_result("../results/ht/splice_test_t.txt", ht_test_t_l)
        write_result("../results/ht/splice_v_m.txt", ht_v_m_l)
        write_result("../results/ht/splice_s_m.txt", ht_s_m_l)

if args.all or args.mf:
    mf_acc_l = []
    mf_train_t_l = []
    mf_test_t_l = []
    mf_v_m_l = []
    mf_s_m_l = []
    for i in range(10):
        p = permutation(X_train.shape[0])

        X_r = X_train[p]
        y_r = y_train[p]

        mf_acc, mf_train_t, mf_test_t, mf_v_m, mf_s_m = experiment_mf()
        mf_acc_l.append(mf_acc)
        mf_train_t_l.append(mf_train_t)
        mf_test_t_l.append(mf_test_t)
        mf_v_m_l.append(mf_v_m)
        mf_s_m_l.append(mf_s_m)

        write_result("../results/mf/splice_acc.txt", mf_acc_l)
        write_result("../results/mf/splice_train_t.txt", mf_train_t_l)
        write_result("../results/mf/splice_test_t.txt", mf_test_t_l)
        write_result("../results/mf/splice_v_m.txt", mf_v_m_l)
        write_result("../results/mf/splice_s_m.txt", mf_s_m_l)

if args.all or args.sdt:
    sdt_acc_l = []
    sdt_train_t_l = []
    sdt_test_t_l = []
    sdt_v_m_l = []
    sdt_s_m_l = []
    for i in range(10):
        p = permutation(X_train.shape[0])

        X_r = X_train[p]
        y_r = y_train[p]

        sdt_acc, sdt_train_t, sdt_test_t, sdt_v_m, sdt_s_m = experiment_sdt()
        sdt_acc_l.append(sdt_acc)
        sdt_train_t_l.append(sdt_train_t)
        sdt_test_t_l.append(sdt_test_t)
        sdt_v_m_l.append(sdt_v_m)
        sdt_s_m_l.append(sdt_s_m)

        write_result("../results/sdt/splice_acc.txt", sdt_acc_l)
        write_result("../results/sdt/splice_train_t.txt", sdt_train_t_l)
        write_result("../results/sdt/splice_test_t.txt", sdt_test_t_l)
        write_result("../results/sdt/splice_v_m.txt", sdt_v_m_l)
        write_result("../results/sdt/splice_s_m.txt", sdt_s_m_l)

if args.all or args.sdf:
    sdf_acc_l = []
    sdf_train_t_l = []
    sdf_test_t_l = []
    sdf_v_m_l = []
    sdf_s_m_l = []
    for i in range(10):
        p = permutation(X_train.shape[0])

        X_r = X_train[p]
        y_r = y_train[p]

        sdf_acc, sdf_train_t, sdf_test_t, sdf_v_m, sdf_s_m = experiment_sdf()
        sdf_acc_l.append(sdf_acc)
        sdf_train_t_l.append(sdf_train_t)
        sdf_test_t_l.append(sdf_test_t)
        sdf_v_m_l.append(sdf_v_m)
        sdf_s_m_l.append(sdf_s_m)

        write_result("../results/sdf/splice_acc.txt", sdf_acc_l)
        write_result("../results/sdf/splice_train_t.txt", sdf_train_t_l)
        write_result("../results/sdf/splice_test_t.txt", sdf_test_t_l)
        write_result("../results/sdf/splice_v_m.txt", sdf_v_m_l)
        write_result("../results/sdf/splice_s_m.txt", sdf_s_m_l)
