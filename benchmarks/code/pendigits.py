"""
Author: Haoyin Xu
"""
import time
import psutil
import argparse
import pandas as pd
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
    n_node_l = []
    size_l = []

    dt = DecisionTreeClassifier()
    p = psutil.Process()

    for i in range(74):
        X_t = X_r[: (i + 1) * 100]
        y_t = y_r[: (i + 1) * 100]

        # Train the model
        start_time = time.perf_counter()
        dt.fit(X_t, y_t)
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Check size
        size = clf_size(dt, "../results/dt/temp.pickle")
        size_l.append(size)

        # Check memory
        v_m = (
            p.memory_full_info().rss / 1024 / 1024 / 1024,
            p.memory_full_info().vms / 1024 / 1024 / 1024,
        )
        v_m_l.append(v_m)

        # Check node counts
        n_node = node_count(dt, forest=False)
        n_node_l.append(n_node)

        # Test the model
        start_time = time.perf_counter()
        dt_l.append(prediction(dt, X_test, y_test))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    return dt_l, train_time_l, test_time_l, v_m_l, n_node_l, size_l


def experiment_rf():
    """Runs experiments for Random Forest"""
    rf_l = []
    train_time_l = []
    test_time_l = []
    v_m_l = []
    n_node_l = []
    size_l = []

    rf = RandomForestClassifier(n_estimators=10)
    p = psutil.Process()

    for i in range(74):
        X_t = X_r[: (i + 1) * 100]
        y_t = y_r[: (i + 1) * 100]

        # Train the model
        start_time = time.perf_counter()
        rf.fit(X_t, y_t)
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Check size
        size = clf_size(rf, "../results/rf/temp.pickle")
        size_l.append(size)

        # Check memory
        v_m = (
            p.memory_full_info().rss / 1024 / 1024 / 1024,
            p.memory_full_info().vms / 1024 / 1024 / 1024,
        )
        v_m_l.append(v_m)

        # Check node counts
        n_node = node_count(rf, forest=True)
        n_node_l.append(n_node)

        # Test the model
        start_time = time.perf_counter()
        rf_l.append(prediction(rf, X_test, y_test))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    return rf_l, train_time_l, test_time_l, v_m_l, n_node_l, size_l


def experiment_ht():
    """Runs experiments for Hoeffding Tree"""
    ht_l = []
    train_time_l = []
    test_time_l = []
    v_m_l = []
    n_node_l = []
    size_l = []

    ht = tree.HoeffdingTreeClassifier(max_size=1000, grace_period=2)
    p = psutil.Process()

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
            # Check size
            size = clf_size(ht, "../results/ht/temp.pickle")
            size_l.append(size)

            # Check memory
            v_m = (
                p.memory_full_info().rss / 1024 / 1024 / 1024,
                p.memory_full_info().vms / 1024 / 1024 / 1024,
            )
            v_m_l.append(v_m)

            # Check node counts
            n_node = ht.n_nodes
            n_node_l.append(n_node)

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

    return ht_l, train_time_l, test_time_l, v_m_l, n_node_l, size_l


def experiment_mf():
    """Runs experiments for Mondrian Forest"""
    mf_l = []
    train_time_l = []
    test_time_l = []
    v_m_l = []
    n_node_l = []
    size_l = []

    mf = MondrianForestClassifier(n_estimators=10)
    p = psutil.Process()

    for i in range(74):
        X_t = X_r[i * 100 : (i + 1) * 100]
        y_t = y_r[i * 100 : (i + 1) * 100]

        # Train the model
        start_time = time.perf_counter()
        mf.partial_fit(X_t, y_t)
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Check size
        size = clf_size(mf, "../results/mf/temp.pickle")
        size_l.append(size)

        # Check memory
        v_m = (
            p.memory_full_info().rss / 1024 / 1024 / 1024,
            p.memory_full_info().vms / 1024 / 1024 / 1024,
        )
        v_m_l.append(v_m)

        # Check node counts
        n_node = node_count(mf, forest=True)
        n_node_l.append(n_node)

        # Test the model
        start_time = time.perf_counter()
        mf_l.append(prediction(mf, X_test, y_test))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    # Reformat the train times
    for i in range(1, 74):
        train_time_l[i] += train_time_l[i - 1]

    return mf_l, train_time_l, test_time_l, v_m_l, n_node_l, size_l


def experiment_sdt():
    """Runs experiments for Stream Decision Tree"""
    sdt_l = []
    train_time_l = []
    test_time_l = []
    v_m_l = []
    n_node_l = []
    size_l = []

    sdt = DecisionTreeClassifier()
    p = psutil.Process()

    for i in range(74):
        X_t = X_r[i * 100 : (i + 1) * 100]
        y_t = y_r[i * 100 : (i + 1) * 100]

        # Train the model
        start_time = time.perf_counter()
        sdt.partial_fit(X_t, y_t, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Check size
        size = clf_size(sdt, "../results/sdt/temp.pickle")
        size_l.append(size)

        # Check memory
        v_m = (
            p.memory_full_info().rss / 1024 / 1024 / 1024,
            p.memory_full_info().vms / 1024 / 1024 / 1024,
        )
        v_m_l.append(v_m)

        # Check node counts
        n_node = node_count(sdt, forest=False)
        n_node_l.append(n_node)

        # Test the model
        start_time = time.perf_counter()
        sdt_l.append(prediction(sdt, X_test, y_test))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    # Reformat the train times
    for i in range(1, 74):
        train_time_l[i] += train_time_l[i - 1]

    return sdt_l, train_time_l, test_time_l, v_m_l, n_node_l, size_l


def experiment_sdf():
    """Runs experiments for Stream Decision Forest"""
    sdf_l = []
    train_time_l = []
    test_time_l = []
    v_m_l = []
    n_node_l = []
    size_l = []

    sdf = StreamDecisionForest(n_estimators=10)
    p = psutil.Process()

    for i in range(74):
        X_t = X_r[i * 100 : (i + 1) * 100]
        y_t = y_r[i * 100 : (i + 1) * 100]

        # Train the model
        start_time = time.perf_counter()
        sdf.partial_fit(X_t, y_t, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Check size
        size = clf_size(sdf, "../results/sdf/temp.pickle")
        size_l.append(size)

        # Check memory
        v_m = (
            p.memory_full_info().rss / 1024 / 1024 / 1024,
            p.memory_full_info().vms / 1024 / 1024 / 1024,
        )
        v_m_l.append(v_m)

        # Check node counts
        n_node = node_count(sdf, forest=True)
        n_node_l.append(n_node)

        # Test the model
        start_time = time.perf_counter()
        sdf_l.append(prediction(sdf, X_test, y_test))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    # Reformat the train times
    for i in range(1, 74):
        train_time_l[i] += train_time_l[i - 1]

    return sdf_l, train_time_l, test_time_l, v_m_l, n_node_l, size_l


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
args = parser.parse_args()

# Perform experiments
if args.all or args.dt:
    dt_acc_l = []
    dt_train_t_l = []
    dt_test_t_l = []
    dt_v_m_l = []
    dt_n_node_l = []
    dt_size_l = []
    for i in range(1):
        p = pendigits.sample(frac=1)

        X_r = p.iloc[:, :-1]
        y_r = p.iloc[:, -1]

        dt_acc, dt_train_t, dt_test_t, dt_v_m, dt_n_node, dt_size = experiment_dt()
        dt_acc_l.append(dt_acc)
        dt_train_t_l.append(dt_train_t)
        dt_test_t_l.append(dt_test_t)
        dt_v_m_l.append(dt_v_m)
        dt_n_node_l.append(dt_n_node)
        dt_size_l.append(dt_size)

        write_result("../results/dt/pendigits_acc", dt_acc_l)
        write_result("../results/dt/pendigits_train_t", dt_train_t_l)
        write_result("../results/dt/pendigits_test_t", dt_test_t_l)
        write_result("../results/dt/pendigits_v_m", dt_v_m_l, True)
        write_result("../results/dt/pendigits_n_node", dt_n_node_l)
        write_result("../results/dt/pendigits_size", dt_size_l, True)

if args.all or args.rf:
    rf_acc_l = []
    rf_train_t_l = []
    rf_test_t_l = []
    rf_v_m_l = []
    rf_n_node_l = []
    rf_size_l = []
    for i in range(1):
        p = pendigits.sample(frac=1)

        X_r = p.iloc[:, :-1]
        y_r = p.iloc[:, -1]

        rf_acc, rf_train_t, rf_test_t, rf_v_m, rf_n_node, rf_size = experiment_rf()
        rf_acc_l.append(rf_acc)
        rf_train_t_l.append(rf_train_t)
        rf_test_t_l.append(rf_test_t)
        rf_v_m_l.append(rf_v_m)
        rf_n_node_l.append(rf_n_node)
        rf_size_l.append(rf_size)

        write_result("../results/rf/pendigits_acc", rf_acc_l)
        write_result("../results/rf/pendigits_train_t", rf_train_t_l)
        write_result("../results/rf/pendigits_test_t", rf_test_t_l)
        write_result("../results/rf/pendigits_v_m", rf_v_m_l, True)
        write_result("../results/rf/pendigits_n_node", rf_n_node_l)
        write_result("../results/rf/pendigits_size", rf_size_l, True)

if args.all or args.ht:
    ht_acc_l = []
    ht_train_t_l = []
    ht_test_t_l = []
    ht_v_m_l = []
    ht_n_node_l = []
    ht_size_l = []
    for i in range(1):
        p = pendigits.sample(frac=1)

        X_r = p.iloc[:, :-1]
        y_r = p.iloc[:, -1]

        ht_acc, ht_train_t, ht_test_t, ht_v_m, ht_n_node, ht_size = experiment_ht()
        ht_acc_l.append(ht_acc)
        ht_train_t_l.append(ht_train_t)
        ht_test_t_l.append(ht_test_t)
        ht_v_m_l.append(ht_v_m)
        ht_n_node_l.append(ht_n_node)
        ht_size_l.append(ht_size)

        write_result("../results/ht/pendigits_acc", ht_acc_l)
        write_result("../results/ht/pendigits_train_t", ht_train_t_l)
        write_result("../results/ht/pendigits_test_t", ht_test_t_l)
        write_result("../results/ht/pendigits_v_m", ht_v_m_l, True)
        write_result("../results/ht/pendigits_n_node", ht_n_node_l)
        write_result("../results/ht/pendigits_size", ht_size_l, True)

if args.all or args.mf:
    mf_acc_l = []
    mf_train_t_l = []
    mf_test_t_l = []
    mf_v_m_l = []
    mf_n_node_l = []
    mf_size_l = []
    for i in range(1):
        p = pendigits.sample(frac=1)

        X_r = p.iloc[:, :-1]
        y_r = p.iloc[:, -1]

        mf_acc, mf_train_t, mf_test_t, mf_v_m, mf_n_node, mf_size = experiment_mf()
        mf_acc_l.append(mf_acc)
        mf_train_t_l.append(mf_train_t)
        mf_test_t_l.append(mf_test_t)
        mf_v_m_l.append(mf_v_m)
        mf_n_node_l.append(mf_n_node)
        mf_size_l.append(mf_size)

        write_result("../results/mf/pendigits_acc", mf_acc_l)
        write_result("../results/mf/pendigits_train_t", mf_train_t_l)
        write_result("../results/mf/pendigits_test_t", mf_test_t_l)
        write_result("../results/mf/pendigits_v_m", mf_v_m_l, True)
        write_result("../results/mf/pendigits_n_node", mf_n_node_l)
        write_result("../results/mf/pendigits_size", mf_size_l, True)

if args.all or args.sdt:
    sdt_acc_l = []
    sdt_train_t_l = []
    sdt_test_t_l = []
    sdt_v_m_l = []
    sdt_n_node_l = []
    sdt_size_l = []
    for i in range(1):
        p = pendigits.sample(frac=1)

        X_r = p.iloc[:, :-1]
        y_r = p.iloc[:, -1]

        (
            sdt_acc,
            sdt_train_t,
            sdt_test_t,
            sdt_v_m,
            sdt_n_node,
            sdt_size,
        ) = experiment_sdt()
        sdt_acc_l.append(sdt_acc)
        sdt_train_t_l.append(sdt_train_t)
        sdt_test_t_l.append(sdt_test_t)
        sdt_v_m_l.append(sdt_v_m)
        sdt_n_node_l.append(sdt_n_node)
        sdt_size_l.append(sdt_size)

        write_result("../results/sdt/pendigits_acc", sdt_acc_l)
        write_result("../results/sdt/pendigits_train_t", sdt_train_t_l)
        write_result("../results/sdt/pendigits_test_t", sdt_test_t_l)
        write_result("../results/sdt/pendigits_v_m", sdt_v_m_l, True)
        write_result("../results/sdt/pendigits_n_node", sdt_n_node_l)
        write_result("../results/sdt/pendigits_size", sdt_size_l, True)

if args.all or args.sdf:
    sdf_acc_l = []
    sdf_train_t_l = []
    sdf_test_t_l = []
    sdf_v_m_l = []
    sdf_n_node_l = []
    sdf_size_l = []
    for i in range(1):
        p = pendigits.sample(frac=1)

        X_r = p.iloc[:, :-1]
        y_r = p.iloc[:, -1]

        (
            sdf_acc,
            sdf_train_t,
            sdf_test_t,
            sdf_v_m,
            sdf_n_node,
            sdf_size,
        ) = experiment_sdf()
        sdf_acc_l.append(sdf_acc)
        sdf_train_t_l.append(sdf_train_t)
        sdf_test_t_l.append(sdf_test_t)
        sdf_v_m_l.append(sdf_v_m)
        sdf_n_node_l.append(sdf_n_node)
        sdf_size_l.append(sdf_size)

        write_result("../results/sdf/pendigits_acc", sdf_acc_l)
        write_result("../results/sdf/pendigits_train_t", sdf_train_t_l)
        write_result("../results/sdf/pendigits_test_t", sdf_test_t_l)
        write_result("../results/sdf/pendigits_v_m", sdf_v_m_l, True)
        write_result("../results/sdf/pendigits_n_node", sdf_n_node_l)
        write_result("../results/sdf/pendigits_size", sdf_size_l, True)
