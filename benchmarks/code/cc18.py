"""
Author: Haoyin Xu
"""
import time
import argparse
import numpy as np
import json
import openml
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sdtf import StreamDecisionForest


def prediction(classifier, X_test, y_test):
    """Generates predictions from model"""
    y_preds = classifier.predict(X_test)

    return accuracy_score(y_preds, y_test)


def experiment_rf(X_train, X_test, y_train, y_test):
    """Runs experiments for Random Forest"""
    rf_l = []
    train_time_l = []
    test_time_l = []

    rf = RandomForestClassifier(n_jobs=-1)
    batch_counts = len(y_train) / BATCH_SIZE
    for i in range(int(batch_counts)):
        X_t = X_train[: (i + 1) * BATCH_SIZE]
        y_t = y_train[: (i + 1) * BATCH_SIZE]

        # Train the model
        start_time = time.perf_counter()
        rf.fit(X_t, y_t)
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Test the model
        start_time = time.perf_counter()
        rf_l.append(prediction(rf, X_test, y_test))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    return rf_l, train_time_l, test_time_l


def experiment_sdf(X_train, X_test, y_train, y_test):
    """Runs experiments for Stream Decision Forest"""
    sdf_l = []
    train_time_l = []
    test_time_l = []

    sdf = StreamDecisionForest(n_jobs=-1)
    batch_counts = len(y_train) / BATCH_SIZE
    classes = np.unique(y_train)
    for i in range(int(batch_counts)):
        X_t = X_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        y_t = y_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

        # Train the model
        start_time = time.perf_counter()
        sdf.partial_fit(X_t, y_t, classes=classes)
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Test the model
        start_time = time.perf_counter()
        sdf_l.append(prediction(sdf, X_test, y_test))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    # Reformat the train times
    for i in range(1, int(batch_counts)):
        train_time_l[i] += train_time_l[i - 1]

    return sdf_l, train_time_l, test_time_l


# Parse classifier choices
parser = argparse.ArgumentParser()
parser.add_argument("-all", help="all classifiers", required=False, action="store_true")
parser.add_argument("-rf", help="random forests", required=False, action="store_true")
parser.add_argument(
    "-sdf", help="stream decision forests", required=False, action="store_true"
)
args = parser.parse_args()

BATCH_SIZE = 100

rf_acc_dict = {}
rf_train_t_dict = {}
rf_test_t_dict = {}

sdf_acc_dict = {}
sdf_train_t_dict = {}
sdf_test_t_dict = {}

# Prepare cc18 data
for data_id in openml.study.get_suite("OpenML-CC18").data:
    # Retrieve dataset
    dataset = openml.datasets.get_dataset(data_id)
    X, y, is_categorical, _ = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    X = np.nan_to_num(X)

    rf_acc_dict[data_id] = []
    rf_train_t_dict[data_id] = []
    rf_test_t_dict[data_id] = []

    sdf_acc_dict[data_id] = []
    sdf_train_t_dict[data_id] = []
    sdf_test_t_dict[data_id] = []

    # Split the datasets into 5-fold CV
    skf = StratifiedKFold(shuffle=True)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test, y_train, y_test = (
            X[train_index],
            X[test_index],
            y[train_index],
            y[test_index],
        )

        if args.all or args.rf:
            rf_acc, rf_train_t, rf_test_t = experiment_rf(
                X_train, X_test, y_train, y_test
            )
            rf_acc_dict[data_id].append(rf_acc)
            rf_train_t_dict[data_id].append(rf_train_t)
            rf_test_t_dict[data_id].append(rf_test_t)

            f = open("../results/rf/cc18_acc.json", "w")
            json.dump(rf_acc_dict, f)
            f.close()

            f = open("../results/rf/cc18_train_t.json", "w")
            json.dump(rf_train_t_dict, f)
            f.close()

            f = open("../results/rf/cc18_test_t.json", "w")
            json.dump(rf_test_t_dict, f)
            f.close()

        if args.all or args.sdf:
            sdf_acc, sdf_train_t, sdf_test_t = experiment_sdf(
                X_train, X_test, y_train, y_test
            )
            sdf_acc_dict[data_id].append(sdf_acc)
            sdf_train_t_dict[data_id].append(sdf_train_t)
            sdf_test_t_dict[data_id].append(sdf_test_t)

            f = open("../results/sdf/cc18_acc.json", "w")
            json.dump(sdf_acc_dict, f)
            f.close()

            f = open("../results/sdf/cc18_train_t.json", "w")
            json.dump(sdf_train_t_dict, f)
            f.close()

            f = open("../results/sdf/cc18_test_t.json", "w")
            json.dump(sdf_test_t_dict, f)
            f.close()
