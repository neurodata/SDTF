"""
Author: Haoyin Xu
"""
import time
import numpy as np
import torchvision.datasets as datasets
from numpy.random import permutation
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

    for i in range(500):
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


# prepare CIFAR data
# normalize
scale = np.mean(np.arange(0, 256))
normalize = lambda x: (x - scale) / scale

# train data
cifar_trainset = datasets.CIFAR10(root="./", train=True, download=True, transform=None)
X_train = normalize(cifar_trainset.data)
y_train = np.array(cifar_trainset.targets)

# test data
cifar_testset = datasets.CIFAR10(root="./", train=False, download=True, transform=None)
X_test = normalize(cifar_testset.data)
y_test = np.array(cifar_testset.targets)

X_train = X_train.reshape(-1, 32 * 32 * 3)
X_test = X_test.reshape(-1, 32 * 32 * 3)

# Perform experiments
dt_acc_l = []
dt_train_t_l = []
dt_test_t_l = []
for i in range(10):
    p = permutation(X_train.shape[0])

    X_r = X_train[p]
    y_r = y_train[p]

    dt_acc, dt_train_t, dt_test_t = experiment_dt()
    dt_acc_l.append(dt_acc)
    dt_train_t_l.append(dt_train_t)
    dt_test_t_l.append(dt_test_t)

    write_result("../dt/cifar10_acc.txt", dt_acc_l)
    write_result("../dt/cifar10_train_t.txt", dt_train_t_l)
    write_result("../dt/cifar10_test_t.txt", dt_test_t_l)
