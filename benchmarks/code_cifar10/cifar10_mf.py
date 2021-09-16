"""
Author: Haoyin Xu
"""
import time
import numpy as np
import torchvision.datasets as datasets
from numpy.random import permutation
from skgarden import MondrianForestClassifier


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


def experiment_mf():
    """Runs experiments for Mondrian Forest"""
    mf_l = []
    train_time_l = []
    test_time_l = []

    mf = MondrianForestClassifier(n_estimators=10)

    for i in range(500):
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
    for i in range(1, 500):
        train_time_l[i] += train_time_l[i - 1]

    return mf_l, train_time_l, test_time_l


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
mf_acc_l = []
mf_train_t_l = []
mf_test_t_l = []
for i in range(10):
    p = permutation(X_train.shape[0])

    X_r = X_train[p]
    y_r = y_train[p]

    mf_acc, mf_train_t, mf_test_t = experiment_mf()
    mf_acc_l.append(mf_acc)
    mf_train_t_l.append(mf_train_t)
    mf_test_t_l.append(mf_test_t)

    write_result("../mf/cifar10_acc.txt", mf_acc_l)
    write_result("../mf/cifar10_train_t.txt", mf_train_t_l)
    write_result("../mf/cifar10_test_t.txt", mf_test_t_l)
