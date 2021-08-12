"""
Author: Haoyin Xu
"""
import time
import numpy as np
import torchvision.datasets as datasets
from numpy.random import permutation
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

    ht = tree.HoeffdingTreeClassifier(max_size=1000)

    for i in range(X_train.shape[0]):
        X_t = X_r[i]
        y_t = y_r[i]

        idx = range(1024)
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
            print(p_t / X_test.shape[0], i)
            end_time = time.perf_counter()
            test_time_l.append(end_time - start_time)

    # Reformat the train times
    new_train_time_l = []
    for i in range(1, X_train.shape[0]):
        train_time_l[i] += train_time_l[i - 1]
        if i > 0 and (i + 1) % 100 == 0:
            new_train_time_l.append(train_time_l[i])
    train_time_l = new_train_time_l

    return ht_l, train_time_l, test_time_l


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
ht_acc_l = []
ht_train_t_l = []
ht_test_t_l = []
for i in range(100):
    print(i)
    p = permutation(X_train.shape[0])

    X_r = X_train[p]
    y_r = y_train[p]

    ht_acc, ht_train_t, ht_test_t = experiment_ht()
    ht_acc_l.append(ht_acc)
    ht_train_t_l.append(ht_train_t)
    ht_test_t_l.append(ht_test_t)

    write_result("../ht/cifar10_acc.txt", ht_acc_l)
    write_result("../ht/cifar10_train_t.txt", ht_train_t_l)
    write_result("../ht/cifar10_test_t.txt", ht_test_t_l)
