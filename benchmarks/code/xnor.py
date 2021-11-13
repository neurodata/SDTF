"""
Author: Nick Hahn
"""
import argparse
import xor_functions as fn
import numpy as np


def write_result(filename, acc_ls):
    """Writes results to specified text file"""
    output = open(filename, "w")
    for acc in acc_ls:
        output.write(str(acc) + "\n")


# Parse classifier choices
parser = argparse.ArgumentParser()
parser.add_argument("-all", help="all classifiers", required=False, action="store_true")
parser.add_argument("-ht", help="hoeffding trees", required=False, action="store_true")
parser.add_argument("-mf", help="mondrian forests", required=False, action="store_true")
parser.add_argument(
    "-sdt", help="stream decision trees", required=False, action="store_true"
)
parser.add_argument(
    "-sdf", help="stream decision forests", required=False, action="store_true"
)
args, unknown = parser.parse_known_args()

# Perform experimentss
mc_rep = 50
n_test = 1000
classifiers = np.zeros(4)
if args.all or args.ht:
    classifiers[0] = 1
if args.all or args.mf:
    classifiers[1] = 1
if args.all or args.sdt:
    classifiers[2] = 1
if args.all or args.sdf:
    classifiers[3] = 1

means = fn.run("XNOR", classifiers, mc_rep, n_test)


# Write mean errors to appropriate txt files
write_result("../results/ht/xnor_exp_xor_error.txt", means[0])
write_result("../results/ht/xnor_exp_xnor_error.txt", means[1])
write_result("../results/mf/xnor_exp_xor_error.txt", means[2])
write_result("../results/mf/xnor_exp_xnor_error.txt", means[3])
write_result("../results/sdt/xnor_exp_xor_error.txt", means[4])
write_result("../results/sdt/xnor_exp_xnor_error.txt", means[5])
write_result("../results/sdf/xnor_exp_xor_error.txt", means[6])
write_result("../results/sdf/xnor_exp_xnor_error.txt", means[7])
