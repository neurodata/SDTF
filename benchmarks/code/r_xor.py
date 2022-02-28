"""
Coauthors: Nick Hahn
           Haoyin Xu
"""
import argparse
import xor_functions as fn


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
parser.add_argument(
    "-synf", help="synergistic forests", required=False, action="store_true"
)
args, unknown = parser.parse_known_args()

# Perform experimentss
mc_rep = 50
n_test = 1000
classifiers = np.zeros(5)
if args.all or args.ht:
    classifiers[0] = 1
if args.all or args.mf:
    classifiers[1] = 1
if args.all or args.sdt:
    classifiers[2] = 1
if args.all or args.sdf:
    classifiers[3] = 1
if args.all or args.synf:
    classifiers[4] = 1


means = fn.run("R-XOR", classifiers, mc_rep, n_test)


# Write mean errors to appropriate txt files
if args.all or args.ht:
    write_result("../results/ht/rxor_exp_xor_error.txt", means[0])
    write_result("../results/ht/rxor_exp_r_xor_error.txt", means[1])
if args.all or args.mf:
    write_result("../results/mf/rxor_exp_xor_error.txt", means[2])
    write_result("../results/mf/rxor_exp_r_xor_error.txt", means[3])
if args.all or args.sdt:
    write_result("../results/sdt/rxor_exp_xor_error.txt", means[4])
    write_result("../results/sdt/rxor_exp_r_xor_error.txt", means[5])
if args.all or args.sdf:
    write_result("../results/sdf/rxor_exp_xor_error.txt", means[6])
    write_result("../results/sdf/rxor_exp_r_xor_error.txt", means[7])
if args.all or args.synf:
    write_result("../results/synf/rxor_exp_xor_error.txt", means[8])
    write_result("../results/synf/rxor_exp_r_xor_error.txt", means[9])
