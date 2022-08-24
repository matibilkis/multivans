import os
import sys
sys.path.insert(0, os.getcwd())
import multiprocessing as mp
import numpy as np
import argparse

# cores = mp.cpu_count()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--L_HEA", type=int, default=1)
args = parser.parse_args()
L_HEA = args.L_HEA

cores = 10
js = np.logspace(-7,-2,10)

def send_vans(ns):
    #print("python3 running/main.py --params '{}' --n_qubits 8".format(str([1.0, J])))
    # os.system("python3.8 running/tfq/tfq_main.py --params '{}' --n_qubits 8 --problem TFIM --itraj {}".format(str([1.0, itraj]), 1))
    os.system("python3.8 running/tfq/main_HEA.py --params '{}' --n_qubits 4 --problem TFIM --itraj 0 --noisy 1 --noise_strength {} --vans_its 40 --L_HEA {}".format(str([1.0, 1.0]), ns, L_HEA))

    # os.system("python3 running/main.py --params {} --n_qubits 8".format(str([1.0, J])))

with mp.Pool(cores) as p:
    p.map(send_vans, js)


#python3.8 running/tfq/tfq_main.py --params '{[1.0, 1.0]}' --n_qubits 4 --problem TFIM --itraj 0 --noisy 1 --noise_strength --vans_its 40
#python3.8 running/tfq/tfq_main.py --params "[1.0, 0.4]" --n_qubits 8 --problem TFIM --itraj 0
