import os
import sys
sys.path.insert(0, os.getcwd())
import multiprocessing as mp
import numpy as np
import argparse
from utilities.evaluator.misc import *
# cores = mp.cpu_count()
### NOTE i don't use HEA input any more, since i minimize sequentially

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
args = parser.parse_args()
itraj = args.itraj

cores = 1
js = list(np.logspace(-6,-2,20))

def send_vans(ns):
    os.system("{} running/tfq/HEA/main_HEA.py --params '{}' --L_HEA 1 --noise_strength {} --noisy 1 --itraj {}".format(get_python(),str([1.0, 1.0]), ns, itraj))

with mp.Pool(cores) as p:
    p.map(send_vans, js)


#python3.8 running/tfq/tfq_main.py --params '{[1.0, 1.0]}' --n_qubits 4 --problem TFIM --itraj 0 --noisy 1 --noise_strength --vans_its 40
#python3.8 running/tfq/tfq_main.py --params "[1.0, 0.4]" --n_qubits 8 --problem TFIM --itraj 0
