import os
import sys
sys.path.insert(0, os.getcwd())
import multiprocessing as mp
import numpy as np
import argparse
from utilities.evaluator.misc import *

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
args = parser.parse_args()
itraj = args.itraj

cores = 8
js = list(np.logspace(-5,-4,8))

def send_vans(ns):
    os.system("{} running/tfq/noisy/tfq_main.py --params '{}' --n_qubits 8 --problem TFIM --itraj {} --noisy 1 --noise_strength {} ".format(get_python(), str([1.0, 1.0]), itraj, ns))

with mp.Pool(cores) as p:
    p.map(send_vans, js)
