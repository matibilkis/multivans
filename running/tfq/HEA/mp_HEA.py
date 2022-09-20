import os
import sys
sys.path.insert(0, os.getcwd())
import multiprocessing as mp
import numpy as np
import argparse
from utilities.evaluator.misc import *
# cores = mp.cpu_count()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
args = parser.parse_args()
itraj = args.itraj

ncores = 16
js = list(np.logspace(-5,-3,ncores))

def send_vans(ns):
    os.system("{} running/tfq/HEA/main_HEA.py --params '{}' --n_qubits 8 --lr 0.001 --L_HEA 1 --noise_strength {} --noisy 1 --itraj {} --run_name HEA_fixed".format(get_python(),str([1.0, 1.0]), ns, itraj))

with mp.Pool(ncores) as p:
    p.map(send_vans, js)
