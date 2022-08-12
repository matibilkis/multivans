import os
import sys
sys.path.insert(0, os.getcwd())
import multiprocessing as mp
import numpy as np
# cores = mp.cpu_count()

cores = 4
js = np.linspace(-1.4,2.,16)

def send_vans(J):
    os.system("python3 running/main.py --params {} --n_qubits 8".format([1.0, J]))

with mp.Pool(cores) as p:
    p.map(send_vans, js)
