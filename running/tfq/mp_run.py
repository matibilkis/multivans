import os
import sys
sys.path.insert(0, os.getcwd())
import multiprocessing as mp
import numpy as np
# cores = mp.cpu_count()

cores = 4
js = np.linspace(-1.4,2.,16)

def send_vans(J):
    #print(J)
    #print("python3 running/main.py --params '{}' --n_qubits 8".format(str([1.0, J])))
    os.system("python3.8 running/tfq/tfq_main.py --params '{}' --n_qubits 8".format(str([1.0, J])))

    # os.system("python3 running/main.py --params {} --n_qubits 8".format(str([1.0, J])))

with mp.Pool(cores) as p:
    p.map(send_vans, js)
