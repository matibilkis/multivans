import os
import sys
sys.path.insert(0, os.getcwd())
import multiprocessing as mp
import numpy as np
# cores = mp.cpu_count()

cores = 4
# js = np.linspace(0.,2.,4)
js = [.4]
# js = list(range(1,8,1))
def send_vans(itraj):
    #print(J)
    #print("python3 running/main.py --params '{}' --n_qubits 8".format(str([1.0, J])))
    os.system("python3.8 running/tfq/tfq_main.py --params '{}' --n_qubits 8 --problem TFIM --itraj {}".format(str([1.0, itraj]), 1))

    # os.system("python3 running/main.py --params {} --n_qubits 8".format(str([1.0, J])))

with mp.Pool(cores) as p:
    p.map(send_vans, js)



#python3.8 running/tfq/tfq_main.py --params "[1.0, 0.4]" --n_qubits 8 --problem TFIM --itraj 0
