import os
import numpy as np

ep = 0.01
etas = np.arange(0,1+ep, ep)

for eta1 in etas:
    for eta2 in etas:
        if eta2 < eta1:
            os.system("python3 main.py --eta1 {} --eta2 {}".format(eta1, eta2))
