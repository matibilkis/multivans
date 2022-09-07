### Inferring parameters of depolarizing channel through Petrucciones' https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-020-0272-6/MediaObjects/41534_2020_272_MOESM1_ESM.pdf
import numpy as np

ep1 = 1e-2
ep2 = 1e-3

tgx = 36 ###1qubit gate, only x
t1 = 56.15
t2 =  56.01  ##t2<t1 \forall Q

d1 = np.exp(-tgx/t1) + 2*np.exp(-tgx/t2)
d1

tg2 = 300
tau = lambda x,y: np.exp(-x/y)

tau(tg2,t1) + tau(tg2, t1) +   tau(tg2, t1)**2 + 4*tau(tg2, t2)**2 + 4*tau(tg2, t2) + 4*tau(tg2, t1)*tau(tg2,t2)
