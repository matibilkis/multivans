### Inferring parameters of depolarizing channel through Petrucciones' https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-020-0272-6/MediaObjects/41534_2020_272_MOESM1_ESM.pdf
import numpy as np
import matplotlib.pyplot as plt

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



### Suspisious !!
def depo1(t1, epsilon=1e-3, tgx=36):
    t2 = .95*t1
    d1 = np.exp(-tgx/t1) + 2*np.exp(-tgx/t2)
    return 1+ 3*(2*epsilon -1)/d1

times = np.linspace(40,60,100)
plt.plot(times, depo1(times))

56.01/56.15






####    CUSTOM MODEL   ###

We want to have a single parameter governing the error model

let's say that depolarizing error is   P_1 = \lambda 1e-5    ----> we should sweep \lambda like 1e-2 -> 1
                                       P_2 = \lambda 1e-3   --> this should account for the fact that 2qubit gates are noisier

then we have the bit_flip errors, which can also be like

                                        P_flip = \lambda 1e-2

Then we have the thermal relaxation errors which we model as phase flips followed by amplitude damping channels

                                        p_z = \lambda 1e-3
                                        p_adc = \lambda 1e-3


###  Possible follow-ups:
if you implement Z you don't have thermal relaxation erros since Tg = 0.
