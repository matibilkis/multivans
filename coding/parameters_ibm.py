from qiskit import QuantumCircuit, execute
from qiskit import IBMQ, Aer
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel
from qiskit import QuantumCircuit, execute, Aer

noise = NoiseModel()


.device.thermal_relaxation_values

# Build noise model from backend properties
backend = Aer.get_backend('qasm_simulator')
noise_model = NoiseModel.from_backend(backend)






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










coupling_map = backend.configuration().coupling_map
coupling_map

# Build noise model from backend properties


backend = provider.get_backend('ibmq_vigo')
backend.available_devices
dir(backend)

aa = backend.properties()
bb = backend.configuration()
bb.description

bb.gates

backend

aabackends
noise_model = NoiseModel.from_backend(backend)
