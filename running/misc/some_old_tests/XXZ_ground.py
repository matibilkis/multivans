import os
import sys
sys.path.insert(0, os.getcwd())
from importlib import reload

import utilities.translator.pennylane_translator as penny_translator
import utilities.variational.pennylane_model as penny_variational
import utilities.database.database as database
import utilities.database.templates as templates

import tensorflow as tf
import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
from tqdm import tqdm


reload(penny_translator)
reload(penny_variational)
reload(templates)
reload(database)

translator = penny_translator.PennyLaneTranslator(n_qubits = 4)
circuit_db = templates.z_layer(translator)
qnode, circuit_db = translator.give_circuit(circuit_db)

js = np.linspace(-3,2,40)
grounds = []
translator = penny_translator.PennyLaneTranslator(n_qubits = 4)
for j in tqdm(js):
    model = penny_variational.PennyModel(translator,lr=0.1, shots=None, J=j)
    grounds.append(model.ground)
plt.plot(js,grounds)


js = np.linspace(-3,2,40)
grounds = []
translator = penny_translator.PennyLaneTranslator(n_qubits = 4)
for j in tqdm(js):
    model = penny_variational.PennyModel(translator,lr=0.1, shots=None, J=j)
    grounds.append(model.translator.ground)
plt.plot(js,grounds)




model.give_cost(translator.db_train)
circuit_db, [cost, resolver, history] = model.variational(epochs=10)
model.get_observable()

{}.get("hola",2)

from datetime import datetime
a = datetime.now()
delta = datetime.now()- a

delta.seconds
print("a {}".format(delta)

H=qml.Hamiltonian(list(model.h_coeffs),list(model.ops))
hh = qml.utils.sparse_hamiltonian(H).toarray()
ground = np.min(np.linalg.eigvals(hh))






model.observable()


g = J = 1.
obs = [.1*qml.PauliZ(k%translator.n_qubits)@qml.PauliZ((k+1)%translator.n_qubits) for k in range(translator.n_qubits)]
coeffs = [g for k in range(len(obs))]


H = model.qmlobs
grouped_obs = [[H.ops[i] for i in indices] for indices in H.grouping_indices]


obs = [qml.PauliZ(k%translator.n_qubits)@qml.PauliZ((k+1)%translator.n_qubits) for k in range(translator.n_qubits)]
coeffs = [np.random.random() for k in range(len(obs))]

ham = qml.Hamiltonian(coeffs, obs)























##
