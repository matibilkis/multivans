import pennylane as qml
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())
from importlib import reload
import tensorflow as tf
import utilities.database as database
import utilities.templates as templates
import coding.penny_template as coding_template
reload(coding_template)
import matplotlib.pyplot as plt




translator = coding_template.PennyLaneTranslator(n_qubits=10)
circuit_db = templates.z_layer(translator)
global circuit_db_c
_, circuit_db_c = translator.give_circuit(circuit_db)

n_qubits = 10
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev)
def qnode(inputs, weights):
    for ind,k in enumerate(range(100)):
        qml.RY(weights[ind], k%n_qubits)
    return [qml.expval(qml.PauliZ(k)) for k in range(translator.n_qubits)]

reload(coding_template)

qnode.train_params = 100
model = coding_template.modelito(qnode)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1))

history = model.fit([1.], epochs=100)
type(history.history)

history.history.keys()

plt.plot(history.history["cost"])
