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




translator = coding_template.PennyLaneTranslator(n_qubits=10)
circuit_db = templates.z_layer(translator)
global circuit_db_c
_, circuit_db_c = translator.give_circuit(circuit_db)

dev = qml.device("default.qubit", wires=translator.n_qubits)
@qml.qnode(dev)
def qnode(inputs, weights):
    cinputs = circuit_db_c.copy()
    symbols = database.get_trainable_symbols(translator,cinputs)
    ww = {s:w for s,w in zip(symbols, weights)}
    cinputs = database.update_circuit_db_param_values(translator, cinputs, ww)
    list_of_gate_ids = [templates.gate_template(**dict(cinputs.iloc[k])) for k in range(len(cinputs))]
    for gate_id in list_of_gate_ids:
        translator.spit_gate(gate_id)
    return [qml.expval(qml.PauliZ(k)) for k in range(translator.n_qubits)]

reload(coding_template)

qnode.train_params = len(weights)
model = coding_template.modelito(qnode)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1))

history = model.fit([1.], epochs=100)
type(history.history)

history.history.keys()

import matplotlib.pyplot as plt
plt.plot(history.history["cost"])
