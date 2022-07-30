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
qnode, circuit_db = translator.give_circuit(circuit_db)
__, c = translator.give_circuit(circuit_db)

pm = coding_template.PennyModel(translator)

pm.fit(x=[1.], y=[1.], epochs=10)



from tqdm import tqdm
h={k:[] for k in pm.metrics_names}
for k in tqdm(range(100)):
    h+=pm.train_step([translator.db_train]*2)
g=pm.train_step([]*2)


h


model.trainable_variables






model = coding_template.modelito(qnode_keras)


symbols = database.get_trainable_symbols(translator,circuit_db)
ws = np.zeros(len(symbols))

### i need a way to call translator.db inside qnode
qnode(c, weigths)











model.qlayer([])

model([c, weigths])


















# dev = qml.device("default.qubit", wires=translator.n_qubits)
# @qml.qnode(dev)
# def qnode(inputs, weights,**kwargs):
#     for k in range(translator.n_qubits):
#         qml.RY(np.pi/2, k)
#     return [qml.expval(qml.PauliZ(k)) for k in range(translator.n_qubits)]
#


#
#
# symbols = database.get_trainable_symbols(translator,circuit_db)
# weigths = np.zeros(len(symbols))
#
#
# inputs = circuit_db
# cinputs = inputs.copy()
# symbols = database.get_trainable_symbols(translator,cinputs)
# ww = {s:w for s,w in zip( symbols, weigths)}
#
# cinputs = database.update_circuit_db_param_values(translator, cinputs, ww)
#
#
# list_of_gate_ids = [templates.gate_template(**dict(cinputs.iloc[k])) for k in range(len(cinputs))]
#
# translator.db = {}
# for i,gate_id in enumerate(list_of_gate_ids):
#     translator.db = translator.append_to_circuit(translator.db, gate_id)















type(history.history)

history.history.keys()

import matplotlib.pyplot as plt
plt.plot(history.history["cost"])
