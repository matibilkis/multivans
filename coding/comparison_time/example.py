import pennylane as qml
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())
from importlib import reload


n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights_0, weight_1):
    qml.RX(inputs[0], wires=0)
    qml.RX(inputs[1], wires=1)
    qml.Rot(*weights_0, wires=0)
    qml.RY(weight_1, wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

weight_shapes = {"weights_0": 3, "weight_1": 1}
qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=2)

qlayer([.1,.1])

### esto indica que lo que tengo que hacer es pasar los *weights
#como trainable variables, osea, como input del modelo (será muy lento ?!)








n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)
gates = [qml.RZ, qml.RY, qml.RX]*10

@qml.qnode(dev)
def qnode(inputs, trainable_variables):
    for ind,k in enumerate(gates[:5]):
        k(inputs[ind], ind%3)
    for ind,k in enumerate(gates[5:]):
        k(trainable_variables[ind],ind%3)
    return [qml.expval(qml.PauliZ(k)) for k in range(n_qubits)]#, qml.expval(qml.PauliZ(1))

weight_shapes = {"trainable_variables": len(gates[5:])}#, "weight_1": 1}
qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

weight_shapes

qlayer(np.random.randn(10))


### osea que los inputs son los parámetros, los weights son los pesos. ¿qué pasa si pongo como inputs todo el circuito, y si hay una condición (trainable),
#entonces el weight es weights[k]?



import utilities.templates as templates
import coding.penny_template as coding_template
reload(coding_template)



translator = coding_template.PennyLaneTranslator(n_qubits)
translator = coding_template.PennyLaneTranslator(n_qubits=4)
circuit_db = templates.z_layer(translator)

_, circuit_db_c = translator.give_circuit(circuit_db)


def spit_gate(translator,gate_id):
    ## the symbols are all elements we added but the very last one (added on the very previous line)
    ind = gate_id["ind"]

    if ind<translator.number_of_cnots:
        qml.CNOT(translator.indexed_cnots[str(ind)])
    else:
        cgate_type = (ind-translator.number_of_cnots)%3
        qubit = (ind-translator.number_of_cnots)//translator.n_qubits
        symbol_name = gate_id["symbol"]
        param_value = gate_id["param_value"]
        translator.cgates[cgate_type](param_value,qubit)


global circuit_db_c

dev = qml.device("default.qubit", wires=translator.n_qubits)
@qml.qnode(dev)
def qnode(inputs, weights):
    cinputs = circuit_db_c.copy()
    symbols = database.get_trainable_symbols(translator,cinputs)
    ww = {s:w for s,w in zip(symbols, weights)}
    cinputs = database.update_circuit_db_param_values(translator, cinputs, ww)
    list_of_gate_ids = [templates.gate_template(**dict(cinputs.iloc[k])) for k in range(len(cinputs))]
    for gate_id in list_of_gate_ids:
        spit_gate(translator, gate_id)
    return [qml.expval(qml.PauliZ(k)) for k in range(translator.n_qubits)]

symbols = database.get_trainable_symbols(translator,circuit_db_c)
weights = np.ones(len(symbols))


qnode(1, weights)


weight_shapes = {"weights": len(weights)}
qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

qlayer(1)

qlayer.weights
qnode.train_params = len(weights)

reload(coding_template)
model = coding_template.modelito(qnode)
model([1])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01))


data  = tf.random.uniform((2,1))
x,y = data
reload(coding_template)

data[0]
history = model.fit(data[0], epochs=100)
type(history.history)

history.history.keys()

import matplotlib.pyplot as plt
plt.plot(history.history["grad_norm"])


with tf.GradientTape() as tape:
    tape.watch(model.trainable_variables)
    preds = model([1])
#
tape.gradient(preds, model.trainable_variables)
