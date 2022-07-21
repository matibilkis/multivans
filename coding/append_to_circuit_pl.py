import os
import sys
sys.path.insert(0, os.getcwd())
from importlib import reload

import tensorflow as tf
from pennylane import numpy as np
import pennylane as qml



import utilities.templates as templates
reload(templates)

class PennyLaneTranslator:
    def __init__(self, n_qubits, **kwargs):
        self.n_qubits = n_qubits
        #self.qubits = cirq.GridQubit.rect(1, n_qubits)

        ### blocks that are fixed-structure (i.e. channels, state_preparation, etc.)
        untouchable_blocks = kwargs.get("untouchable_blocks",[])
        self.untouchable_blocks = untouchable_blocks
        self.discard_qubits = kwargs.get("discard_qubits",[]) ###these are the qubits that you don't measure, i.e. environment


        #### keep a register on which integers corresponds to which CNOTS, target or control.
        self.indexed_cnots = {}
        self.cnots_index = {}
        count = 0
        for control in range(self.n_qubits):
            for target in range(self.n_qubits):
                if control != target:
                    self.indexed_cnots[str(count)] = [control, target]
                    self.cnots_index[str([control,target])] = count
                    count += 1
        self.number_of_cnots = len(self.indexed_cnots)
        #self.cgates = {0:cirq.rz, 1: cirq.rx, 2:cirq.ry}

translator = PennyLaneTranslator(n_qubits=2)
circuit_db = templates.z_layer(translator)
circuit_db


n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev)
def qnode(inputs, weights):
    qml.RX(weights[0], wires=0)
    qml.RX(weights[1], wires=1)
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

circuit_db


cgates = {0:qml.RZ, 1: qml.RX, 2:qml.RY}

cgates[0](0.,wires=0)


gate_id = circuit_db.loc[0]
gate_id

def filter_data(gate_id):
    symbol_name = gate_id["symbol"]
    param_value = gate_id["param_value"]
    ind = gate_id["ind"]
    cgate_type = (ind-translator.number_of_cnots)%3
    qubit = (ind-translator.number_of_cnots)//translator.n_qubits
    cgates[cgate_type](param_value,qubit)

[filter_data(circuit_db.loc[k]) for k in range(len(circuit_db))]


###

dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev)
def qnode(inputs, weights):
    for k in range(len(inputs)):
        filter_data(inputs.loc[k])
    return [qml.expval(qml.PauliZ(k)) for k in range(translator.n_qubits)]

qnode(circuit_db, 2)

####



def cc(l):
    if ind>translator.number_of_cnots:
        cgates[(ind-translator.number_of_cnots)%3]()






class modelito(tf.keras.Model):
    def __init__(self, circuit):
        super(modelito,self).__init__()

        weight_shapes = {"weights": 2}
        self.qlayer = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=2)

    def call(self, inputs):
        return self.qlayer(inputs)

    def fit(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            energy = tf.math.reduce_sum(self(x))
        grads=tape.gradient(energy, self.trainable_variables)
        self.optimizer.apply_gradients(grads, self.trainable_variables)
        return
momo = modelito(qnode)


momo(1)

momo.trainable_variables[0].assign(tf.convert_to_tensor([0., 0.]))
momo.trainable_variables
##

tf.math.reduce_sum(momo(1))
