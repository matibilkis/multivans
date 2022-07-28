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

translator = PennyLaneTranslator(n_qubits=4)
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

###

qml.CNOT([0,1])

translator.circuit_db = {}

def append_to_circuit(gate_id):

    ## the symbols are all elements we added but the very last one (added on the very previous line)
    symbols = []
    for j in [k["symbol"] for k in translator.circuit_db.values()][:-1]:
        if j != None:
            symbols.append(j)

    ind = gate_id["ind"]

    gate_index = len(list(translator.circuit_db.keys()))
    translator.circuit_db[gate_index] = gate_id #this is the new item to add

    if ind<translator.number_of_cnots:
        qml.CNOT(translator.indexed_cnots[str(ind)])
    else:
        cgate_type = (ind-translator.number_of_cnots)%3
        qubit = (ind-translator.number_of_cnots)//translator.n_qubits
        symbol_name = gate_id["symbol"]
        param_value = gate_id["param_value"]
        if symbol_name is None:
            symbol_name = "th_"+str(len(symbols))
            translator.circuit_db[gate_index]["symbol"] = symbol_name
        else:
            if symbol_name in symbols:
                print("warning, repeated symbol while constructing the circuit, see circuut_\n  symbol_name {}\n symbols {}\ncircuit_db {} \n\n\n".format(symbol_name, symbols, circuit_db))
        cgates[cgate_type](param_value,qubit)


dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev)
def qnode(inputs, weights):
    list_of_gate_ids = [gate_template(**dict(circuit_db.iloc[k])) for k in range(len(circuit_db))]
    for gate_id in list_of_gate_ids:
        filter_data(gate_id)
    return [qml.expval(qml.PauliZ(k)) for k in range(translator.n_qubits)]

qnode(circuit_db, 2)
####


def give_circuit(translator, dd,**kwargs):
    """
    retrieves circuit from circuit_db. It is assumed that the order in which the symbols are labeled corresponds to order in which their gates are applied in the circuit.
    If unresolved is False, the circuit is retrieved with the values of rotations (not by default, since we feed this to a TFQ model)
    """
    unresolved = kwargs.get("unresolved",True)
    list_of_gate_ids = [gate_template(**dict(dd.iloc[k])) for k in range(len(dd))]
    circuit, translator.circuit_db = [],{}

    dev = qml.device("default.qubit", wires=translator.n_qubits)
    @qml.qnode(dev)
    def qnode(inputs, weights):
        list_of_gate_ids = [gate_template(**dict(inputs.iloc[k])) for k in range(len(inputs))]
        for gate_id in list_of_gate_ids:
            filter_data(gate_id)
        return [qml.expval(qml.PauliZ(k)) for k in range(translator.n_qubits)]

    circuit = qnode(dd, 2)
    circuit_db = pd.DataFrame.from_dict(translator.circuit_db,orient="index")
    return dev, circuit_db



circuit, circuit_db = give_circuit(translator, circuit_db)

circuit.probability()


circuit.state

circuit.density_matrix([0,1])



translator.circuit_db





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
