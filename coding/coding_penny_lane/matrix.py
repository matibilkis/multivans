
import os
import sys
sys.path.insert(0, os.getcwd())
from importlib import reload
import tensorflow as tf
import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
from tqdm import tqdm
import utilities.translator.pennylane_translator as penny_translator
import utilities.evaluator.pennylane_evaluator as penny_evaluator
import utilities.variational.pennylane_model as penny_variational
import utilities.simplification.simplifier as penny_simplifier
import utilities.simplification.misc as simplification_misc#.kill_and_simplify
import utilities.simplification.gate_killer as penny_killer
import utilities.database.database as database
import utilities.database.templates as templates
import utilities.mutator.idinserter as idinserter
import running.misc as miscrun
from utilities.simplification.misc import get_qubits_involved, reindex_symbol, shift_symbols_down, shift_symbols_up, type_get, check_rot, order_symbol_labels, check_cnot, check_symbols_ordered
import coding.coding_simplifier as simplli






reload(penny_simplifier)
reload(templates)
reload(penny_translator)
reload(miscrun)
reload(idinserter)
reload(penny_variational)

n_qubits = 4

translator = penny_translator.PennyLaneTranslator(n_qubits = n_qubits, initialize="x")
simplifier = simplli.PennyLane_Simplifier(translator)
db = database.concatenate_dbs([templates.x_layer(translator, params=True),templates.z_layer(translator, params=True) ]*2)
c, cdb = translator.give_circuit(db)
simplified_db = cdb.copy()

import pennylane as qml

def my_quantum_function(params):
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0,1])
    qml.RY(y, wires=1)
    return [qml.PauliZ(wires=k) for k in range(2)]

ma = qml.matrix(my_quantum_function)
ma(1,2)



import pennylane as qml

qml.drawer.tape_mpl



params = np.random.randn(4)*2*np.pi
gates = [qml.RZ, qml.RX]*2
def circu(params, gates):
    for p,g in zip(params,gates):
        g(p,wires=0)
    return [qml.PauliZ(wires=k) for k in range(1)]

u_matrix = qml.matrix(circu)(params,gates)

def qfu():
    qml.QubitUnitary(u,wires=0)
    return qml.expval(qml.PauliZ(0))

dev = qml.device("default.qubit",wires=1)
qnode = qml.QNode(qfu, dev)

transformed_qfunc = qml.transforms.unitary_to_rot(qfu)
qq = qml.QNode(transformed_qfunc, dev)

qq.construct(qfu)()












params = np.random.randn(4)*2*np.pi
gates = [qml.RZ, qml.RX]*2
def circu(params, gates):
    for p,g in zip(params,gates):
        g(p,wires=0)
    return [qml.PauliZ(wires=k) for k in range(1)]

u_matrix = qml.matrix(circu)(params,gates)

def qfu():
    qml.QubitUnitary(u,wires=0)
    return qml.expval(qml.PauliZ(0))

dev = qml.device("default.qubit",wires=1)
qnode = qml.QNode(qfu, dev)

transformed_qfunc = qml.transforms.unitary_to_rot(qfu)
qq = qml.QNode(transformed_qfunc, dev)

qq.construct(qfu)()

qq.construct(qfu,qq.tape)












##
