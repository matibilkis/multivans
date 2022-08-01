import os

## THIS WORKS... but see also
#https://pennylane.ai/blog/2021/05/how-to-visualize-quantum-circuits-in-pennylane/



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
db = translator.initialize()
qnode, db = translator.give_circuit(db)
print(qnode.device.)

print(qml.draw(qnode, show_all_wires=True)(db,[]))

model = penny_variational.PennyModel(translator,lr=0.1, shots=100)
model.give_cost(translator.db_train)
circuit_db, [cost, resolver, history] = model.variational(epochs=10)
model.translator.db_train


cost
translator.ground










#
