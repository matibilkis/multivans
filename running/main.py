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

reload(penny_translator)
reload(penny_variational)
reload(templates)
reload(database)

translator = penny_translator.PennyLaneTranslator(n_qubits = 4)

circuit_db = templates.z_layer(translator)
qnode, circuit_db = translator.give_circuit(circuit_db)


model = penny_variational.PennyModel(translator,lr=0.1, shots=100)
model.give_cost(translator.db_train)
circuit_db, [cost, resolver, history] = model.variational(epochs=10)
