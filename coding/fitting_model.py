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
import matplotlib.pyplot as plt



reload(coding_template)
reload(database)
reload(templates)

translator = coding_template.PennyLaneTranslator(n_qubits=4)
circuit_db = templates.z_layer(translator)
qnode, circuit_db = translator.give_circuit(circuit_db)
__, c = translator.give_circuit(circuit_db)


pm = coding_template.PennyModel(translator, lr=0.1, shots=100)

pm.train_step([])


history = pm.fit(x=[1.], y=[1.], epochs=200)

translator.db_train = database.correct_param_value_dtype(translator,translator.db_train)

plt.plot(history.history["cost"])
