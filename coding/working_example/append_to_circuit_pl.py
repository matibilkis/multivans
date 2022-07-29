import os
import sys
sys.path.insert(0, os.getcwd())
from importlib import reload
import tensorflow as tf
from pennylane import numpy as np
import pennylane as qml
import pandas as pd
import utilities.templates as templates
import coding.penny_template as coding_template


reload(coding_template)


translator = coding_template.PennyLaneTranslator(n_qubits=4)
circuit_db = templates.z_layer(translator)
dev, circuit_db_constructed =  translator.give_circuit(circuit_db)


momo = coding_template.modelito(dev)

momo(circuit_db[])

momo.trainable_variables[0].assign(tf.convert_to_tensor([0., 0.]))
tf.math.reduce_sum(momo(1))
