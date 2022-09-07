import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
from datetime import datetime
sys.path.insert(0, os.getcwd())

import tensorflow as tf
import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
from tqdm import tqdm
import utilities.translator.tfq_translator as tfq_translator
import utilities.evaluator.evaluator as tfq_evaluator
import utilities.variational.tfq.variational as tfq_minimizer
import utilities.simplification.simplifier as penny_simplifier
import utilities.simplification.misc as simplification_misc#.kill_and_simplify
import utilities.simplification.tfq.gate_killer as tfq_killer
import utilities.database.database as database
import utilities.database.templates as templates
import utilities.mutator.idinserter as idinserter
import running.misc.misc as miscrun
import argparse
import ast
from importlib import reload


# #
# reload(tfq_minimizer)
# reload(tfq_translator)
# reload(penny_simplifier)


# parser = argparse.ArgumentParser(add_help=False)
# parser.add_argument("--problem", type=str, default="XXZ")
# parser.add_argument("--n_qubits", type=int, default=4)
# parser.add_argument("--params", type=str, default="[1., 1.1]")
# parser.add_argument("--nrun", type=int, default=0)
# parser.add_argument("--shots", type=int, default=0)
# parser.add_argument("--epochs", type=int, default=5000)
# parser.add_argument("--vans_its", type=int, default=200)
# parser.add_argument("--itraj", type=int, default=1)
#
# args = parser.parse_args()


start = datetime.now()

args = {"problem":"TFIM", "params":"[1.,.1]","nrun":0, "shots":0, "epochs":500, "n_qubits":10, "vans_its":200,"itraj":1}
args = miscrun.FakeArgs(args)
problem = args.problem
params = ast.literal_eval(args.params)
g,J = params
shots = miscrun.convert_shorts(args.shots)
epochs = args.epochs
n_qubits = args.n_qubits
learning_rate=0.01
tf.random.set_seed(args.itraj)
np.random.seed(args.itraj)


#### tfq.layers.NoisyPQC
reload(tfq_minimizer)

translator = tfq_translator.TFQTranslator(n_qubits = 2, initialize="x")#, device_name="forest.numpy_wavefunction")
minimizer = tfq_minimizer.Minimizer(translator, mode="VQE", hamiltonian = problem, params = params, lr=learning_rate, shots=shots, g=g, J=J, patience=10, max_time_training=600, noisy=True, epochs=100)

circuit_db = translator.initialize(mode="u2")
circuit, circuit_db = translator.give_circuit(translator.db_train)

db = pd.DataFrame([templates.gate_template(k) for k in range(translator.number_of_cnots)])

circuit, db = translator.give_circuit(db)

gate = list(circuit.all_operations())[1]
gate.qubits
gate.qubits





for k in circuit.all_operations():
    print(k)


nois = circuit + cirq.Circuit(cirq.depolarize(.01).on_each(*circuit.all_qubits()))

#minimizer.variational(circuit_db, verbose=2)

noisy_circuit = []
for k in list(circuit.all_operations()):
    for q in k.qubits:
        noisy_circuit.append(cirq.depolarize(0.01).on(q))
    noisy_circuit.append(k)

cirq.Circuit(noisy_circuit)
k.qubits


list(circuit.all_operations())[6].qubits










#####
