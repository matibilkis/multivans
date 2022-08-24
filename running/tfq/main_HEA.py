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
from utilities.evaluator.misc import get_def_path

# #
reload(tfq_minimizer)
reload(tfq_minimizer)
reload(tfq_translator)
reload(penny_simplifier)


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--problem", type=str, default="TFIM")
parser.add_argument("--n_qubits", type=int, default=4)
parser.add_argument("--params", type=str, default="[1., 1.1]")
parser.add_argument("--nrun", type=int, default=0)
parser.add_argument("--shots", type=int, default=0)
parser.add_argument("--epochs", type=int, default=5000)
parser.add_argument("--vans_its", type=int, default=100)
parser.add_argument("--itraj", type=int, default=1)
parser.add_argument("--noise_strength", type=float, default=.01)
parser.add_argument("--noisy", type=int, default=0)
parser.add_argument("--L_HEA", type=int, default=1)

args = parser.parse_args()

reload(miscrun)
start = datetime.now()

#args = {"problem":"TFIM", "params":"[1.,.1]","nrun":0, "shots":0, "epochs":500, "n_qubits":4, "vans_its":200,"itraj":1, "noisy":True, "noise_strength":0.0 }
# args = miscrun.FakeArgs(args)
problem = args.problem
L_HEA = args.L_HEA
params = ast.literal_eval(args.params)
shots = miscrun.convert_shorts(args.shots)
epochs = args.epochs
n_qubits = args.n_qubits
learning_rate=1e-4
noise_strength = args.noise_strength
int_2_bool = lambda x: True if x==1 else False
noisy = int_2_bool(args.noisy)
tf.random.set_seed(args.itraj)
np.random.seed(args.itraj)

translator = tfq_translator.TFQTranslator(n_qubits = n_qubits, initialize="x", noisy=args.noisy, noise_strength = args.noise_strength)#, device_name="forest.numpy_wavefunction")
minimizer = tfq_minimizer.Minimizer(translator, mode="VQE", hamiltonian = problem, params = params, lr=learning_rate, shots=shots, patience=30, max_time_training=int(10*3600), verbose=0, epochs=1)

circuit_db = database.concatenate_dbs([templates.hea_layer(translator)]*L_HEA)
circuit, circuit_db = translator.give_circuit(circuit_db)
minimized_db, [cost, resolver, history] = minimizer.variational(circuit_db)

path = get_def_path()+"QADC_{}/{}/".format(translator.noise_strength,L_HEA)
os.makedirs(path,exist_ok=True)

np.save(path+"history",np.stack(list(history.history.values())))
np.save(path+"cost",np.array([cost]))
np.save(path+"resolver",np.array(list(resolver.values())))
#
