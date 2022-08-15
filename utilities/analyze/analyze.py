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
# #
# # parser = argparse.ArgumentParser(add_help=False)
# # parser.add_argument("--problem", type=str, default="XXZ")
# # parser.add_argument("--n_qubits", type=int, default=8)
# # parser.add_argument("--params", type=str, default="[1., 1.1]")
# # parser.add_argument("--nrun", type=int, default=0)
# # parser.add_argument("--shots", type=int, default=0)
# # parser.add_argument("--epochs", type=int, default=500)
# # parser.add_argument("--vans_its", type=int, default=200)
# #
# # args = parser.parse_args()


start = datetime.now()

args = {"problem":"XXZ", "params":"[1.,0.2]","nrun":0, "shots":0, "epochs":500, "n_qubits":8, "vans_its":200}
args = miscrun.FakeArgs(args)
problem = args.problem
params = ast.literal_eval(args.params)
g,J = params
shots = miscrun.convert_shorts(args.shots)
epochs = args.epochs
n_qubits = args.n_qubits
learning_rate=0.1

translator = tfq_translator.TFQTranslator(n_qubits = n_qubits, initialize="x")#, device_name="forest.numpy_wavefunction")
translator_killer = tfq_translator.TFQTranslator(n_qubits = translator.n_qubits, initialize="x")#, device_name=translator.device_name)
minimizer = tfq_minimizer.Minimizer(translator, mode="VQE", hamiltonian = problem, params = params, lr=learning_rate, shots=shots, g=g, J=J, patience=10, max_time_training=600)


simplifier = penny_simplifier.PennyLane_Simplifier(translator)
killer = tfq_killer.GateKiller(translator, translator_killer, hamiltonian=problem, params=params, lr=learning_rate, shots=shots, g=g, J=J)
inserter = idinserter.IdInserter(translator)
args_evaluator = {"n_qubits":translator.n_qubits, "problem":problem,"params":params,"nrun":args.nrun}
evaluator = tfq_evaluator.PennyLaneEvaluator(args=args_evaluator, lower_bound=translator.ground, nrun=args.nrun, stopping_criteria=1e-3, vans_its=args.vans_its)


evaluator.load_dicts_and_displaying(evaluator.identifier)


evaluator.raw_history.keys()


print(list(evaluator.raw_history[4][0]["symbol"]))
circuit_db  = evaluator.raw_history[2][0]
circuit_db

translator.give_circuit(evaluator.raw_history[2][0], unresolved=False)[0]

circuit, circuit_db = translator.give_circuit(evaluator.raw_history[1][0])



reload(penny_simplifier)
simplifier = penny_simplifier.PennyLane_Simplifier(translator)



tf.random.seed(1)

sdb, ns = simplifier.reduce_circuit(circuit_db)

minimizer.variational(sdb)



translator.give_circuit(sdb)[0]






evaluator.raw_history[9][-2]






.J=list(evaluator.evolution.values())
evo = [J[k][-1] for k in range(len(J))]
