import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
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

reload(tfq_translator)
reload(miscrun)


def load(args, ind):
    args = {"problem":"TFIM", "params":"[1.,{}]".format(js[ind]),"nrun":0, "shots":0, "epochs":500, "n_qubits":4, "vans_its":200,"itraj":0, "noisy":False, "noise_strength":0.0, "acceptange_percentage": 0.01, "L_HEA":1}
    args = miscrun.FakeArgs(args)
    L_HEA = args.L_HEA
    problem = args.problem
    params = ast.literal_eval(args.params)
    shots = miscrun.convert_shorts(args.shots)
    epochs = args.epochs
    n_qubits = args.n_qubits
    learning_rate=1e-4
    acceptange_percentage = args.acceptange_percentage
    noise_strength = args.noise_strength
    int_2_bool = lambda x: True if x==1 else False
    noisy = int_2_bool(args.noisy)
    tf.random.set_seed(args.itraj)
    np.random.seed(args.itraj)

    translator = tfq_translator.TFQTranslator(n_qubits = n_qubits, initialize="x", noisy=args.noisy, noise_strength = noise_strength)#, device_name="forest.numpy_wavefunction")
    translator_killer = tfq_translator.TFQTranslator(n_qubits = translator.n_qubits, initialize="x", noisy=translator.noisy, noise_strength = args.noise_strength)
    minimizer = tfq_minimizer.Minimizer(translator, mode="VQE", hamiltonian = problem, params = params, lr=learning_rate, shots=shots, patience=30, max_time_training=600, verbose=0)
    simplifier = penny_simplifier.PennyLane_Simplifier(translator)
    killer = tfq_killer.GateKiller(translator, translator_killer, hamiltonian=problem, params=params, lr=learning_rate, shots=shots, accept_wall = 2/args.acceptange_percentage)
    inserter = idinserter.IdInserter(translator, noise_in_rotations=1e-1)
    args_evaluator = {"n_qubits":translator.n_qubits, "problem":problem,"params":params,"nrun":args.itraj}
    evaluator = tfq_evaluator.PennyLaneEvaluator(minimizer = minimizer, killer=killer, args=args_evaluator, lower_bound=translator.ground, stopping_criteria=1e-3, vans_its=args.vans_its, acceptange_percentage = acceptange_percentage)


    evaluator.load_dicts_and_displaying(evaluator.identifier)
    return evaluator
js = np.linspace(0.,2.,16)
data = []
for ind in range(len(js)):
    evaluator = load(args, ind)
    [database, cost, lowest_cost, lower_bound, acceptance_percentage, operation, history] = evaluator.evolution[evaluator.get_best_iteration()]
    data.append([database, cost, lowest_cost, lower_bound, acceptance_percentage, operation, history])
ll=[]
for ind in range(len(js)):
    evaluator = load(args, ind)
    ll.append(len(evaluator.raw_history.keys()))

costs = np.array([data[k][1] for k in range(len(data))])
lowest = np.array([data[k][3] for k in range(len(data))])


plt.subplot(121)
plt.plot(js, costs,'.')
plt.plot(js, lowest)
plt.subplot(122)
plt.plot(js, np.array(np.abs(costs-lowest))/np.abs(lowest),'.')


ll=[]
for ind in range(len(js)):
    evaluator = load(args, ind)
    ll.append(len(evaluator.raw_history.keys()))


ll











.[
 for value in variable]
ells=[]
for k in list(evaluator.raw_history.keys()):
    info = evaluator.raw_history[k]
    if (info[-2] == "variational"):
        ells +=list(info[-1]["cost"])

plt.plot(ells)
plt.plot(np.ones(len(ells))*evaluator.lower_bound,'--')


dd = translator.draw(info[0])
penny_circuit, db = translator.penny_translator.give_circuit(info[0])
import pennylane as qml


aa = penny_circuit.tape.to_openqasm()
drawing = qml.draw(penny_circuit)(db, [])
print(drawing)
