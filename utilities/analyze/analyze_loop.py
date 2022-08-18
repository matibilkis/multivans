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


js = np.linspace(0,.2,4)



evs = {}
stre = [0.0, 0.001, 0.01, 0.1]
for ns in stre:

    args = {"problem":"TFIM", "params":"[1.,1.]","nrun":0, "shots":0, "epochs":500, "n_qubits":4, "vans_its":200,"itraj":0, "noisy":1, "noise_strength":ns}
    args = miscrun.FakeArgs(args)
    problem = args.problem
    params = ast.literal_eval(args.params)
    g,J = params
    shots = miscrun.convert_shorts(args.shots)
    epochs = args.epochs
    n_qubits = args.n_qubits
    learning_rate=1e-3
    noise_strength = args.noise_strength
    int_2_bool = lambda x: True if x==1 else False
    noisy = int_2_bool(args.noisy)
    tf.random.set_seed(args.itraj)
    np.random.seed(args.itraj)


    translator = tfq_translator.TFQTranslator(n_qubits = n_qubits, initialize="x", noisy=args.noisy, noise_strength = args.noise_strength)
    minimizer = tfq_minimizer.Minimizer(translator, mode="VQE", hamiltonian = problem, params = params, lr=learning_rate, shots=shots, patience=30, max_time_training=600, verbose=0)
    args_evaluator = {"n_qubits":translator.n_qubits, "problem":problem,"params":params,"nrun":args.nrun}
    evaluator = tfq_evaluator.PennyLaneEvaluator(minimizer = minimizer, args=args_evaluator, lower_bound=translator.ground, nrun=args.itraj, stopping_criteria=1e-3, vans_its=args.vans_its)

    evaluator.load_dicts_and_displaying(evaluator.identifier)
    evs[ns] = evaluator

[len(list(evs[n].evolution.keys())) for n in stre]


plt.scatter(stre,[evs[n].evolution[list(evs[n].evolution.keys())[-1]][2] for n in stre])








ells=[]
c=0
for k in list(evaluator.evolution.keys()):
    info = evaluator.evolution[k]
    ells +=list(info[-1]["cost"])

plt.plot(ells)


evaluator.raw_history.keys()

evaluator.raw_history[76]

cdb = evaluator.raw_history[76][0]
minimizer.build_and_give_cost(cdb)
tt = minimizer.variational(cdb)

minimizer.build_and_give_cost(tt[0])




plt.plot(ells)

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
