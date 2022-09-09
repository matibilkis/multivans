import os
import sys
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
from datetime import datetime
sys.path.insert(0, os.getcwd())

import tensorflow as tf
import matplotlib.pyplot as plt
import pennylane as qml

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


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))




def load_ev_HEA(ns, itraj=1):

    args = {"problem":"TFIM", "params":"[1.,1.]","nrun":0, "shots":0, "epochs":500, "n_qubits":4, "vans_its":200,"itraj":itraj, "noisy":True,
            "noise_strength":ns, "acceptange_percentage": 0.01, "L_HEA":1,"run_name":"HEA_fixed","noise_model":"aer"}
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
    tf.random.set_seed(abs(args.itraj))
    np.random.seed(abs(args.itraj))

    translator = tfq_translator.TFQTranslator(n_qubits = n_qubits, initialize="x", noisy=args.noisy, noise_strength = noise_strength, noise_model="aer")#, device_name="forest.numpy_wavefunction")
    translator_killer = tfq_translator.TFQTranslator(n_qubits = translator.n_qubits, initialize="x", noisy=translator.noisy, noise_strength = args.noise_strength, noise_model="aer")
    minimizer = tfq_minimizer.Minimizer(translator, mode="VQE", hamiltonian = problem, params = params, lr=learning_rate, shots=shots, patience=30, max_time_training=600, verbose=0)
    simplifier = penny_simplifier.PennyLane_Simplifier(translator)
    killer = tfq_killer.GateKiller(translator, translator_killer, hamiltonian=problem, params=params, lr=learning_rate, shots=shots, accept_wall = 2/args.acceptange_percentage)
    inserter = idinserter.IdInserter(translator, noise_in_rotations=1e-1, mutation_rate = 1.5)
    args_evaluator = {"n_qubits":translator.n_qubits, "problem":problem,"params":params,"nrun":args.itraj,"name":"HEA_fixed"}
    evaluator = tfq_evaluator.PennyLaneEvaluator(minimizer = minimizer, killer=killer, inserter = inserter, args=args_evaluator, lower_bound=translator.ground, stopping_criteria=1e-3, vans_its=args.vans_its, acceptange_percentage = acceptange_percentage)

    evaluator.load_dicts_and_displaying(evaluator.identifier)
    return evaluator


js = list(np.logspace(-3.7,-2.9,16))


evo = load_ev_HEA(js[0], itraj=1)


costj = {}
jj=[]
lens={}
ers=[]
for j in js:
    costj[j] = {k:[] for k in range(3)}
    lens[j] = {k:[] for k in range(3)}

    for i in range(50):
        try:
            evo = load_ev_HEA(j, itraj=i)
            for k in range(3):
                costj[j][k].append(evo.evolution[k][1])
                lens[j][k] = len(costj[j][k])

        except Exception:
            pass

lens

opt = {j:[] for j in js}
for k, j in enumerate(js):
    for l in range(3):
        opt[j].append(np.min(costj[j][l]))

evo.minimizer.noisy=False
ground = tfq_minimizer.compute_lower_bound_cost_vqe(evo.minimizer)

opts = np.array(list(opt.values()))
colors=["yellow","blue","red"]
ax=plt.subplot()
for k in range(3):
    ax.scatter(opt.keys(), opts[:,k],s=200,alpha=0.7, color=colors[k], label="HEA-{}".format(k+1))
ax.plot(opt.keys(), ground*np.ones(len(opt.keys())),'--',color="black",label="ground")
ax.legend()
ax.set_xscale("log")
ax.set_xlabel(r'$\lambda$',size=20)
ax.set_ylabel(r'$E_{HEA}$',size=20)
