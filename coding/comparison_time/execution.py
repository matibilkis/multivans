import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
sys.path.insert(0, os.getcwd())
from importlib import reload
import tensorflow as tf
import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
from tqdm import tqdm
import utilities.translator.pennylane_translator as penny_translator
import utilities.evaluator.pennylane_evaluator as penny_evaluator
import utilities.variational.pennylane_model as penny_variational
import utilities.simplification.simplifier as penny_simplifier
import utilities.simplification.misc as simplification_misc#.kill_and_simplify
import utilities.simplification.gate_killer as penny_killer
import utilities.database.database as database
import utilities.database.templates as templates
import utilities.mutator.idinserter as idinserter
import running.misc as miscrun
import argparse


reload(penny_translator)
reload(miscrun)
reload(idinserter)
reload(penny_variational)



#

convert_shorts = lambda x: None if x==0 else x
args = {"problem":"XXZ", "params":[1.,.1],"nrun":0, "shots":0, "epochs":500}
args = miscrun.FakeArgs(args)
problem = args.problem
params = list(args.params)
g,J = params
shots = convert_shorts(args.shots)
epochs = args.epochs

n_qubits = 10


translator = penny_translator.PennyLaneTranslator(n_qubits = n_qubits, initialize="x", device_name='lightning.qubit')#'qiskit.basicaer')#, backend='unitary_simulator')
translator_killer = penny_translator.PennyLaneTranslator(n_qubits = translator.n_qubits, initialize="x", device_name=translator.device_name)
minimizer = penny_variational.PennyModel(translator,lr=0.1, shots=shots, g=g, J=J, patience=10)
simplifier = penny_simplifier.PennyLane_Simplifier(translator)
killer = penny_killer.GateKiller(translator, translator_killer, lr=0.1, shots=shots, g=g, J=J)


inserter = idinserter.IdInserter(translator)
args_evaluator = {"n_qubits":translator.n_qubits, "problem":problem,"params":params,"nrun":args.nrun}
evaluator = penny_evaluator.PennyLaneEvaluator(args=args_evaluator, lower_bound=translator.ground, nrun=args.nrun, stopping_criteria=1e-3)



circuit, circuit_db = translator.give_circuit(translator.db_train)


minimized_db, [cost, resolver, history] = minimizer.variational(epochs=epochs, verbose=1)







###
