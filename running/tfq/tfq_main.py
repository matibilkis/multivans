import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import os
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




#
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--problem", type=str, default="XXZ")
parser.add_argument("--n_qubits", type=int, default=8)
parser.add_argument("--params", type=str, default="[1., 1.1]")
parser.add_argument("--nrun", type=int, default=0)
parser.add_argument("--shots", type=int, default=0)
parser.add_argument("--epochs", type=int, default=500)
args = parser.parse_args()

convert_shorts = lambda x: None if x==0 else x
#args = {"problem":"XXZ", "params":"[1.,.1]","nrun":0, "shots":0, "epochs":500, "n_qubits":10}
#args = miscrun.FakeArgs(args)
problem = args.problem
params = ast.literal_eval(args.params)
g,J = params
shots = convert_shorts(args.shots)
epochs = args.epochs
n_qubits = args.n_qubits

learning_rate=0.01

translator = tfq_translator.TFQTranslator(n_qubits = n_qubits, initialize="x")#, device_name="forest.numpy_wavefunction")
translator_killer = tfq_translator.TFQTranslator(n_qubits = translator.n_qubits, initialize="x")#, device_name=translator.device_name)
minimizer = tfq_minimizer.Minimizer(translator, mode="VQE", hamiltonian = problem, params = params, lr=learning_rate, shots=shots, g=g, J=J, patience=10, max_time_training=600)

simplifier = penny_simplifier.PennyLane_Simplifier(translator)
killer = tfq_killer.GateKiller(translator, translator_killer, lr=learning_rate, shots=shots, g=g, J=J)
inserter = idinserter.IdInserter(translator)
args_evaluator = {"n_qubits":translator.n_qubits, "problem":problem,"params":params,"nrun":args.nrun}
evaluator = tfq_evaluator.PennyLaneEvaluator(args=args_evaluator, lower_bound=translator.ground, nrun=args.nrun, stopping_criteria=1e-3)



#### begin the algorithm
circuit_db = translator.initialize(mode="x")
circuit, circuit_db = translator.give_circuit(translator.db_train)

minimized_db, [cost, resolver, history] = minimizer.variational(circuit_db)

evaluator.add_step(minimized_db, cost, relevant=True, operation="variational", history = history.history)#$history_training.history["cost"])
circuit, circuit_db = translator.give_circuit(minimized_db)



for vans_it in range(evaluator.vans_its):
    print("vans_it: {}\n current cost: {}\ntarget cost: {} \nrelative error: {}\n\n\n".format(vans_it, cost, evaluator.lower_bound, (cost-evaluator.lower_bound)/abs(evaluator.lower_bound)))
    mutated_db, number_mutations = inserter.mutate(circuit_db)
    mutated_cost = minimizer.build_and_give_cost(mutated_db)

    evaluator.add_step(mutated_db, mutated_cost, relevant=False, operation="mutation", history = number_mutations)

    simplified_db, ns =  simplifier.reduce_circuit(mutated_db)
    simplified_cost = minimizer.build_and_give_cost(simplified_db)
    evaluator.add_step(simplified_db, simplified_cost, relevant=False, operation="simplification", history = ns)

    minimized_db, [cost, resolver, history_training] = minimizer.variational(simplified_db)
    evaluator.add_step(minimized_db, cost, relevant=False, operation="variational", history = history_training.history["cost"])

    accept_cost, stop = evaluator.accept_cost(cost)
    if accept_cost == True:

        reduced_db, reduced_cost, ops = simplification_misc.kill_and_simplify(simplified_db, cost, killer, simplifier)
        evaluator.add_step(reduced_db, reduced_cost, relevant=False, operation="reduction", history = ops)

        minimized_db, [cost, resolver, history_training] = minimizer.variational(reduced_db)
        evaluator.add_step(minimized_db, cost, relevant=True, operation="variational", history = history_training.history["cost"])

        circuit_db = minimized_db.copy()
    else:
        circuit, circuit_db = translator.give_circuit(circuit_db)

    if stop == True:
        print("ending VAns")
        delta_cost = (cost-evaluator.lower_bound)/abs(evaluator.lower_bound)
        print("\n final cost: {}\ntarget cost: {}, relative error: {} \n\n\n\n".format(cost, evaluator.lower_bound, delta_cost))
        break