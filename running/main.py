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

reload(simplification_misc)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--problem", type=str, default="XXZ")
parser.add_argument("--params", type=list, default=[1., 1.1])
parser.add_argument("--nrun", type=int, default=0)
args = parser.parse_args()


args = {"problem":"XXZ", "params":[1.,.1],"nrun":0}
fargs = miscrun.FakeArgs(args)
problem = fargs.problem
params = list(fargs.params)
g,J = params


translator = penny_translator.PennyLaneTranslator(n_qubits = 4, initialize="u1")
translator_killer = penny_translator.PennyLaneTranslator(n_qubits = 4, initialize="u1")
minimizer = penny_variational.PennyModel(translator,lr=0.1, shots=100, g=g, J=J)
simplifier = penny_simplifier.PennyLane_Simplifier(translator)


args_evaluator = {"n_qubits":translator.n_qubits, "problem":problem,"params":params,"nrun":fargs.nrun}
evaluator = penny_evaluator.PennyLaneEvaluator(args=args_evaluator, lower_bound_cost=translator.ground, nrun=fargs.nrun, stopping_criteria=1e-3)

killer = penny_killer.GateKiller(translator, translator_killer, lr=0.1, shots=100, g=g, J=J)
inserter = idinserter.IdInserter(translator.n_qubits)

circuit_db, [cost, resolver, history] = model.variational(epochs=1, verbose=0)

circuit, circuit_db = translator.give_circuit(translator.db_train)
minimized_db, [cost, resolver, history_training] = minimizer.variational(epochs=10)
evaluator.add_step(minimized_db, cost, relevant=True, operation="variational", history = history_training.history["cost"])

### reduce circuit (same cost) ###
simplified_db, ns =  simplifier.reduce_circuit(minimized_db)
newcost = minimizer.give_cost(simplified_db)
reduced_db, reduced_cost, ops = simplification_misc.kill_and_simplify(simplified_db, newcost, killer, simplifier)
evaluator.add_step(simplified_db, reduced_cost, relevant=True, operation="reduction", history = ops)


circuit_db = reduced_db.copy()
for vans_it in range(evaluator.vans_its):
    print("vans_it: {}\n current cost: {}\ntarget cost: {} \nrelative error: {}\n\n\n".format(vans_it, cost, minimizer.lower_bound_cost, (cost-minimizer.lower_bound_cost)/abs(minimizer.lower_bound_cost)))
    mutated_db, number_mutations = inserter.mutate(circuit_db)
    mutated_cost = minimizer.give_cost(mutated_db)
    evaluator.add_step(mutated_db, mutated_cost, relevant=False, operation="mutation", history = number_mutations)

    simplified_db, ns =  simplifier.reduce_circuit(mutated_db)
    simplified_cost = minimizer.give_cost(simplified_db)
    evaluator.add_step(simplified_db, simplified_cost, relevant=False, operation="simplification", history = ns)

    minimized_db, [cost, resolver, history_training] = minimizer.variational(simplified_db)
    evaluator.add_step(minimized_db, cost, relevant=False, operation="variational", history = history_training.history["cost"])

    accept_cost, stop = evaluator.accept_cost(cost)
    if accept_cost == True:

        reduced_db, reduced_cost, ops = kill_and_simplify(simplified_db, cost, killer, simplifier)
        evaluator.add_step(reduced_db, reduced_cost, relevant=False, operation="reduction", history = ops)

        minimized_db, [cost, resolver, history_training] = minimizer.variational(reduced_db)
        evaluator.add_step(minimized_db, cost, relevant=True, operation="variational", history = history_training.history["cost"])

        circuit_db = minimized_db.copy()

    if stop == True:
        print("ending VAns")
        print("\n final cost: {}\ntarget cost: {} \n\n\n\n".format(cost, minimizer.lower_bound_cost))
        break














#
#
# simplified_db, ns =  simplifier.reduce_circuit(minimized_db)
# newcost = minimizer.give_cost(simplified_db)
# reduced_db, reduced_cost, ops = kill_and_simplify(simplified_db, newcost, killer, simplifier)
# evaluator.add_step(simplified_db, reduced_cost, relevant=True, operation="reduction", history = ops)
#
# kill_and_simplify
#
#
#
# #killed_db, new_cost, murder_attempt = pk.remove_irrelevant_gates(cost, circuit_db)
# cost
# new_cost
#
#
#
#
# inserter.mutate(translator.db_train)
#
# ###
