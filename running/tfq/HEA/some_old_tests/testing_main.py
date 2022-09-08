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


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--problem", type=str, default="XXZ")
parser.add_argument("--params", type=list, default=[1., 1.1])
parser.add_argument("--nrun", type=int, default=0)
parser.add_argument("--shots", type=int, default=0)

args = parser.parse_args()

convert_shorts = lambda x: None if x==0 else x
args = {"problem":"XXZ", "params":[1.,.1],"nrun":0, "shots":0}
args = miscrun.FakeArgs(args)
problem = args.problem
params = list(args.params)
g,J = params
shots = convert_shorts(args.shots)


translator = penny_translator.PennyLaneTranslator(n_qubits = 4, initialize="x")
translator_killer = penny_translator.PennyLaneTranslator(n_qubits = 4, initialize="x")
minimizer = penny_variational.PennyModel(translator,lr=0.1, shots=shots, g=g, J=J)
simplifier = penny_simplifier.PennyLane_Simplifier(translator)
killer = penny_killer.GateKiller(translator, translator_killer, lr=0.1, shots=shots, g=g, J=J)
inserter = idinserter.IdInserter(translator.n_qubits)

args_evaluator = {"n_qubits":translator.n_qubits, "problem":problem,"params":params,"nrun":args.nrun}
evaluator = penny_evaluator.PennyLaneEvaluator(args=args_evaluator, lower_bound_cost=translator.ground, nrun=args.nrun, stopping_criteria=1e-3)



#### begin the algorithm

circuit, circuit_db = translator.give_circuit(translator.db_train)

minimized_db, [cost, resolver, history] = minimizer.variational(epochs=100, verbose=0)

evaluator.add_step(minimized_db, cost, relevant=True, operation="variational", history = history.history)#$history_training.history["cost"])


plt.plot(history.history["cost"])
plt.plot(np.ones(len(history.history["cost"]))*translator.ground)

minimized_db
simplified_db, ns =  simplifier.reduce_circuit(minimized_db)
newcost = minimizer.give_cost(simplified_db)


### reduce circuit (same cost) ###
simplified_db, ns =  simplifier.reduce_circuit(minimized_db)
newcost = minimizer.give_cost(simplified_db)
reduced_db, reduced_cost, ops = simplification_misc.kill_and_simplify(simplified_db, newcost, killer, simplifier)
evaluator.add_step(reduced_db, reduced_cost, relevant=True, operation="reduction", history = ops)

circuit_db = reduced_db.copy()


mutated_db, number_mutations = inserter.mutate(circuit_db)

mc, mutated_db = translator.give_circuit(mutated_db)
translator.draw(mc, mutated_db)

mutated_cost = minimizer.give_cost(mutated_db)


mutated_cost
evaluator.add_step(mutated_db, mutated_cost, relevant=False, operation="mutation", history = number_mutations)

simplified_db, ns =  simplifier.reduce_circuit(mutated_db)
translator.draw(mc, minimizer.translator.db_train)


minimizer.build_model()
minimized_db, [cost, resolver, history] = minimizer.variational(epochs=100, verbose=0)

plt.plot(history.history["cost"])
plt.plot(np.ones(len(history.history["cost"]))*translator.ground)

stop

accept_cost, stop = evaluator.accept_cost(cost)


reduced_db, reduced_cost, ops = simplification_misc.kill_and_simplify(minimized_db, cost, killer, simplifier)

rdb, db = translator.give_circuit(minimized_db)
translator.draw(rdb, db)


rdb, db = translator.give_circuit(reduced_db)
translator.draw(rdb, db)
reduced_cost


minimized_db, [cost, resolver, history_training] = minimizer.variational(epochs = 100, verbose=0)


for vans_it in range(evaluator.vans_its):
    print("vans_it: {}\n current cost: {}\ntarget cost: {} \nrelative error: {}\n\n\n".format(vans_it, cost, minimizer.lower_bound_cost, (cost-minimizer.lower_bound_cost)/abs(minimizer.lower_bound_cost)))
    mutated_db, number_mutations = inserter.mutate(circuit_db)
    mutated_cost = minimizer.give_cost(mutated_db)
    evaluator.add_step(mutated_db, mutated_cost, relevant=False, operation="mutation", history = number_mutations)

    simplified_db, ns =  simplifier.reduce_circuit(mutated_db)
    simplified_cost = minimizer.give_cost(simplified_db)
    evaluator.add_step(simplified_db, simplified_cost, relevant=False, operation="simplification", history = ns)

    minimized_db, [cost, resolver, history_training] = minimizer.variational(epochs=100, verbose=0)
    evaluator.add_step(minimized_db, cost, relevant=False, operation="variational", history = history_training.history["cost"])

    accept_cost, stop = evaluator.accept_cost(cost)
    if accept_cost == True:

        reduced_db, reduced_cost, ops = kill_and_simplify(simplified_db, cost, killer, simplifier)
        evaluator.add_step(reduced_db, reduced_cost, relevant=False, operation="reduction", history = ops)

        minimized_db, [cost, resolver, history_training] = minimizer.variational(epochs = epochs, verbose=0)
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
