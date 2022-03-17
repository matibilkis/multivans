from utilities.idinserter import IdInserter
from utilities.evaluator import Evaluator
from utilities.gate_killer import GateKiller
from utilities.simplifier import Simplifier
from utilities.variational import Minimizer
from utilities.circuit_database import CirqTranslater
from utilities.templates import *
from utilities.misc import kill_and_simplify
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--eta1", type=float, default=0.0)
parser.add_argument("--eta2", type=float, default=1.)
args = parser.parse_args()

eta1 = args.eta1
eta2 = args.eta2
etas = [eta1,eta2]

translator = CirqTranslater(3, untouchable_blocks = [1], discard_qubits=[2])
simplifier = Simplifier(translator)
inserter = IdInserter(translator.n_qubits, untouchable_blocks=translator.untouchable_blocks, untouchable_qubits = [2])
killer = GateKiller(translator, mode="discrimination", params=etas)
minimizer = Minimizer(translator, mode="discrimination", params=etas)

args_evaluator = {"n_qubits":translator.n_qubits, "problem":"acd","params":etas}
evaluator = Evaluator(args=args_evaluator, lower_bound_cost=minimizer.lower_bound_cost, nrun=0, stopping_criteria=1e-3)


### prepare initial circuit ####
channel_db = amplitude_damping_db(translator, qubits_ind=[0,inserter.untouchable_qubits[0]], eta=1, block_id = translator.untouchable_blocks[0])
circuit_db = concatenate_dbs([x_layer(translator,qubits_ind=[0,1], block_id=0), channel_db, x_layer(translator,qubits_ind=[0,1], block_id=2)])
circuit, circuit_db = translator.give_circuit(circuit_db)

# optimize continuous parameters ##
minimized_db, [cost, resolver, history_training] = minimizer.variational(circuit_db)
evaluator.add_step(minimized_db, cost, relevant=True, operation="variational", history = history_training.history["cost"])

### reduce circuit (same cost) ###
simplified_db, ns =  simplifier.reduce_circuit(minimized_db)
newcost = minimizer.give_cost(simplified_db)
reduced_db, reduced_cost, ops = kill_and_simplify(simplified_db, newcost, killer, simplifier)
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
