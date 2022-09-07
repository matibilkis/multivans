import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

num_threads = 1
os.environ["OMP_NUM_THREADS"] = "{}".format(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = "{}".format(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "{}".format(num_threads)

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
tf.config.set_soft_device_placement(True)

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


# #
reload(tfq_minimizer)
reload(tfq_minimizer)
reload(tfq_translator)
reload(penny_simplifier)


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--problem", type=str, default="TFIM")
parser.add_argument("--n_qubits", type=int, default=4)
parser.add_argument("--params", type=str, default="[1., 1.]")
parser.add_argument("--nrun", type=int, default=0)
parser.add_argument("--shots", type=int, default=0)
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--vans_its", type=int, default=200)
parser.add_argument("--itraj", type=int, default=1)
parser.add_argument("--noise_strength", type=float, default=.01)
parser.add_argument("--noisy", type=int, default=0)
parser.add_argument("--L_HEA", type=int, default=1)
parser.add_argument("--acceptange_percentage", type=float, default=1e-3)
parser.add_argument("--run_name", type=str, default="")


args = parser.parse_args()

reload(miscrun)
start = datetime.now()

# args = {"problem":"TFIM", "params":"[1.,1.]","nrun":0, "shots":0, "epochs":500, "n_qubits":4, "vans_its":200,"itraj":1, "noisy":False, "noise_strength":0.0, "acceptange_percentage": 0.01, "L_HEA":2, "run_name":""}
# args = miscrun.FakeArgs(args)
L_HEA = args.L_HEA
problem = args.problem
params = ast.literal_eval(args.params)
shots = miscrun.convert_shorts(args.shots)
epochs = args.epochs
n_qubits = args.n_qubits
learning_rate=1e-2
acceptange_percentage = args.acceptange_percentage
noise_strength = args.noise_strength
int_2_bool = lambda x: True if x==1 else False
noisy = int_2_bool(args.noisy)
tf.random.set_seed(abs(args.itraj))
np.random.seed(abs(args.itraj))

reload(idinserter)

translator = tfq_translator.TFQTranslator(n_qubits = n_qubits, initialize="x", noisy=args.noisy, noise_strength = noise_strength)#, device_name="forest.numpy_wavefunction")
translator_killer = tfq_translator.TFQTranslator(n_qubits = translator.n_qubits, initialize="x", noisy=translator.noisy, noise_strength = args.noise_strength)
minimizer = tfq_minimizer.Minimizer(translator, mode="VQE", hamiltonian = problem, params = params, lr=learning_rate, shots=shots, patience=30, max_time_training=0.5*3600, verbose=0)
simplifier = penny_simplifier.PennyLane_Simplifier(translator)
killer = tfq_killer.GateKiller(translator, translator_killer, hamiltonian=problem, params=params, lr=learning_rate, shots=shots, accept_wall = 2/args.acceptange_percentage)
inserter = idinserter.IdInserter(translator, noise_in_rotations=1e-1, mutation_rate = 1.5, prob_big=0.01, p3body=0.1 ,pu1=0.5)
args_evaluator = {"n_qubits":translator.n_qubits, "problem":problem,"params":params,"nrun":args.itraj, "name":args.run_name}
evaluator = tfq_evaluator.PennyLaneEvaluator(minimizer = minimizer, killer=killer, inserter = inserter, args=args_evaluator, lower_bound=translator.ground, stopping_criteria=1e-3, vans_its=args.vans_its, acceptange_percentage = acceptange_percentage, accuraccy_to_end=1e-2)



#### begin the algorithm
circuit_db = translator.initialize(mode="xz")
circuit, circuit_db = translator.give_circuit(circuit_db)
minimized_db, [cost, resolver, history] = minimizer.variational(circuit_db)

evaluator.add_step(minimized_db, cost, relevant=True, operation="variational", history = history.history)
circuit, circuit_db = translator.give_circuit(minimized_db)



progress = True
for vans_it in range(evaluator.vans_its):
    print("\n"*4 + "vans_it: {}\n Time since beggining: {} sec\ncurrent cost: {}\ntarget cost: {} \nrelative error: {}\n\n\n".format(vans_it, (datetime.now()-start).seconds, cost, evaluator.lower_bound, (cost-evaluator.lower_bound)/abs(evaluator.lower_bound)))
    print(translator.give_circuit(circuit_db,unresolved=False)[0], "\n","*"*30)

    mutated_db, number_mutations = inserter.mutate(circuit_db, cost, evaluator.lowest_cost)
    mutated_cost = minimizer.build_and_give_cost(mutated_db)

    if progress is True:
        print("mutated, new cost: {}, new circuit {}\n".format(mutated_cost, database.describe_circuit(translator,mutated_db)))

    evaluator.add_step(mutated_db, mutated_cost, relevant=False, operation="mutation", history = number_mutations)
    simplified_db, ns =  simplifier.reduce_circuit(mutated_db)
    simplified_cost = minimizer.build_and_give_cost(simplified_db)
    if progress is True:
        print("simplifyed, new cost {}, new circuit {}\nminimizing..\n".format(simplified_cost, database.describe_circuit(translator,simplified_db)))
    evaluator.add_step(simplified_db, simplified_cost, relevant=False, operation="simplification", history = ns)

    minimized_db, [cost, resolver, history_training] = minimizer.variational(simplified_db, parameter_perturbation_wall=1.)
    evaluator.add_step(minimized_db, cost, relevant=False, operation="variational", history = history_training.history)

    previous_cost = evaluator.lowest_cost
    accept_cost, stop, circuit_db = evaluator.accept_cost(cost, minimized_db)
    if accept_cost == True:

        if progress is True:
            print("accepted!, {}, {}".format(cost, previous_cost)+"\nReducing\n")
        reduced_db, reduced_cost, ops = simplification_misc.kill_and_simplify(circuit_db, cost, killer, simplifier)
        evaluator.add_step(reduced_db, reduced_cost, relevant=True, operation="reduction", history = ops)
        circuit_db = reduced_db.copy()

    if stop == True:
        print("ending VAns")
        if evaluator.lower_bound == -np.inf:
            print("Done, cost {}".format(cost) + "\n"*10)
        else:
            delta_cost = (cost-evaluator.lower_bound)/abs(evaluator.lower_bound)
            print("\n final cost: {}\ntarget cost: {}, relative error: {} \n Final circuit {}\n\n\n".format(cost, evaluator.lower_bound, delta_cost, database.describe_circuit(translator,circuit_db)))
            break
