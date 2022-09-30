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


from utilities.evaluator.misc import *
import pickle






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


def load_ev_vans(ns, itraj=1):

    args = {"problem":"TFIM", "params":"[1.,1.]","nrun":0, "shots":0, "epochs":500, "n_qubits":4, "vans_its":200,"itraj":itraj, "noisy":True,
            "noise_strength":ns, "acceptange_percentage": 0.01, "L_HEA":1,"run_name":"VANS","noise_model":"aer"}
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
    args_evaluator = {"n_qubits":translator.n_qubits, "problem":problem,"params":params,"nrun":args.itraj,"name":"VANS"}
    evaluator = tfq_evaluator.PennyLaneEvaluator(minimizer = minimizer, killer=killer, inserter = inserter, args=args_evaluator, lower_bound=translator.ground, stopping_criteria=1e-3, vans_its=args.vans_its, acceptange_percentage = acceptange_percentage)

    evaluator.load_dicts_and_displaying(evaluator.identifier)
    return evaluator#import os



ncores = 16
js = list(np.logspace(-3.7,-2.9,16))

costj = {}
jj=[]
lens={}
ers=[]
for j in js:
    costj[j] = {k:[] for k in range(3)}
    lens[j] = {k:[] for k in range(3)}
    for i in range(10):
        try:
            evo = load_ev_HEA(j, itraj=i)
            for k in range(3):
                costj[j][k].append(evo.evolution[k][1])
                lens[j][k] = len(costj[j][k])
        except Exception:
            pass


evo = load_ev_HEA(js[0], itraj=1)
evo.minimizer.noisy=False
ground = tfq_minimizer.compute_lower_bound_cost_vqe(evo.minimizer)


costs_HEA = []
for ns in js:
    l=[]
    for k in range(3):
        l.append(np.min(costj[ns][k]))
    costs_HEA.append(l)



cost_vans={j:[] for j in js}
l={k:0 for k in js}
evals={}
for j in tqdm(js):
    evals[j] = []
    for i in range(70):
        try:
            evo = load_ev_vans(j, itraj=i)
            evals[j].append([evo.evolution[k][:2] for k in range(len(evo.evolution.keys()))])
            cost_vans[j].append(evo.evolution[evo.get_best_iteration()][1])
            l[j] +=1
        except Exception:
            pass

evals_4qb_vans = evals
eva = evo
ev_opt= {}
for j,k in evals_4qb_vans.items():
    costs_vans_j = []
    for itraj in k:
        costs_vans_j.append(np.min([itraj[k][1] for k in range(len(itraj))]))
    ind_min_j_vans = np.argmin(costs_vans_j)
    ev_opt[j] = k[ind_min_j_vans]
best_structures_vans = []
cost_vans = {}
for ns in tqdm(js):
    ind_opt_db_j = np.argmin([ev_opt[ns][vans_it][1] for vans_it in range(len(ev_opt[ns]))])
    cost_vans[j] = ev_opt[ns][ind_opt_db_j][0]
    stru=database.describe_circuit(eva.minimizer.translator, ev_opt[ns][ind_opt_db_j][0])
    best_structures_vans.append(stru)



opt_vans=[]
for k in cost_vans.values():
    opt_vans.append(np.min(k))

costs_HEA=np.array(costs_HEA)
step=2
colors=["red","blue","black"]
plt.figure(figsize=(20,20))
s=30
SS=10*s
ll=3
st=int(.7*s)
ax=plt.subplot()
for k in range(3):
    ax.plot(js[::step],costs_HEA[:,k][::step],'-', color=colors[k], linewidth=ll,label="{}-HEA".format(k+1))
    ax.scatter(js[::step],costs_HEA[:,k][::step],s=SS, color=colors[k])
ax.plot(js[::step], opt_vans[::step],'.-',linewidth=ll,color="green", label="VAns")
ax.scatter(js[::step], opt_vans[::step],s=SS,color="green")
ax.plot(js[::step], np.ones(len(opt_vans[::step]))*ground,'--',linewidth=ll,color="brown", label="Ground state energy (noiseless)")
ax.set_xlabel(r'$\lambda$',size=s)
ax.set_ylabel(r'$Energy$',size=s)
ax.legend(prop={"size":20})
ax.set_xscale("log")
ax.xaxis.set_tick_params(labelsize=st)
ax.yaxis.set_tick_params(labelsize=st)




path = get_def_path() + "figures/4qubits_lambda_model_tfim/"
os.makedirs(path,exist_ok=True)

os.makedirs("data",exist_ok=True)
with open("data/evals_4qb.pickle","wb") as f:
    pickle.dump(evals,f)


np.save("data/opt_vans_4qb",opt_vans)
np.save("data/hea_costs_optimal_4qb",costs_HEA)
np.save("data/structures_vans_best_4qb",best_structures_vans)




colors=["red","blue","black"]
plt.figure(figsize=(20,20))
s=30
SS=10*s
ll=3
st=int(.7*s)
ax=plt.subplot()
for k in range(2):
    ax.plot(js[::step],costs_HEA[:,k][::step],'-', color=colors[k], linewidth=ll,label="{}-HEA".format(k+1))
    ax.scatter(js[::step],costs_HEA[:,k][::step],s=SS, color=colors[k])
ax.plot(js[::step], opt_vans[::step],'.-',linewidth=ll,color="green", label="VAns")
ax.scatter(js[::step], opt_vans[::step],s=SS,color="green")
ax.plot(js[::step], np.ones(len(opt_vans[::step]))*ground,'--',linewidth=ll,color="brown", label="Ground state energy (noiseless)")
ax.set_xlabel(r'$\lambda$',size=s)
ax.set_ylabel(r'$Energy$',size=s)
ax.legend(prop={"size":20})
ax.xaxis.set_tick_params(labelsize=st)
ax.set_xscale("log")
ax.yaxis.set_tick_params(labelsize=st)

axi=ax.inset_axes([.1,.6,.35,.2])
axi.plot(js[::step], opthea[::step,1],'--', linewidth=ll,label="#CNOT HEA")
axi.plot(js[::step], opthea[::step,0],'-', linewidth=ll,label="#Parameters HEA")
axi.plot(js[::step], np.stack(best_structures_vans)[::step,1],'--', linewidth=ll,label="#CNOT VAns")
axi.plot(js[::step], np.stack(best_structures_vans)[::step,0], linewidth=ll,label="#Parameters VAns")
axi.legend(prop={"size":8})
axi.set_xscale("log")
axi.xaxis.set_tick_params(labelsize=int(.5*st))
axi.yaxis.set_tick_params(labelsize=int(.5*st))
axi.set_xlabel(r'$\lambda$',size=int(.5*s))
axi.set_ylabel("circuit structure",size=int(.5*s))
plt.savefig("data/4qb_noise.pdf")











###
