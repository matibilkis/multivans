import os
import sys
sys.path.insert(0, os.getcwd())
from importlib import reload

import utilities.translator.pennylane_translator as penny_translator
import utilities.evaluator.pennylane_evaluator as penny_evaluator
import utilities.variational.pennylane_model as penny_variational
import utilities.database.database as database
import utilities.database.templates as templates

import tensorflow as tf
import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
from tqdm import tqdm

reload(penny_evaluator)
reload(penny_translator)
reload(penny_variational)
reload(templates)
reload(database)
[g, J] = [1., 1.1]

translator = penny_translator.PennyLaneTranslator(n_qubits = 4)
circuit_db = translator.initialize(mode="x")
model = penny_variational.PennyModel(translator,lr=0.1, shots=100, g=g, J=J)

args_evaluator = {"n_qubits":translator.n_qubits, "problem":"XXZ","params":[g,J]}
evaluator = penny_evaluator.PennyLaneEvaluator(args=args_evaluator, lower_bound_cost=translator.ground, nrun=0, stopping_criteria=1e-3)

qnode, db = translator.give_circuit(db)
circuit_db, [cost, resolver, history] = model.variational(epochs=1, verbose=0)

import utilities.simplification.simplifier as penny_simplifier
reload(penny_simplifier)

simplifier = penny_simplifier.PennyLane_Simplifier(translator)
circuit, circuit_db = translator.give_circuit(circuit_db)
simplifier.reduce_circuit(translator.db_train)
