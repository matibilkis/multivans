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


# #
# reload(tfq_minimizer)
# reload(tfq_minimizer)
# reload(tfq_translator)
# reload(penny_simplifier)


# parser = argparse.ArgumentParser(add_help=False)
# parser.add_argument("--problem", type=str, default="XXZ")
# parser.add_argument("--n_qubits", type=int, default=4)
# parser.add_argument("--params", type=str, default="[1., 1.1]")
# parser.add_argument("--nrun", type=int, default=0)
# parser.add_argument("--shots", type=int, default=0)
# parser.add_argument("--epochs", type=int, default=5000)
# parser.add_argument("--vans_its", type=int, default=200)
# parser.add_argument("--itraj", type=int, default=1)
#
# args = parser.parse_args()


start = datetime.now()

args = {"problem":"TFIM", "params":"[1.,.1]","nrun":0, "shots":0, "epochs":500, "n_qubits":10, "vans_its":200,"itraj":1}
args = miscrun.FakeArgs(args)
problem = args.problem
params = ast.literal_eval(args.params)
g,J = params
shots = miscrun.convert_shorts(args.shots)
epochs = args.epochs
n_qubits = args.n_qubits
learning_rate=0.01
tf.random.set_seed(args.itraj)
np.random.seed(args.itraj)

translator = tfq_translator.TFQTranslator(n_qubits = n_qubits, initialize="x")#, device_name="forest.numpy_wavefunction")
translator_killer = tfq_translator.TFQTranslator(n_qubits = translator.n_qubits, initialize="x")#, device_name=translator.device_name)
minimizer = tfq_minimizer.Minimizer(translator, mode="VQE", hamiltonian = problem, params = params, lr=learning_rate, shots=shots, g=g, J=J, patience=100, max_time_training=600)


simplifier = penny_simplifier.PennyLane_Simplifier(translator)
killer = tfq_killer.GateKiller(translator, translator_killer, hamiltonian=problem, params=params, lr=learning_rate, shots=shots, g=g, J=J)
inserter = idinserter.IdInserter(translator, noise_in_rotations=0.1)
args_evaluator = {"n_qubits":translator.n_qubits, "problem":problem,"params":params,"nrun":args.nrun}
evaluator = tfq_evaluator.PennyLaneEvaluator(minimizer = minimizer, args=args_evaluator, lower_bound=translator.ground, nrun=args.itraj, stopping_criteria=1e-3, vans_its=args.vans_its)


#### begin the algorithm
circuit_db = translator.initialize(mode="x")
circuit, circuit_db = translator.give_circuit(translator.db_train, unresolved=False)


minimizer.build_and_give_cost(circuit_db)
nois = circuit + cirq.Circuit(cirq.depolarize(.01).on_each(*circuit.all_qubits()))
values = np.array([database.get_trainable_params_value(translator,circuit_db)])


noisy = tfq.convert_to_tensor([nois])
symbols = database.get_trainable_symbols(translator, circuit_db)
tfqobs = tfq.convert_to_tensor([minimizer.observable[:1]])

samples = np.array([[1000]*1])
tfq.noise.expectation( noisy,  symbols, values, tfqobs, samples)


diff = tfq.differentiators.ForwardDifference()
my_differentiable_op = diff.generate_differentiable_op(sampled_op=tfq.noise.expectation)
my_differentiable_op( noisy,  symbols, values, tfqobs, samples)




translator = tfq_translator.TFQTranslator(n_qubits = 4, initialize="x")#, device_name="forest.numpy_wavefunction")
minimizer = tfq_minimizer.Minimizer(translator, mode="VQE", hamiltonian = problem, params = params, lr=learning_rate, shots=shots, g=g, J=J, patience=100, max_time_training=600)


circuit_db = translator.initialize(mode="x")
circuit, circuit_db = translator.give_circuit(translator.db_train)

nois = circuit + cirq.Circuit(cirq.depolarize(.01).on_each(*circuit.all_qubits()))

lala = tfq.layers.NoisyPQC(nois, minimizer.observable, repetitions=1000,sample_based=False)
inpu = tfq.convert_to_tensor([cirq.Circuit([])])
lala.trainable_variables[0].assign(tf.convert_to_tensor(list(database.get_trainable_params_value(translator, circuit_db))))
with tf.GradientTape() as tape:
    tape.watch(lala.trainable_variables)
    cost = tf.reduce_sum(lala(inpu))

cost
minimizer.build_and_give_cost(circuit_db)

class model(tf.keras.Model):
    def __init__(self, cir, observable ):
        super(model,self).__init__()
        self.lay = tfq.layers.NoisyPQC(cir, observable, repetitions=1000,sample_based=False)

    def call(self, inputs):
        """
        inputs: tfq circuits (resolved or not, to train one has to feed unresolved).
        """
        feat = inputs
        f = self.lay(inputs)#feat, operators=self.observable, symbol_names=self.symbols)
        f = tf.math.reduce_sum(f,axis=-1)
        return f

    def train_step(self, data):
        x,y=data
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            preds = self(x)#,training=True)
            cost = self.compiled_loss(preds, preds) #notice that compiled loss takes care only about the preds
        grads=tape.gradient(cost,self.trainable_variables)
        #self.gradient_norm.update_state(tf.reduce_sum(tf.pow(grads[0],2)))
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #self.cost_value.update_state(cost)
        #self.lr_value.update_state(self.optimizer.lr)
        return {k.name:k.result() for k in self.metrics}


momo = model(nois, minimizer.observable)
momo(inpu)
momo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss= EnergyLoss())
momo.train_step([inpu, inpu])
momo.fit(x=inpu, y=inpu, epochs=100)

### this takes like 1 sec per gradient descent step












#### customized... this is nicer, but only for 1 qubit !!
translator = tfq_translator.TFQTranslator(n_qubits = 1, initialize="x")#, device_name="forest.numpy_wavefunction")
minimizer = tfq_minimizer.Minimizer(translator, mode="VQE", hamiltonian = problem, params = params, lr=learning_rate, shots=shots, g=g, J=J, patience=100, max_time_training=600)


circuit_db = translator.initialize(mode="x")
circuit, circuit_db = translator.give_circuit(translator.db_train, unresolved=True)

nois = circuit + cirq.Circuit(cirq.depolarize(.01).on_each(*circuit.all_qubits()))
values = np.array([database.get_trainable_params_value(translator,circuit_db)])


noisy = tfq.convert_to_tensor([nois])
symbols = database.get_trainable_symbols(translator, circuit_db)
tfqobs = tfq.convert_to_tensor([minimizer.observable[:1]])

samples = np.array([[1000]*1])
tfq.noise.expectation( noisy,  symbols, values, tfqobs, samples)


diff = tfq.differentiators.ForwardDifference()
my_differentiable_op = diff.generate_differentiable_op(sampled_op=tfq.noise.expectation)
my_differentiable_op( noisy,  symbols, values, tfqobs, samples)

values = tf.convert_to_tensor(values)
with tf.GradientTape() as tape:
    tape.watch(values)
    p = my_differentiable_op( noisy,  symbols, values, tfqobs, samples)
tape.gradient(p,values)




























#### customized... this is nicer, but only for 1 qubit !!
translator = tfq_translator.TFQTranslator(n_qubits = 2, initialize="x")#, device_name="forest.numpy_wavefunction")
minimizer = tfq_minimizer.Minimizer(translator, mode="VQE", hamiltonian = problem, params = params, lr=learning_rate, shots=shots, g=g, J=J, patience=100, max_time_training=600)


circuit_db = translator.initialize(mode="x")
circuit, circuit_db = translator.give_circuit(translator.db_train, unresolved=True)

nois = circuit + cirq.Circuit(cirq.depolarize(.01).on_each(*circuit.all_qubits()))
values = np.array([database.get_trainable_params_value(translator,circuit_db)])


noisy = tfq.convert_to_tensor([nois])
symbols = database.get_trainable_symbols(translator, circuit_db)
tfqobs = tfq.convert_to_tensor([minimizer.observable[:1]])

samples = np.array([[1000]*1])
tfq.noise.expectation( noisy,  symbols, values, tfqobs, samples)


diff = tfq.differentiators.ForwardDifference()
my_differentiable_op = diff.generate_differentiable_op(sampled_op=tfq.noise.expectation)
my_differentiable_op( noisy,  symbols, values, tfqobs, samples)

values = tf.convert_to_tensor(values)

with tf.GradientTape() as tape:
    tape.watch(values)
    p = my_differentiable_op( noisy,  symbols, values, tfqobs, samples)
























my_op = tfq.get_expectation_op()
linear_differentiator = tfq.differentiators.ForwardDifference(2, 0.01)


op = linear_differentiator.generate_differentiable_op(
    analytic_op=my_op
)

qubit = cirq.GridQubit(0, 0)
circuit = tfq.convert_to_tensor([
    cirq.Circuit(cirq.X(qubit) ** sympy.Symbol('alpha'))
])


psums = tfq.convert_to_tensor([[cirq.Z(qubit)]])
symbol_values_array = np.array([[0.123]], dtype=np.float32)


# Calculate tfq gradient.
symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
with tf.GradientTape() as g:
    g.watch(symbol_values_tensor)
    expectations = op(circuit, ['alpha'], symbol_values_tensor, psums)

# Gradient would be: -50 * f(x + 0.02) +  200 * f(x + 0.01) - 150 * f(x)
grads = g.gradient(expectations, symbol_values_tensor)
grads





sympy.__version__
tfq.__version__
cirq.__version__


my_op = tfq.get_expectation_op()
linear_differentiator = tfq.differentiators.ForwardDifference(2, 0.01)
op = linear_differentiator.generate_differentiable_op(
    analytic_op=my_op
)
qubit = cirq.GridQubit(0, 0)
circuit = tfq.convert_to_tensor([
    cirq.Circuit([cirq.X(qubit) ** sympy.Symbol('alpha'), cirq.X(qubit) ** sympy.Symbol('beta')])
])

psums = tfq.convert_to_tensor([[cirq.Z(qubit)]])
symbol_values_array = np.array([[0.123, .2]], dtype=np.float32)

# Calculate tfq gradient.
symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
with tf.GradientTape() as g:
    g.watch(symbol_values_tensor)
    expectations = op(circuit, ['alpha','beta'], symbol_values_tensor, psums)
grads = g.gradient(expectations, symbol_values_tensor)
grads

# Gradient would be: -50 * f(x + 0.02) +  200 * f(x + 0.01) - 150 * f(x)
