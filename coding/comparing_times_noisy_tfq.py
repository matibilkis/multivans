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


#### tfq.layers.NoisyPQC
translator = tfq_translator.TFQTranslator(n_qubits = 8, initialize="x")#, device_name="forest.numpy_wavefunction")
minimizer = tfq_minimizer.Minimizer(translator, mode="VQE", hamiltonian = problem, params = params, lr=learning_rate, shots=shots, g=g, J=J, patience=100, max_time_training=600)

circuit_db = translator.initialize(mode="x")
circuit, circuit_db = translator.give_circuit(translator.db_train)

nois = circuit + cirq.Circuit(cirq.depolarize(.01).on_each(*circuit.all_qubits()))

lala = tfq.layers.NoisyPQC(nois, minimizer.observable, repetitions=1000,sample_based=False)
inpu = tfq.convert_to_tensor([cirq.Circuit([])])
lala.trainable_variables[0].assign(tf.convert_to_tensor(list(database.get_trainable_params_value(translator, circuit_db))))
def get_gr():
    with tf.GradientTape() as tape:
        tape.watch(lala.trainable_variables)
        cost = tf.reduce_sum(lala(inpu))
    return tape.gradient(cost, lala.trainable_variables)

%timeit get_gr()






#### tfq.noise.expectation
translator = tfq_translator.TFQTranslator(n_qubits = 8, initialize="x")#, device_name="forest.numpy_wavefunction")
minimizer = tfq_minimizer.Minimizer(translator, mode="VQE", hamiltonian = problem, params = params, lr=learning_rate, shots=shots, g=g, J=J, patience=100, max_time_training=600)

circuit_db = translator.initialize(mode="x")
circuit, circuit_db = translator.give_circuit(translator.db_train, unresolved=True)

nois = circuit + cirq.Circuit(cirq.depolarize(.01).on_each(*circuit.all_qubits()))
values = np.array([database.get_trainable_params_value(translator,circuit_db)])

noisy = tfq.convert_to_tensor([nois])
symbols = database.get_trainable_symbols(translator, circuit_db)
tfqobs = tfq.convert_to_tensor([minimizer.observable])
samples = np.array([[1000]*len(minimizer.observable)])


diff = tfq.differentiators.ForwardDifference()
my_differentiable_op = diff.generate_differentiable_op(sampled_op=tfq.noise.expectation)
values = tf.convert_to_tensor(values)
tsymbols = tf.convert_to_tensor(symbols)

def get_grad():
    with tf.GradientTape() as tape:
        tape.watch(values)
        p = my_differentiable_op( noisy,  tsymbols, values, tfqobs, samples)
    tape.gradient(p,values)
%timeit get_grad()


class NoisyAvg(tf.keras.layers.Layer):
    def __init__(self, units, observable, symbols):
        super(NoisyAvg, self).__init__()
        self.w = self.add_weight(
            shape=(1, units), initializer="random_normal", trainable=True
        )
        self.samples = np.array([[1000]*len(observable)])
        self.symbols = tf.convert_to_tensor(symbols)
        self.observable = tfq.convert_to_tensor([observable])
        self.lay = tfq.differentiators.ForwardDifference().generate_differentiable_op(sampled_op = tfq.noise.expectation)

    def call(self, inputs):
        return tf.math.reduce_sum(self.lay( inputs,  self.symbols, self.w, self.observable, self.samples), axis=-1)

lay = NoisyAvg(8, minimizer.observable, symbols)

lay(noisy)

### what happens if i create a model ? (maybe compiling gets thing faster ?)

class model(tf.keras.Model):
    def __init__(self, observable, symbols, initial_params=None ):
        super(model,self).__init__()
        self.lay = NoisyAvg(8, minimizer.observable, symbols)
#        tfq.differentiators.ForwardDifference().generate_differentiable_op(sampled_op = tfq.noise.expectation)
        self.observable = tfq.convert_to_tensor([observable])
        self.samples = np.array([[1000]*len(observable)])

        self.symbols = tf.convert_to_tensor(symbols)
        #self.tvar = [tf.ones(len(symbols))]
        self.tvar =tf.Variable(initial_value=tf.ones((1,len(symbols))))#random.uniform((1,len(symbols))), trainable=True)

        #if not (initial_params is None):
    #        self.tvar[0].assign(tf.Variable(tf.convert_to_tensor(initial_params.astype(np.float32))))

    def call(self, inputs):
        """
        inputs: tfq circuits (resolved or not, to train one has to feed unresolved).
        """
        f = tf.math.reduce_sum(self.lay( inputs),axis=-1)#,  self.symbols, self.tvar, self.observable, self.samples), axis=-1)
        return f

    def train_step(self, data):
        x,y=data
        with tf.GradientTape() as tape:
            tape.watch(self.tvar)
            preds = self(x)#,training=True)
            cost = self.compiled_loss(preds, preds) #notice that compiled loss takes care only about the preds
        grads=tape.gradient(cost,self.tvar)
        self.optimizer.apply_gradients(zip(grads, self.tvar))
        return {k.name:k.result() for k in self.metrics}

translator = tfq_translator.TFQTranslator(n_qubits = 8, initialize="x")#, device_name="forest.numpy_wavefunction")
minimizer = tfq_minimizer.Minimizer(translator, mode="VQE", hamiltonian = problem, params = params, lr=learning_rate, shots=shots, g=g, J=J, patience=100, max_time_training=600)
circuit_db = translator.initialize(mode="x")
circuit, circuit_db = translator.give_circuit(translator.db_train, unresolved=True)
nois = circuit + cirq.Circuit(cirq.depolarize(.01).on_each(*circuit.all_qubits()))
values = np.array([database.get_trainable_params_value(translator,circuit_db)])
noisy = tfq.convert_to_tensor([nois])

symbols = database.get_trainable_symbols(translator, circuit_db)
momo = model(minimizer.observable,symbols)

momo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss= EnergyLoss())
momo(noisy)


momo.trainable_variables


momo.optimizer


#momo.train_step([noisy,noisy])


#%timeit momo.train_step([noisy, noisy])

with tf.GradientTape() as t:
    t.watch(momo.tvar)
    g = momo(noisy)
    c = momo.compiled_loss(g,g)
gg = t.gradient(c, momo.tvar)

gg
momo.optimizer.apply_gradients(zip(gg, momo.tvar))


momo.fit(x=noisy, y=noisy, epochs=10)










### model wth NoisyPQC layer

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
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {k.name:k.result() for k in self.metrics}

momo = model(nois, minimizer.observable)
momo(inpu)
momo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss= EnergyLoss())
%timeit momo.train_step([inpu, inpu])
momo.fit(x=inpu, y=inpu,epochs=10)
### this takes like 1 sec per gradient descent step
