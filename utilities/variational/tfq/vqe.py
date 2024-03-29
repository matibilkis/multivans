import tensorflow as tf
import tensorflow_quantum as tfq
import pandas as pd
from datetime import datetime
import utilities.database.database as database
import utilities.database.templates as templates
from utilities.variational.misc import *
tf.keras.backend.set_floatx('float32')


import cirq
import tensorflow_quantum as tfq
import tensorflow as tf
import numpy as np

def give_observable_vqe(minimizer, hamiltonian, params):
    if hamiltonian.upper() == "TFIM":
        database.check_params(params,2)
        g, J = params
        observable = [-float(g)*cirq.Z.on(q) for q in minimizer.qubits]
        for q in range(len(minimizer.qubits)):
            observable.append(-float(J)*cirq.X.on(minimizer.qubits[q])*cirq.X.on(minimizer.qubits[(q+1)%len(minimizer.qubits)]))
        return observable
    elif hamiltonian.upper() == "XXZ":
        database.check_params(params,2)
        g, J = params
        observable = [float(g)*cirq.Z.on(q) for q in minimizer.qubits]
        for q in range(len(minimizer.qubits)):
            observable.append(cirq.X.on(minimizer.qubits[q])*cirq.X.on(minimizer.qubits[(q+1)%len(minimizer.qubits)]))
            observable.append(cirq.Y.on(minimizer.qubits[q])*cirq.Y.on(minimizer.qubits[(q+1)%len(minimizer.qubits)]))
            observable.append(float(J)*cirq.Z.on(minimizer.qubits[q])*cirq.Z.on(minimizer.qubits[(q+1)%len(minimizer.qubits)]))
        return observable
    elif hamiltonian.upper() == "Z":
        return [cirq.Z.on(q) for q in minimizer.qubits]
    else:
        raise NotImplementedError("Hamiltonian not implemented yet")

def compute_lower_bound_cost_vqe(minimizer):
    if minimizer.noisy == True:
        return -np.inf
    else:
        return np.real(np.min(np.linalg.eigvals(sum(minimizer.observable).matrix())))

def prepare_optimization_vqe(translator, circuit_db):
    trainable_symbols = database.get_trainable_symbols(translator,circuit_db)
    trainable_param_values = database.get_trainable_params_value(translator,circuit_db)
    return trainable_symbols, trainable_param_values


class NoisyExpectation(tf.keras.layers.Layer):
    def __init__(self, units, observable, symbols):#, init_params=None):
        super(NoisyExpectation, self).__init__()
        self.w = self.add_weight(
            shape=(1,units), initializer="random_normal", trainable=True
        )
        # if not (init_params is None):
        #    self.trainable_variables.assign(tf.Variable(tf.convert_to_tensor(initial_params.astype(np.float32))))

        self.samples = np.array([[1000]*len(observable)])
        self.symbols = tf.convert_to_tensor(symbols)
        self.observable = tfq.convert_to_tensor([observable])
        self.lay = tfq.differentiators.ForwardDifference().generate_differentiable_op(sampled_op = tfq.noise.expectation)

    def call(self, inputs,operators=None, symbol_names=None):
        return self.lay( inputs,  self.symbols, self.trainable_variables[0], self.observable, self.samples)


class QNN_VQE(tf.keras.Model):
    def __init__(self, symbols, observable, batch_sizes=1,**kwargs):
        """
        symbols: symbolic variable [sympy.Symbol]*len(rotations_in_circuit)
        batch_size: how many circuits you feed the model at, at each call (this might )
        """
        super(QNN_VQE,self).__init__()
        if kwargs.get("noisy",False) == True:
            self.expectation_layer = NoisyExpectation(len(symbols), observable, symbols)
        else:
            self.expectation_layer = tfq.layers.Expectation()
        self.symbols = symbols
        self.observable = tfq.convert_to_tensor([observable]*batch_sizes)
        self.cost_value = Metrica(name="cost")
        self.lr_value = Metrica(name="lr")
        self.gradient_norm = Metrica(name="grad_norm")


    def call(self, inputs):
        """
        inputs: tfq circuits (resolved or not, to train one has to feed unresolved).
        """
        feat = inputs
        f = self.expectation_layer(feat, operators=self.observable, symbol_names=self.symbols)
        f = tf.math.reduce_sum(f,axis=-1)
        return f

    def train_step(self, data):
        x,y=data
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            preds = self(x,training=True)
            cost = self.compiled_loss(preds, preds) #notice that compiled loss takes care only about the preds
        train_vars = self.trainable_variables
        grads=tape.gradient(cost,train_vars)
        self.gradient_norm.update_state(tf.reduce_sum(tf.pow(grads[0],2)))

        if self.optimizer.get_config()["name"] == "SGD":
            self.qacq_gradients(cost, grads, x)
        else:
            self.optimizer.apply_gradients(zip(grads,train_vars))

        self.cost_value.update_state(cost)
        self.lr_value.update_state(self.optimizer.lr)

        return {k.name:k.variables[0] for k in self.metrics}

    def qacq_gradients(self, cost, grads, x):
        """
        Algorithm 4 of https://arxiv.org/pdf/1807.00800.pdf
        """
        g=tf.reduce_sum(tf.pow(grads[0],2))
        initial_lr = tf.identity(self.optimizer.lr)
        initial_params = tf.identity(self.trainable_variables)

        #compute line 10
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        alpha1 = tf.identity(self.trainable_variables)
        preds1 = self(x)
        cost1 = self.compiled_loss(preds1,preds1)

        #compute line 11
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        alpha2 = tf.identity(self.trainable_variables)
        preds2=self(x)
        cost2 = self.compiled_loss(preds2,preds2)

        self.condi(tf.math.greater_equal(cost - cost2, initial_lr*g),tf.math.greater_equal(initial_lr*g/2,cost - cost1), initial_lr,alpha1, alpha2)
        return

    @tf.function
    def condi(self,var1, var2, initial_lr, alpha1, alpha2):
        if var1 == True:
            self.optimizer.lr.assign(2*initial_lr)
            self.trainable_variables[0].assign(alpha2[0])
        else:
            if var2 == True:
                #self.optimizer.lr.assign(tf.reduce_max([1e-4,initial_lr/2]))
                self.optimizer.lr.assign(initial_lr/2)
                self.trainable_variables[0].assign(alpha1[0])
            else:
                self.trainable_variables[0].assign(alpha1[0])

    @property
    def metrics(self):
        return [self.cost_value, self.lr_value,self.gradient_norm]

class EnergyLoss(tf.keras.losses.Loss):
    def __init__(self, mode_var="vqe", **kwargs):
        super(EnergyLoss,self).__init__()
        self.mode_var = mode_var

    def call(self, y_true, y_pred):
        return tf.math.reduce_sum(y_pred,axis=-1)


class Metrica(tf.keras.metrics.Metric):
    def __init__(self, name):
        super(Metrica, self).__init__()
        self._name=name
        self.metric_variable = self.add_weight(name=name, initializer='zeros')

    def update_state(self, new_value, sample_weight=None):
        self.metric_variable.assign(new_value)

    def result(self):
        return self.metric_variable

    def reset_state(self):
        self.metric_variable.assign(self.metric_variable)
