import tensorflow as tf
import numpy as np
import time
from utilities.compiling import *
from utilities.vqe import *
from utilities.discrimination import *


"""
To do:
* check constrianed optimization for the angles
* minimizer for hamiltonian
* check TF precision, float64... the limits are 1e-16 (i suspect it's because you split real and imaginary, but unsure)
"""

class Minimizer:
    def __init__(self,
                translator,
                mode,
                lr=0.01,
                optimizer="adam",
                epochs=1000,
                patience=200,
                max_time_continuous=120,
                parameter_noise=0.01,
                n_qubits = 2,
                **kwargs):

            ## training hyperparameters
            self.lr = lr
            self.translator = translator
            self.epochs = epochs
            self.patience = patience
            self.max_time_training = max_time_continuous
            self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)
            self.parameter_noise = parameter_noise
            self.minimization_step=0 #used for tensorboard
            self.mode = mode

            if mode.upper() == "VQE":
                hamiltonian = kwargs.get("hamiltonian")
                params = kwargs.get("params")
                self.observable = give_observable_vqe(translator,hamiltonian, params)
                self.loss = EnergyLoss()
                self.model_class = QNN_VQEetas
                self.lower_bound_cost = compute_lower_bound_cost_vqe(self) ## this will only work
                self.target_preds = None ##this is to compute the cost

            elif mode.upper() == "DISCRIMINATION":

                params = kwargs.get("params",[1., 0.01])
                number_hyp = kwargs.get("number_hyp",2)
                self.params = params

                self.observable = [cirq.Z.on(q) for q in translator.qubits]
                self.loss = PerrLoss(discard_qubits=translator.discard_qubits, number_hyp = number_hyp)
                self.model_class = QNN_DISCRIMINATION
                self.lower_bound_cost = compute_lower_bound_discrimination(params)

                self.target_preds = None ##this is to compute the cost

            elif mode.upper() == "COMPILING":

                self.observable = give_observable_compiling(translator)
                self.loss = CompilingLoss(d = translator.n_qubits)
                self.model_class = QNN_Compiling
                self.lower_bound_cost = compute_lower_bound_cost_compiling(self) ## this will only work
                self.target_preds = None ##this is to compute the cost
                self.patience = 50 #don't wait too much

    def give_cost_external_model(self, batched_circuit, model):
        return self.loss(*[model(batched_circuit)]*2) ###useful for unitary killer


    def give_cost(self, circuit_db):
        ### example: minimizer.give_cost(  [translator.give_circuit(circuit_db)[0]], resolver )
        if not hasattr(self,"model"):
            raise AttributeError("give me a model!")

        if self.mode.upper() == "DISCRIMINATION":
            batch_circuits, trainable_symbols, trainable_params_value = prepare_optimization_discrimination(self.translator, circuit_db, self.params, unresolved=False)
            return self.loss(*[self.model(batch_circuits)]*2)


    def variational(self, circuit_db):
        """
        proxy for minimize
        """
        #circuit, circuit_db = self.translator.give_circuit(circuit_db)
        if self.mode.upper() == "DISCRIMINATION":
            batch_circuits, trainable_symbols, trainable_params_value = prepare_optimization_discrimination(self.translator, circuit_db, self.params)
            cost, resolver, training_history = self.minimize(batch_circuits, symbols = trainable_symbols, parameter_values = trainable_params_value )
            optimized_circuit_db = self.translator.update_circuit_db_param_values(circuit_db, resolver)
            return optimized_circuit_db, [cost, resolver, training_history]


    def minimize(self, batched_circuits, symbols, parameter_values=None, parameter_perturbation_wall=1):
        """
        batched_circuits:: list of cirq.Circuits (should NOT be resolved or with Sympy.Symbol)
        symbols:: list of strings containing symbols for each rotation
        parameter_values:: values of previously optimized parameters
        parameter_perturbation_wall:: with some probability move away from the previously optimized parameters (different initial condition)
        """
        batch_size = len(batched_circuits)
        self.model = self.model_class(symbols=symbols, observable=self.observable, batch_sizes=batch_size)

        tfqcircuit = tfq.convert_to_tensor(batched_circuits)
        self.model(tfqcircuit) #this defines the weigths
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        #in case we have already travelled the parameter space,
        if parameter_values is not None:
            self.model.trainable_variables[0].assign(tf.convert_to_tensor(parameter_values.astype(np.float32)))
        else:
            self.model.trainable_variables[0].assign(tf.convert_to_tensor(np.pi*4*np.random.randn(len(symbols)).astype(np.float32)))

        if np.random.uniform() < parameter_perturbation_wall:
            perturbation_strength = abs(np.random.normal(scale=np.max(np.abs(self.model.trainable_variables[0]))))
            self.model.trainable_variables[0].assign(self.model.trainable_variables[0] + tf.convert_to_tensor(perturbation_strength*np.random.randn(len(symbols)).astype(np.float32)))

        calls=[tf.keras.callbacks.EarlyStopping(monitor='cost', patience=self.patience, mode="min", min_delta=0),TimedStopping(seconds=self.max_time_training)]

        if hasattr(self, "tensorboarddata"):
            self.minimization_step+=1 #this is because we call the module many times !
            calls.append(tf.keras.callbacks.TensorBoard(log_dir=self.tensorboarddata+"/logs/{}".format(self.minimization_step)))

        training_history = self.model.fit(x=tfqcircuit, y=tf.zeros((batch_size,)),verbose=0, epochs=self.epochs, batch_size=batch_size, callbacks=calls)

        cost = self.model.cost_value.result()
        final_params = self.model.trainable_variables[0].numpy()
        resolver = {"th_"+str(ind):var  for ind,var in enumerate(final_params)}
        return cost, resolver, training_history



class TimedStopping(tf.keras.callbacks.Callback):
    '''Stop training when enough time has passed.
        # Arguments
        seconds: maximum time before stopping.
        verbose: verbosity mode.
    '''
    def __init__(self, seconds=None, verbose=1):
        super(TimedStopping, self).__init__()
        self.start_time = 0
        self.seconds = seconds
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        if time.time() - self.start_time > self.seconds:
            self.model.stop_training = True
            if self.verbose>0:
                print('Stopping after %s seconds.' % self.seconds)


def prepare_circuit_vqe(circuit_db):
    trainable_symbols = translator.get_trainable_symbols(circuit_db)
    trainable_param_values = translator.get_trainable_params_value(circuit_db)
    return trainable_symbols, trainable_param_values
