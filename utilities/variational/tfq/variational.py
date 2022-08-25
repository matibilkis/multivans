import tensorflow as tf
import numpy as np
import time
# from utilities.compiling import *
from utilities.variational.tfq.vqe import *
# from utilities.discrimination import *


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
                **kwargs):


            self.translator = translator
            self.mode = mode ##VQE, DISCRIMINATION, COMPILING... (latter not sure if we can implement it now, that's why I coded everyhting again in pennylane)
            ## training hyperparameters
            self.lr = kwargs.get("lr", 0.01)
            self.initial_lr = kwargs.get("lr", 0.01)
            self.epochs=kwargs.get("epochs",5000)
            self.patience = kwargs.get("patience",100)
            self.max_time_training = kwargs.get("max_time_training",300)
            #self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)
            self.optimizer = tf.keras.optimizers.SGD(learning_rate = self.lr)
            self.minimization_step=0 #used for tensorboard
            self.noisy = self.translator.noisy#kwargs.get("noisy",False)
            self.verbose = kwargs.get("verbose",0)

            if mode.upper() == "VQE":
                hamiltonian = kwargs.get("hamiltonian")
                params = kwargs.get("params")
                self.observable = give_observable_vqe(translator,hamiltonian, params)
                self.loss = EnergyLoss()
                self.model_class = QNN_VQE

                lower_bound_cost = kwargs.get("lower_bound_cost",-np.inf)
                if lower_bound_cost == -np.inf:
                    self.lower_bound_cost = compute_lower_bound_cost_vqe(self) ## this will only work
                else:
                    self.lower_bound_cost = lower_bound_cost

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
            else:
                raise Error("what about mode? {}",format(mode))
            who = kwargs.get("who","minimizer")
            if who == "minimizer":
                self.translator.ground = self.lower_bound_cost

    def build_and_give_cost(self,circuit_db):
        cc, cdb = self.translator.give_circuit(circuit_db, just_call=True)
        trainable_symbols, trainable_param_values = prepare_optimization_vqe(self.translator, cdb)

        model = self.model_class(symbols=trainable_symbols, observable=self.observable, batch_sizes=1, noisy=self.noisy)

        tfqcircuit = tfq.convert_to_tensor([cc])
        model(tfqcircuit) #this defines the weigths
        if self.noisy == True:
            trainable_param_values = np.array(trainable_param_values)[tf.newaxis]
        model.trainable_variables[0].assign(tf.convert_to_tensor(trainable_param_values.astype(np.float32)))

        model.compile(optimizer=self.optimizer, loss=self.loss)
        return self.loss(*[model(tfqcircuit)]*2)


    def give_cost_external_model(self, batched_circuit, model):
        """I think i don0't use this, TODO!"""
        return self.loss(*[model(batched_circuit)]*2) ###useful for unitary killer


    def give_cost(self, circuit_db):
        ### example: minimizer.give_cost(  [translator.give_circuit(circuit_db)[0]], resolver )
        if not hasattr(self,"model"):
            raise AttributeError("give me a model!")

        if self.mode.upper() == "DISCRIMINATION":
            batch_circuits, trainable_symbols, trainable_params_value = prepare_optimization_discrimination(self.translator, circuit_db, self.params, unresolved=False)
            return self.loss(*[self.model(batch_circuits)]*2)
        elif self.mode.upper() == "VQE":
            cc, cdb = self.translator.give_circuit(circuit_db, just_call=True)
            batched_circuit = [cc]
            trainable_symbols, trainable_param_values = prepare_optimization_vqe(self.translator, cdb)
            return self.loss(*[self.model(batched_circuit)]*2)

    def variational(self, circuit_db, **kwargs):
        """
        proxy for minimize
        """
        #circuit, circuit_db = self.translator.give_circuit(circuit_db)
        if self.mode.upper() == "DISCRIMINATION":
            batch_circuits, trainable_symbols, trainable_params_value = prepare_optimization_discrimination(self.translator, circuit_db, self.params)
            cost, resolver, training_history = self.minimize(batch_circuits, symbols = trainable_symbols, parameter_values = trainable_params_value )
            optimized_circuit_db = self.translator.update_circuit_db_param_values(circuit_db, resolver)
            return optimized_circuit_db, [cost, resolver, training_history]
        elif self.mode.upper() == "VQE":
            parameter_perturbation_wall = kwargs.get("parameter_perturbation_wall",1)
            cc, cdb = self.translator.give_circuit(circuit_db)
            batched_circuit = [cc]

            trainable_symbols, trainable_param_values = prepare_optimization_vqe(self.translator, cdb)
            cost, resolver, training_history = self.minimize(batched_circuit, symbols = trainable_symbols, parameter_values = trainable_param_values , parameter_perturbation_wall=parameter_perturbation_wall)

            optimized_circuit_db = database.update_circuit_db_param_values(self.translator,cdb, resolver)
            self.model.optimizer.lr.assign(self.initial_lr)

            return optimized_circuit_db, [cost, resolver, training_history]


    def minimize(self, batched_circuits, symbols, parameter_values=None, parameter_perturbation_wall=1):
        """
        batched_circuits:: list of cirq.Circuits (should NOT be resolved or with Sympy.Symbol)
        symbols:: list of strings containing symbols for each rotation
        parameter_values:: values of previously optimized parameters
        parameter_perturbation_wall:: with some probability move away from the previously optimized parameters (different initial condition)
        """
        batch_size = len(batched_circuits)
        self.model = self.model_class(symbols=symbols, observable=self.observable, batch_sizes=batch_size, noisy=self.noisy)

        tfqcircuit = tfq.convert_to_tensor(batched_circuits)
        self.model(tfqcircuit) #this defines the weigths
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        #in case we have already travelled the parameter space,
        if parameter_values is not None:
            if self.noisy == True:
                parameter_values = np.array(parameter_values)[tf.newaxis]
            self.model.trainable_variables[0].assign(tf.convert_to_tensor(parameter_values.astype(np.float32)))
        # else:
        #     self.model.trainable_variables[0].assign(tf.convert_to_tensor(np.pi*4*np.random.randn(len(symbols)).astype(np.float32)))

        self.model.cost_value.update_state(self.model.compiled_loss(*[self.model(tfqcircuit)]*2))
        if np.random.uniform() < parameter_perturbation_wall:
            perturbation_strength = abs(np.random.normal(scale=np.max(np.abs(self.model.trainable_variables[0]))))
            random_tensor = tf.random.uniform(self.model.trainable_variables[0].shape)*self.model.trainable_variables[0]/10
            self.model.trainable_variables[0].assign(self.model.trainable_variables[0] + random_tensor)

        calls=[SaveBestModel(),tf.keras.callbacks.EarlyStopping(monitor='cost', patience=self.patience, mode="min", min_delta=0, restore_best_weights=True),TimedStopping(seconds=self.max_time_training)]

        if hasattr(self, "tensorboarddata"):
            self.minimization_step+=1 #this is because we call the module many times !
            calls.append(tf.keras.callbacks.TensorBoard(log_dir=self.tensorboarddata+"/logs/{}".format(self.minimization_step)))

        training_history = self.model.fit(x=tfqcircuit, y=tf.zeros((batch_size,)),verbose=self.verbose, epochs=self.epochs, batch_size=batch_size, callbacks=calls)

        self.model.set_weights(calls[0].best_weights)
        cost = self.model.compiled_loss(*[self.model(tfqcircuit)]*2)
        # cost = self.model.cost_value.result()
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

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='cost'):
        self.save_best_metric = save_best_metric
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if metric_value < self.best:
            self.best = metric_value
            self.best_weights= self.model.get_weights()

def train_from_db(minimizer, prev_db, current_circuit, current_db, perturbation_strength=1e-3):
    """
    this function recycles weights obtained from training a smaller circuit (note there should be a sequential order.)
    This is used for training HEA with L layers, then using the weights to train HEA with L+1 layers
    """
    trainable_symbols, trainable_param_values = prepare_optimization_vqe(minimizer.translator, current_db)
    model = minimizer.model_class(symbols=trainable_symbols, observable=minimizer.observable, batch_sizes=1, noisy=minimizer.noisy)
    tfqcircuit = tfq.convert_to_tensor([current_circuit])
    model(tfqcircuit) #this defines the weigths
    model.compile(optimizer=minimizer.optimizer, loss=minimizer.loss)
    _, param_values_1 = prepare_optimization_vqe(minimizer.translator, prev_db)
    previous_l_and_random = np.array(list(param_values_1) + list(np.random.random(np.squeeze(model.trainable_variables[0]).shape[0]- len(param_values_1))))
    if minimizer.noisy == True:
        previous_l_and_random = np.array(previous_l_and_random)[tf.newaxis]
    model.trainable_variables[0].assign(tf.convert_to_tensor(previous_l_and_random.astype(np.float32)))
    ##now i slighlty perturbate
    random_tensor = tf.random.uniform(model.trainable_variables[0].shape)*model.trainable_variables[0]*perturbation_strength
    model.trainable_variables[0].assign(model.trainable_variables[0] + random_tensor)
    model.cost_value.update_state(model.compiled_loss(*[model(tfqcircuit)]*2))
    calls=[SaveBestModel(),tf.keras.callbacks.EarlyStopping(monitor='cost', patience=minimizer.patience, mode="min", min_delta=0, restore_best_weights=True),TimedStopping(seconds=minimizer.max_time_training)]
    training_history = model.fit(x=tfqcircuit, y=tf.zeros((1,)),verbose=minimizer.verbose, epochs=minimizer.epochs, batch_size=1, callbacks=calls)
    model.set_weights(calls[0].best_weights)
    cost = model.compiled_loss(*[model(tfqcircuit)]*2)
    final_params = model.trainable_variables[0].numpy()
    resolver = {"th_"+str(ind):var  for ind,var in enumerate(final_params)}
    optimized_circuit_db = database.update_circuit_db_param_values(minimizer.translator,current_db, resolver)
    return optimized_circuit_db, [cost, resolver, training_history]

#
