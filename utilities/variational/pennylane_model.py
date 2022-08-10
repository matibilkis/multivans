import tensorflow as tf
from pennylane import numpy as np
import pennylane as qml
import pandas as pd
from datetime import datetime
import utilities.database.database as database
import utilities.database.templates as templates
from utilities.variational.misc import *
import utilities.variational.pennylane_model as penny_variational



class PennyModel(tf.keras.Model):
    def __init__(self, translator,**kwargs):
        super(PennyModel,self).__init__()

        self.translator = translator
        self.shots = kwargs.get("shots",None)
        self.lr = kwargs.get("lr",1e-2)

        self.build_model()

        self.cost_value = Metrica(name="cost")
        self.lr_value = Metrica(name="lr")
        self.gradient_norm = Metrica(name="grad_norm")

        self.patience = kwargs.get("patience",20) #gradient descent iterations without any improvemente
        self.max_time_training = kwargs.get("max_time_training",30*self.translator.n_qubits) #max training time per minimization routine

        self.get_observable(**kwargs)
        self.get_ground()

    def get_ground(self):
        if self.translator.n_qubits > 14:
            print("Very big system, warning!")
        st = datetime.now()
        H=qml.Hamiltonian(list(self.h_coeffs),list(self.ops))
        hh = qml.utils.sparse_hamiltonian(H).toarray()
        self.translator.ground = np.min(np.linalg.eigvals(hh))
        delt = np.round((datetime.now()-st).seconds,4)
        #print("computed ground energy in {}sec".format(delt))
        return


    def get_observable(self,**kwargs):
        H = kwargs.get("hamiltonian", "XXZ")
        print(H)
        if H.upper() == "XXZ":
            g = kwargs.get("g", 1.)
            J = kwargs.get("J", 1.)

            self.obs = [qml.PauliZ(k) for k in range(self.translator.n_qubits)]
            self.h_coeffs = [g for k in range(self.translator.n_qubits)]
            ## \sum_j Z_j Z_{j+1}
            self.obs += [qml.PauliZ(k%self.translator.n_qubits)@qml.PauliZ((k+1)%self.translator.n_qubits) for k in range(self.translator.n_qubits)]
            self.h_coeffs += [J for k in range(self.translator.n_qubits)]
            ## \sum_j X_j X_{j+1}
            self.obs += [qml.PauliX(k%self.translator.n_qubits)@qml.PauliX((k+1)%self.translator.n_qubits) for k in range(self.translator.n_qubits)]
            self.h_coeffs += [1. for k in range(self.translator.n_qubits)]
            ## \sum_j Y_j Y_{j+1}
            self.obs += [qml.PauliY(k%self.translator.n_qubits)@qml.PauliY((k+1)%self.translator.n_qubits) for k in range(self.translator.n_qubits)]
            self.h_coeffs += [1. for k in range(self.translator.n_qubits)]
            self.ops = self.obs.copy()
        elif H.upper() == "Z":
            self.obs = [qml.PauliZ(k) for k in range(self.translator.n_qubits)]
            self.h_coeffs = [1. for k in range(self.translator.n_qubits)]
            self.ops = self.obs.copy()

    def observable(self):
        return [qml.expval(k) for k in self.obs]#*self.h_coeffs

    def variational(self,**kwargs):
        self.build_model()

        if np.random.uniform() < kwargs.get("parameter_perturbation_wall", 1e-1):
            perturbation_strength = abs(np.random.normal(scale=np.max(np.abs(self.trainable_variables[0]))))
            self.trainable_variables[0].assign(self.trainable_variables[0] + tf.convert_to_tensor(perturbation_strength*np.random.randn(self.trainable_variables[0].shape[0]).astype(np.float32)))

        calls=[tf.keras.callbacks.EarlyStopping(monitor='cost', patience=self.patience, mode="min", min_delta=0),TimedStopping(seconds=self.max_time_training)]
        history = self.fit(x=[1.], y=[1.], verbose=kwargs.get("verbose", 0),epochs=kwargs.get("epochs",100), callbacks=calls)

        cost = self.give_cost(self.translator.db_train)
        self.translator.db_train = database.correct_param_value_dtype(self.translator,self.translator.db_train) ##this corrects the dtpye (from tensorflow to np.float32) of param_values

        symbols = database.get_trainable_symbols(self.translator,self.translator.db_train)
        resolver = {s:w for s,w in zip( symbols, self.trainable_variables[0].numpy())}

        return self.translator.db_train, [cost, resolver, history]

    def build_model(self):
        """
        build layer from translator.db_train
        """
        if not hasattr(self.translator, "db_train"):
            print("Caution: building model from uninitialized circuit, dummy circuit z-layer")
            _, _ = self.translator.give_circuit(templates.z_layer(self.translator))
        wshape = len(database.get_trainable_symbols(self.translator,self.translator.db_train))
        self.weight_shapes = {"weights": wshape}
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))

        self.dev = qml.device("default.qubit", wires=self.translator.n_qubits, shots=self.shots,)#.tf
        self.dev.R_DTYPE = np.float32

        @qml.qnode(self.dev)#, diff_method="adjoint")#, interface="tf",,)
        def qnode_keras(inputs, weights):
            """ I don't use inputs at all. Weights are trainable variables """
            cinputs = self.translator.db_train
            symbols = database.get_trainable_symbols(self.translator,cinputs)
            ww = {s:w for s,w in zip( symbols, weights)}
            cinputs = database.update_circuit_db_param_values(self.translator, cinputs, ww)
            list_of_gate_ids = [templates.gate_template(**dict(cinputs.iloc[k])) for k in range(len(cinputs))]
            db = {} ###nobody cares about db here
            for i,gate_id in enumerate(list_of_gate_ids):
                db = self.translator.append_to_circuit(db, gate_id)
            return self.observable()
        self.qlayer = qml.qnn.KerasLayer(qnode_keras, self.weight_shapes, output_dim=self.translator.n_qubits)

    @property
    def metrics(self):
        return [self.cost_value, self.lr_value,self.gradient_norm]

    def call(self, inputs):
        return self.qlayer([])

    def give_cost(self,data_base):
        return tf.cast(tf.math.reduce_sum(self.h_coeffs*self(data_base)),tf.dtypes.DType(1))   #note that this is ruled by self.observable as well

    def give_cost_external(self,data_base):
        cdb, db = self.translator.give_circuit(data_base)
        self.build_model()
        return tf.cast(tf.math.reduce_sum(self.h_coeffs*self(data_base)),tf.dtypes.DType(1))   #note that this is ruled by self.observable as well

    def train_step(self, data):
        #x,y = data
        x = self.translator.db_train
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            cost = self.give_cost(x)

        grads=tape.gradient(cost, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        gnorm = tf.reduce_sum(tf.pow(grads[0],2))

        self.gradient_norm.update_state(tf.cast(gnorm, tf.dtypes.DType(1)))
        self.cost_value.update_state(tf.cast(cost, tf.dtypes.DType(1)))
        self.lr_value.update_state(tf.cast(self.optimizer.lr, tf.dtypes.DType(1)))
        return {k.name:k.result() for k in self.metrics}
