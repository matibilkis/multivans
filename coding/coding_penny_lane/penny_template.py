import tensorflow as tf
from pennylane import numpy as np
import pennylane as qml
import pandas as pd
import utilities.templates as templates
import utilities.database as database


class PennyLaneTranslator:
    def __init__(self, n_qubits, **kwargs):
        self.n_qubits = n_qubits
        #self.qubits = cirq.GridQubit.rect(1, n_qubits)

        ### blocks that are fixed-structure (i.e. channels, state_preparation, etc.)
        untouchable_blocks = kwargs.get("untouchable_blocks",[])
        self.untouchable_blocks = untouchable_blocks
        self.discard_qubits = kwargs.get("discard_qubits",[]) ###these are the qubits that you don't measure, i.e. environment


        #### keep a register on which integers corresponds to which CNOTS, target or control.
        self.indexed_cnots = {}
        self.cnots_index = {}
        count = 0
        for control in range(self.n_qubits):
            for target in range(self.n_qubits):
                if control != target:
                    self.indexed_cnots[str(count)] = [control, target]
                    self.cnots_index[str([control,target])] = count
                    count += 1
        self.number_of_cnots = len(self.indexed_cnots)
        #
        self.cgates = {0:qml.RZ, 1: qml.RX, 2:qml.RY}
        self.temp_circuit_db = {}

    def spit_gate(self,gate_id):
        ## the symbols are all elements we added but the very last one (added on the very previous line)
        ind = gate_id["ind"]

        if ind<self.number_of_cnots:
            qml.CNOT(self.indexed_cnots[str(ind)])
        else:
            cgate_type = (ind-self.number_of_cnots)%3
            qubit = (ind-self.number_of_cnots)//self.n_qubits
            symbol_name = gate_id["symbol"]
            param_value = gate_id["param_value"]
            self.cgates[cgate_type](param_value,qubit)

    def append_to_circuit(self,circuit_db, gate_id):
        ## the symbols are all elements we added but the very last one (added on the very previous line)
        symbols = []
        for j in [k["symbol"] for k in circuit_db.values()]:
            if j != None:
                symbols.append(j)
        ind = gate_id["ind"]

        gate_index = len(list(circuit_db.keys()))
        circuit_db[gate_index] = gate_id #this is the new item to add

        if ind<self.number_of_cnots:
            qml.CNOT(self.indexed_cnots[str(ind)])
        else:
            cgate_type = (ind-self.number_of_cnots)%3
            qubit = (ind-self.number_of_cnots)//self.n_qubits
            symbol_name = gate_id["symbol"]
            param_value = gate_id["param_value"]
            if symbol_name is None:
                symbol_name = "th_"+str(len(symbols))
                circuit_db[gate_index]["symbol"] = symbol_name
            else:
                if symbol_name in symbols:
                    print("Symbol repeated {}, {}".format(symbol_name, symbols))
            self.cgates[cgate_type](param_value,qubit)
        return circuit_db

    def give_circuit(self, dd,**kwargs):
        """

        """
        unresolved = kwargs.get("unresolved",True)

        dev = qml.device("default.qubit", wires=self.n_qubits)
        ### TO-DO: CHECK INPUT COPY!
        @qml.qnode(dev)
        def qnode(inputs, weights,**kwargs):
            """
            weights is a list of variables (automatic in penny-lane, here i feed [] so i don't update parameter values)
            """

#            db = kwargs.get("db",{})
            self.db = {}
            cinputs = inputs#.copy()
            symbols = database.get_trainable_symbols(self,cinputs)
            ww = {s:w for s,w in zip( symbols, weights)}
            cinputs = database.update_circuit_db_param_values(self, cinputs, ww)

            list_of_gate_ids = [templates.gate_template(**dict(cinputs.iloc[k])) for k in range(len(cinputs))]
            for i,gate_id in enumerate(list_of_gate_ids):
                self.db = self.append_to_circuit(self.db, gate_id)
            return [qml.expval(qml.PauliZ(k)) for k in range(self.n_qubits)]

        #self.db = {}
        circuit = qnode(dd, []) ##this creates the database; the weights are used as trainable variables # of optimization
        circuit_db = pd.DataFrame.from_dict(self.db,orient="index")
        self.db = circuit_db.copy()
        self.db_train = self.db.copy() ### copy to be used in PennyLaneModel
        return qnode, circuit_db







class modelito(tf.keras.Model):
    def __init__(self, qnode):
        super(modelito,self).__init__()

        weight_shapes = {"weights": qnode.train_params}
        self.qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=len(qnode.device.wires))

        self.cost_value = Metrica(name="cost")
        self.lr_value = Metrica(name="lr")
        self.gradient_norm = Metrica(name="grad_norm")

    @property
    def metrics(self):
        return [self.cost_value, self.lr_value,self.gradient_norm]

    def call(self, inputs):
        return self.qlayer(inputs)

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            cost = tf.math.reduce_sum(self(x))
        grads=tape.gradient(cost, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        gnorm = tf.reduce_sum(tf.pow(grads[0],2))#,)

        self.gradient_norm.update_state(tf.cast(gnorm, tf.dtypes.DType(1)))
        self.cost_value.update_state(tf.cast(cost, tf.dtypes.DType(1)))
        self.lr_value.update_state(tf.cast(self.optimizer.lr, tf.dtypes.DType(1)))
        return {k.name:k.result() for k in self.metrics}





class PennyModel(tf.keras.Model):
    def __init__(self, translator,**kwargs):
        super(PennyModel,self).__init__()

        self.translator = translator

        weight_shapes = {"weights": self.get_weights_shape()}
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=kwargs.get("lr",1e-2)))

        dev = qml.device("default.qubit", wires=self.translator.n_qubits, shots=kwargs.get("shots",None),)
        dev.R_DTYPE = np.float32
        @qml.qnode(dev)
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
            return [qml.expval(qml.PauliZ(k)) for k in range(translator.n_qubits)]

        self.qlayer = qml.qnn.KerasLayer(qnode_keras, weight_shapes, output_dim=self.translator.n_qubits)

        self.cost_value = Metrica(name="cost")
        self.lr_value = Metrica(name="lr")
        self.gradient_norm = Metrica(name="grad_norm")

    def get_weights_shape(self):
        return len(database.get_trainable_symbols(self.translator,self.translator.db_train))

    @property
    def metrics(self):
        return [self.cost_value, self.lr_value,self.gradient_norm]

    def call(self, inputs):
        return self.qlayer([])

    def train_step(self, data):
        #x,y = data
        x = self.translator.db_train
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            cost = tf.cast(tf.math.reduce_sum(self(x)),tf.dtypes.DType(1))
        
        grads=tape.gradient(cost, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        gnorm = tf.reduce_sum(tf.pow(grads[0],2))#,)

        self.gradient_norm.update_state(tf.cast(gnorm, tf.dtypes.DType(1)))
        self.cost_value.update_state(tf.cast(cost, tf.dtypes.DType(1)))
        self.lr_value.update_state(tf.cast(self.optimizer.lr, tf.dtypes.DType(1)))
        return {k.name:k.result() for k in self.metrics}






class Metrica(tf.keras.metrics.Metric):
    def __init__(self, name):
        super(Metrica, self).__init__()
        self._name=name
        self.metric_variable = self.add_weight(name=name, initializer='zeros', dtype=np.float32)

    def update_state(self, new_value, sample_weight=None):
        self.metric_variable.assign(new_value)

    def result(self):
        return self.metric_variable
    #
    # def reset_states(self):
    #     self.metric_variable.assign(0.)
