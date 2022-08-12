import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import sys
sys.path.insert(0, os.getcwd())
import tensorflow as tf
import tensorflow_quantum as tfq
from importlib import reload
import utilities.translator.tfq_translator as tfq_translator
import utilities.variational.tfq.variational as minimizer
import utilities.variational.tfq.vqe as vqe
import matplotlib.pyplot as plt
import numpy as np
import utilities.database.database as database
import cirq

reload(tfq_translator)
reload(minimizer)
reload(vqe)


resolver = database.give_resolver(translator,cdb)
resol = cirq.resolve_parameters(cc,resolver)

resol.unitary()

translator = tfq_translator.TFQTranslator(n_qubits=10)
tfq_minimizer = minimizer.Minimizer(translator,mode="VQE",hamiltonian="XXZ",params=[1.,.01])
db = translator.initialize(mode='u2')
cc, cdb = tfq_minimizer.translator.give_circuit(db)
tt = tfq_minimizer.variational(db)

aa = translator.give_circuit(cdb, resolved=True)
aa[0]



cost_ev = tt[1][-1].history["cost"]
plt.plot(cost_ev)
plt.plot(np.ones(len(cost_ev))*tfq_minimizer.lower_bound_cost)
