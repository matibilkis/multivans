
import os
import sys
sys.path.insert(0, os.getcwd())
from importlib import reload
import tensorflow as tf
import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
from tqdm import tqdm
import utilities.translator.pennylane_translator as penny_translator
import utilities.evaluator.pennylane_evaluator as penny_evaluator
import utilities.variational.pennylane_model as penny_variational
import utilities.simplification.simplifier as penny_simplifier
import utilities.simplification.misc as simplification_misc#.kill_and_simplify
import utilities.simplification.gate_killer as penny_killer
import utilities.database.database as database
import utilities.database.templates as templates
import utilities.mutator.idinserter as idinserter
import running.misc as miscrun
from utilities.simplification.misc import get_qubits_involved, reindex_symbol, shift_symbols_down, shift_symbols_up, type_get, check_rot, order_symbol_labels, check_cnot, check_symbols_ordered
import coding.coding_simplifier as simplli



reload(penny_simplifier)
reload(templates)
reload(penny_translator)
reload(miscrun)
reload(idinserter)
reload(penny_variational)

n_qubits = 4

translator = penny_translator.PennyLaneTranslator(n_qubits = n_qubits, initialize="x")
simplifier = simplli.PennyLane_Simplifier(translator)
db = database.concatenate_dbs([templates.x_layer(translator, params=True),templates.z_layer(translator, params=True) ]*2)
c, cdb = translator.give_circuit(db)
simplified_db = cdb.copy()

def give_matrix():
    params = np.random.randn(8)*2*np.pi
    gates = [qml.RZ, qml.RX]*4

    def circu(params, gates):
        for p,g in zip(params,gates):
            g(p,wires=0)
        return [qml.PauliZ(wires=k) for k in range(1)]
    ori_u=qml.matrix(circu)(params,gates)
    return ori_u
c = np.array([[ 1-1j, -1-1j],[1-1j, 1+1j]])*.5
cdag = np.conjugate(c.T)



def zyz(uu):
    def qfu(uu):
        qml.QubitUnitary(uu,wires=0)
        return qml.expval(qml.PauliZ(0))

    dev = qml.device("default.qubit",wires=1)
    qnode = qml.QNode(qfu, dev)

    transformed_qfunc = qml.transforms.unitary_to_rot(qfu)
    qq = qml.QNode(transformed_qfunc, dev)
    qq(uu)
    rot = qq.tape.circuit[0]
    return rot

U = give_matrix()
deco , [delta, alpha, th, beta] = u2zyz(U)
np.linalg.det(U)

U*np.exp(1j*)




def u2zyz(U):
    """
    U = e^i \delta RZ(\alpha) Ry(\theta) Rz(\beta) =
    [[cos(th/2)e^{i (delta + alpha/2 + beta/2)}, sin(th/2)e^{i(delta + alpha/2 - \beta/2)}],[-sin(th/2)e^{delta - alpha/2 + beta/2}, cos(th/2)e^{\delta - \alpha/2 - \beta/2}]]
    """
    th = 2*np.arccos(np.abs(U[0,0]))
    beta = np.angle(U[0,0]) - np.angle(U[0,1])
    delta = .5*(np.angle(U[0,0]) + np.angle(U[1,1]))

    alpha = np.angle(U[0,1]) - np.angle(U[1,1])# + f

    rz_alpha = np.diag([np.exp(1j*alpha/2), np.exp(-1j*alpha/2)])
    rz_beta = np.diag([np.exp(1j*beta/2), np.exp(-1j*beta/2)])
    ry_th = np.array([[np.cos(th/2), np.sin(th/2)],[-np.sin(th/2), np.cos(th/2)]])
    r = np.exp(1j*delta)*rz_alpha.dot(ry_th).dot(rz_beta)
    #r = rz_alpha.dot(ry_th).dot(rz_beta)
    return r, [delta, alpha, th, beta]

def u2zxz(U):
    """
    U = e^i \delta RZ(\alpha) RX(\theta) Rz(\beta)
    returns U (decomposed as such, to check) and [\delta, \alpha, \theta, \beta].
    note we just change of basis and apply zyz decomposition
    """
    s = np.diag([1,1j])
    ou = (s.conj().T).dot(U).dot(s)
    _,[delta, alpha, th, beta]=u2zyz(ou)

    rz_alpha = np.diag([np.exp(1j*alpha/2), np.exp(-1j*alpha/2)])
    rz_beta = np.diag([np.exp(1j*beta/2), np.exp(-1j*beta/2)])
    rx_th = np.array([[np.cos(th/2), -1j*np.sin(th/2)],[-1j*np.sin(th/2), np.cos(th/2)]])
    r = rz_alpha.dot(rx_th).dot(rz_beta)

    return r,[delta, alpha, th, beta]

u2zxz(U)
1j*U


np.exp(1j*np.pi)
np.exp(-1j*np.pi)



import scipy.stats as r
p=[]
for k in range(int(1e4)):
    uu = r.unitary_group.rvs(2)
    p.append(np.max(np.abs(u2zyz(uu)[0] - uu)))

np.max(p)




p=[]
for k in range(int(1e4)):
    uu = r.unitary_group.rvs(2)
    p.append(np.max(np.abs(u2zxz(uu) - uu)))

np.max(p)




u_ori
uu = 1j*qml.matrix(zyz(u_ori))
uu

change = np.array([[1j,0],[0,-1j]])
cc = 1j*np.eye(2)
changed = cc.dot(u_ori)
uu = qml.matrix(zyz(u_ori))

uu
1j*np.linalg.inv(u_ori).T

uu = qml.matrix(zyz(1j*u_ori))

uu - 1j*u_ori

zyz(uu)

zyz(1j*u_ori)

zyz(qml.matrix(zyz(1j*u_ori)))

#####
def zyzma(pp):
    for p,g in zip(pp,[qml.RZ, qml.RY, qml.RZ]):
        g(p,wires=0)
    return [qml.PauliZ(wires=k) for k in range(1)]#qml.expval(qml.PauliZ(0))#[qml.PauliZ(wires=k) for k in range(1)]

qml.matrix(zyzma)((zyz(qml.matrix(zyz(1j*u_ori))).parameters))

qml.matrix(zyz(1j*u_ori))







zyz(u_ori).parameters

def xzx(pp):
    for p,g in zip(pp,[qml.RX, qml.RZ, qml.RX]):
        g(p,wires=0)
    return [qml.PauliZ(wires=k) for k in range(1)]#qml.expval(qml.PauliZ(0))#[qml.PauliZ(wires=k) for k in range(1)]

zyz(uu.)

zyz(u_ori).parameters
xzx_rot_comp =qml.matrix(xzx)(zyz(u_ori).parameters)

xzx_rot_comp






1j*c.dot(mm.dot(cdag))
c.dot(u_ori).dot(cdag)
xzx_rot_comp

u_ori







ori_u
dev = qml.device("default.qubit",wires=1)
qxx = qml.QNode(xzx, dev)
qxx(rot.parameters)
xzx_rotation = qml.matrix(qxx.tape.circuit[0])

xzx_rotation
compiled = qml.matrix(rot)

compiled -u_matrix







rot = qq.tape.circuit[0]

compiled








def give_zyz(params,gates):
    def circu(params, gates):
        for p,g in zip(params,gates):
            g(p,wires=0)
        return [qml.PauliZ(wires=k) for k in range(1)]

    global u_matrix
    u_matrix = qml.matrix(circu)(params,gates)
    def qfu():
        qml.QubitUnitary(u_matrix,wires=0)
        return qml.expval(qml.PauliZ(0))

    dev = qml.device("default.qubit",wires=1)
    qnode = qml.QNode(qfu, dev)

    transformed_qfunc = qml.transforms.unitary_to_rot(qfu)
    qq = qml.QNode(transformed_qfunc, dev)
    qq()
    rot = qq.tape.circuit[0]
    return rot

params = np.random.randn(4)*2*np.pi
gates = [qml.RZ, qml.RX]*2
give_zyz(params,gates)

rot.compute_decomposition()

rot.decomposition()

qml.tape.get_active_tape(

help(qml.transforms.unitary_to_rot)
aa = qq.tape

qml.tape.get_active_tape(aa)

aa = qml.tape.get_active_tape(transformed_qfunc.tape)















params = np.random.randn(4)*2*np.pi
gates = [qml.RZ, qml.RX]*2
def circu(params):
    for p,g in zip(params,gates):
        g(p,wires=0)
    return [qml.PauliZ(wires=k) for k in range(1)]






qml.matrix(circu)()

params = np.random.randn(4)*2*np.pi
gates = [qml.RZ, qml.RX]*2
def circu(params):
    for p,g in zip(params,[qml.RZ, qml.RX]*2):
        g(p,wires=0)
    return [qml.expval(qml.PauliZ(wires=k)) for k in range(1)]

dev = qml.device("default.qubit",wires=1)
qnode = qml.QNode(circu,dev)

print(qml.draw(qnode)(params))

opti = qml.transforms.merge_rotations(atol=1e-6)(circu)
qnode1 = qml.QNode(opti, dev)
print(qml.draw(qnode1)(params))


u = qml.matrix(circu)(params)


def qfu():
    qml.QubitUnitary(u,wires=0)
    return qml.expval(qml.PauliZ(0))


dev = qml.device("default.qubit",wires=1)
qnode = qml.QNode(qfu, dev)

print(qml.draw(qnode, show_matrices=True)())
transformed_qfunc = qml.transforms.unitary_to_rot(qfu)


qml.drawer.tape_mpl(qnode.tape)
qml.tape.get_active_tape(qnode.tape)



qq = qml.QNode(transformed_qfunc, dev)

qq.construct(qfu,)



gates_on_qubit, on_qubit_order = simplifier.get_positional_dbs(_, cdb)
simplification = False
for q, qubit_gates_path in gates_on_qubit.items():
    if simplification is True:
        break
    for order_gate_on_qubit, ind_gate in enumerate(qubit_gates_path[:-2]):

        if simplification is True:
            break
        ind_gate_p1 = qubit_gates_path[order_gate_on_qubit+1]
        ind_gate_p2 = qubit_gates_path[order_gate_on_qubit+2]

        if (check_rot(ind_gate, simplifier.translator) == True) and (check_rot(ind_gate_p1, simplifier.translator) == True) and (check_rot(ind_gate_p2, simplifier.translator) == True):

            type_0 = type_get(ind_gate,simplifier.translator)
            type_1 = type_get(ind_gate_p1,simplifier.translator)
            type_2 = type_get(ind_gate_p2,simplifier.translator)

            if type_0 == type_2:
                types = [type_0, type_1, type_2]
                for next_order_gate_on_qubit, ind_gate_next in enumerate(qubit_gates_path[order_gate_on_qubit+3:]):
                    if (check_rot(ind_gate_next, simplifier.translator) == True):# and (next_order_gate_on_qubit < len(qubit_gates_path[order_gate_on_qubit+3:])):
                        types.append(type_get(ind_gate_next, simplifier.translator))
                        simplification=True
                    else:
                        break
                if simplification == True:
                    break


indices_to_compile = [on_qubit_order[q][order_gate_on_qubit+k] for k in range(len(types))]
simplifier.translator_ =  penny_translator.PennyLaneTranslator(n_qubits = 2)

u_to_compile_db = simplified_db.loc[indices_to_compile]
u_to_compile_db["ind"] = simplifier.translator_.n_qubits*type_get(u_to_compile_db["ind"], simplifier.translator) + simplifier.translator_.number_of_cnots
u_to_compile_db["symbol"] = None ##just to be sure it makes no interference with the compiler...
cir, cirdb = simplifier.translator_.give_circuit(u_to_compile_db)

compile_circuit_db = penny_simplifier.construct_compiling_circuit(simplifier.translator_, u_to_compile_db)


c, cdb = simplifier.translator_.give_circuit(compile_circuit_db)

minimizer = penny_variational.PennyModel(simplifier.translator_,lr=0.1, patience=50, hamiltonian="Z")
minimized_db, [cost, resolver, history_training] = minimizer.variational(epochs=100, verbose = 1)


resolver

OneQbit_translator = penny_translator.PennyLaneTranslator(n_qubits=1)
u1s = templates.u1_db(OneQbit_translator, 0, params=True)
u1s["param_value"] = -np.array(list(resolver.values()))
resu_comp, resu_db = OneQbit_translator.give_circuit(u1s)


tdb = simplified_db.loc[indices_to_compile]
tdb["ind"] = simplifier.translator_.n_qubits*type_get(tdb["ind"], simplifier.translator) + OneQbit_translator.number_of_cnots
tdb["symbol"] = None

target_c, target_db = OneQbit_translator.give_circuit(tdb)



resu_comp.device.state
target_c.device.state



u_to_compile_db_1q = u_to_compile_db.copy()
u_to_compile_db_1q["ind"] = u_to_compile_db["ind"] = type_get(u_to_compile_db["ind"], self.translator_)




plt.plot(history_training.history["cost"])







minimizer = Minimizer(self.translator_, mode="compiling", hamiltonian="Z")

cost, resolver, history = minimizer.minimize([compile_circuit], symbols=self.translator.get_trainable_symbols(compile_circuit_db))

OneQbit_translator = CirqTranslater(n_qubits=1)
u1s = u1_db(OneQbit_translator, 0, params=True)
u1s["param_value"] = -np.array(list(resolver.values()))
resu_comp, resu_db = OneQbit_translator.give_circuit(u1s,unresolved=False)


u_to_compile_db_1q = u_to_compile_db.copy()
u_to_compile_db_1q["ind"] = u_to_compile_db["ind"] = type_get(u_to_compile_db["ind"], self.translator_)

cc, cdb = OneQbit_translator.give_circuit(u_to_compile_db_1q, unresolved=False)
c = cc.unitary()
r = resu_comp.unitary()

## phase_shift if necessary
if np.abs(np.mean(c/r) -1) > 1:
    u1s.loc[0] = u1s.loc[0].replace(to_replace=u1s["param_value"][0], value=u1s["param_value"][0] + 2*np.pi)# Rz(\th) = e^{-ii \theta \sigma_z / 2}c0, cdb0 = self.translator.give_circuit(pd.DataFrame([gate_template(0, param_value=2*np.pi)]), unresolved=False)
resu_comp, resu_db = self.translator.give_circuit(u1s,unresolved=False)

first_symbols = simplified_db["symbol"][indices_to_compile][:3]

for new_ind, typ, pval in zip(indices_to_compile[:3],[0,1,0], list(u1s["param_value"])):
    simplified_db.loc[new_ind+0.1] = gate_template(self.translator.number_of_cnots + q + typ*self.translator.n_qubits,
                                                     param_value=pval, block_id=simplified_db.loc[new_ind]["block_id"],
                                                     trainable=True, symbol=first_symbols[new_ind])

for old_inds in indices_to_compile:
    simplified_db = simplified_db.drop(labels=[old_inds],axis=0)#

simplified_db = simplified_db.sort_index().reset_index(drop=True)
killed_indices = indices_to_compile[3:]
db_follows = original_db[original_db.index>indices_to_compile[-1]]


simplified_db = self.order_symbols(simplified_db)
# if len(db_follows)>0:
#     gates_to_lower = list(db_follows.index)
#     number_of_shifts = len(killed_indices)
#     for k in range(number_of_shifts):
#         simplified_db = shift_symbols_down(self.translator, gates_to_lower[0]-number_of_shifts, simplified_db)


























indices_to_compile = [on_qubit_order[q][order_gate_on_qubit+k] for k in range(len(types))]
self.translator_ = PennyLaner(n_qubits=2)
u_to_compile_db = simplified_db.loc[indices_to_compile]
u_to_compile_db["ind"] = self.translator_.n_qubits*type_get(u_to_compile_db["ind"], self.translator) + self.translator_.number_of_cnots
u_to_compile_db["symbol"] = None ##just to be sure it makes no interference with the compiler...

compile_circuit, compile_circuit_db = construct_compiling_circuit(self.translator_, u_to_compile_db)
minimizer = Minimizer(self.translator_, mode="compiling", hamiltonian="Z")

cost, resolver, history = minimizer.minimize([compile_circuit], symbols=self.translator.get_trainable_symbols(compile_circuit_db))

OneQbit_translator = CirqTranslater(n_qubits=1)
u1s = u1_db(OneQbit_translator, 0, params=True)
u1s["param_value"] = -np.array(list(resolver.values()))
resu_comp, resu_db = OneQbit_translator.give_circuit(u1s,unresolved=False)


u_to_compile_db_1q = u_to_compile_db.copy()
u_to_compile_db_1q["ind"] = u_to_compile_db["ind"] = type_get(u_to_compile_db["ind"], self.translator_)

cc, cdb = OneQbit_translator.give_circuit(u_to_compile_db_1q, unresolved=False)
c = cc.unitary()
r = resu_comp.unitary()

## phase_shift if necessary
if np.abs(np.mean(c/r) -1) > 1:
    u1s.loc[0] = u1s.loc[0].replace(to_replace=u1s["param_value"][0], value=u1s["param_value"][0] + 2*np.pi)# Rz(\th) = e^{-ii \theta \sigma_z / 2}c0, cdb0 = self.translator.give_circuit(pd.DataFrame([gate_template(0, param_value=2*np.pi)]), unresolved=False)
resu_comp, resu_db = self.translator.give_circuit(u1s,unresolved=False)

first_symbols = simplified_db["symbol"][indices_to_compile][:3]

for new_ind, typ, pval in zip(indices_to_compile[:3],[0,1,0], list(u1s["param_value"])):
    simplified_db.loc[new_ind+0.1] = gate_template(self.translator.number_of_cnots + q + typ*self.translator.n_qubits,
                                                     param_value=pval, block_id=simplified_db.loc[new_ind]["block_id"],
                                                     trainable=True, symbol=first_symbols[new_ind])

for old_inds in indices_to_compile:
    simplified_db = simplified_db.drop(labels=[old_inds],axis=0)#

simplified_db = simplified_db.sort_index().reset_index(drop=True)
killed_indices = indices_to_compile[3:]
db_follows = original_db[original_db.index>indices_to_compile[-1]]


simplified_db = self.order_symbols(simplified_db)
