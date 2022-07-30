import numpy as np

def get_trainable_symbols(translator, circuit_db):
    trainable_symbols = circuit_db[circuit_db["trainable"] == True]["symbol"]
    return list(trainable_symbols[circuit_db["symbol"].notnull()])

def get_trainable_params_value(translator,circuit_db):
    index_trainable_params = circuit_db[circuit_db["trainable"] == True]["symbol"].dropna().index
    return circuit_db["param_value"][index_trainable_params]


def give_trainable_parameters(translator, circuit_db):
    indices =  circuit_db[circuit_db["trainable"] == True]["ind"]
    trainable_coefficients = indices[(indices < translator.number_of_cnots+ 3*translator.n_qubits) & (indices >= translator.number_of_cnots)]
    return len(trainable_coefficients)

def give_trainable_cnots(translator, circuit_db):
    indices =  circuit_db[circuit_db["trainable"] == True]["ind"]
    cnots = indices[(indices < translator.number_of_cnots)]
    return len(cnots)

def update_circuit_db_param_values(translator, circuit_db,symbol_to_value):
    """
    circuit_db (unoptimized) pd.DataFrame
    symbol_to_value: resolver, dict
    """
    trianables = circuit_db[circuit_db["trainable"] == True]
    trainable_symbols = trianables[~trianables["symbol"].isna()]
    if len(trainable_symbols) == len(symbol_to_value.values()):
        circuit_db["param_value"].update({ind:val for ind, val in zip(trainable_symbols.index, symbol_to_value.values())})
    return circuit_db

def give_resolver(translator, circuit_db):
    trianables = circuit_db[circuit_db["trainable"] == True]
    trainable_symbols = trianables[~trianables["symbol"].isna()]
    return dict(trainable_symbols[["symbol","param_value"]].values)

def correct_param_value_dtype(translator,db):
    resTF = give_resolver(translator, db)
    res = {k:vnp for k, vnp in zip(resTF.keys(), np.stack(resTF.values()))}
    db = update_circuit_db_param_values(translator,db,res)
    return db
