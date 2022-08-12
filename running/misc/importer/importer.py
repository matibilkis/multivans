import os
import sys
sys.path.insert(0, os.getcwd())


def import_vans():

    import utilities.translator.pennylane_translator as penny_translator
    import utilities.evaluator.pennylane_evaluator as penny_evaluator
    import utilities.variational.pennylane_model as penny_variational
    import utilities.simplification.simplifier as penny_simplifier
    import utilities.simplification.gate_killer as penny_killer
    import utilities.database.database as database
    import utilities.database.templates as templates
    import utilities.mutator.idinserter as idinserter


def reload_vans():
    reload(penny_evaluator)
    reload(penny_translator)
    reload(penny_variational)
    reload(templates)
    reload(database)
    reload(penny_simplifier)
    reload(penny_killer)
    reload(idinserter)
