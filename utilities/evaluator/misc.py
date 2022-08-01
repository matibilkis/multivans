import numpy as np
from datetime import datetime
from functools import wraps
import errno
import os
import signal
from ast import literal_eval
import pandas as pd


def get_def_path():
    import getpass
    user = getpass.getuser()
    if (user == "cooper-cooper") or (user == "mati"):
        defpath = '../data-vans/'
    else:
        defpath = "/data/uab-giq/scratch/matias/data-vans/"
    return defpath
