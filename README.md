#installation instructions
##TFQ ---> use https://github.com/tensorflow/quantum/blob/master/docs/install.md
###works OK python3.8 3.9
datetime
tensorflow==2.7.0
pennylane
pennylane-forest
tensorflow-quantum==0.7.2





####import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
sys.version

import tensorflow as tf
import tensorflow_quantum as tfq
tf.__version__

tfq.__version__
import cirq
cirq.__version__
