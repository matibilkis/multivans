#!/bin/bash

eta1=$1
eta2=$2

cd ~/multivans
. ~/qenv_bilkis/bin/activate
python3 main.py --eta1 $eta1 --eta2 $eta2
deactivate