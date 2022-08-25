#!/bin/bash
hea=$1
cd ~/multivans
. ~/vans_env/bin/activate
START=$(date +%s.%N)
python3 running/tfq/mp_HEA.py --L_HEA $hea
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF
deactivate
