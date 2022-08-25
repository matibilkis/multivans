#!/bin/bash
itraj=$1
cd ~/multivans
. ~/vans_env/bin/activate
START=$(date +%s.%N)
python3 running/tfq/noisy/mp_run.py --itraj $itraj
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF
deactivate
