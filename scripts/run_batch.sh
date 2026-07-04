#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh; conda activate rl312
export PYTHONPATH=$HOME/projects/casp_max/baselines:$HOME/projects/casp_max/codes
cd $HOME/projects/casp_max
TAG=$1; shift
for e in "$@"; do
  echo "=== $e start $(date +%H:%M:%S) ==="
  python codes/run_exp.py $e > outputs/run/_log_$e.txt 2>&1
  echo "=== $e end $(date +%H:%M:%S) ==="
done
echo ALLDONE > outputs/run/_done_$TAG.txt
