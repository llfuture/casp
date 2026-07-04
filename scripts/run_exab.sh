#!/bin/bash
# CASP EX-A/EX-B full run (CPU only). Usage: bash run_all.sh [--mve]
set -e
cd "$(dirname "$0")/../codes"
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl312
python -c "import scipy, numpy, matplotlib" 2>/dev/null || \
  pip install scipy numpy matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
mkdir -p ../outputs ../figures

FLAG="--full"; [ "$1" = "--mve" ] && FLAG=""
echo "=== audit (generic vs fast path) ==="
python -u audit_exact.py
echo "=== EX-A margin (Thm 16) ==="
python -u exa_margin.py $FLAG
echo "=== EX-B lponly (Thm 17) ==="
python -u exb_lponly.py $FLAG
echo "=== figures ==="
python -u make_figs_ab.py
echo "ALL DONE $(date)"
