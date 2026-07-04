#!/bin/bash
set -x
cd ~/projects/casp_max/data/benchmarks/steinlib
for c in D ES10FST I160 PUC WRP3; do
  curl -sL -m 120 -o ${c}.tgz "https://steinlib.zib.de/download/${c}.tgz"
  tar xzf ${c}.tgz 2>/dev/null && echo "OK $c"
done
find . -name "*.stp" | wc -l > ~/projects/casp_max/scripts/_steinlib_count.txt
# pyscipopt
source /opt/conda/etc/profile.d/conda.sh && conda activate rl312
pip install pyscipopt -i https://pypi.tuna.tsinghua.edu.cn/simple > ~/projects/casp_max/scripts/_pyscipopt.log 2>&1
python -c "import pyscipopt; print(\"pyscipopt\", pyscipopt.__version__)" >> ~/projects/casp_max/scripts/_pyscipopt.log 2>&1
echo DONE > ~/projects/casp_max/scripts/_prep_done.txt
