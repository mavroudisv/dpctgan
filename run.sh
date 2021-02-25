#$ -l h_rt=80:00:00
#$ -l tmem=40G
#$ -S /bin/bash
#$ -l gpu=true
#$ -j y
#$ -N SDGym_dpctgan




ulimit -v
export MPLBACKEND=agg

export PATH=/share/apps/python-3.7.2-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:$LD_LIBRARY_PATH

export PATH=/share/apps/cuda-11.0/bin:/usr/local/cuda-11.0/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/cuda-11.0/lib64:/usr/local/cuda-11.0/lib:${LD_LIBRARY_PATH}


echo "Execution begins..."
#/share/apps/python-3.7.2-shared/bin/python3 -m pip install opacus --user
/share/apps/python-3.7.2-shared/bin/python3 /SAN/infosec/TLS_fingerprinting/experiments/SDGym/benchmarks.py

