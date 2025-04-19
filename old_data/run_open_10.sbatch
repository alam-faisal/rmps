#!/bin/bash -l
#SBATCH --job-name=run_10
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=run_10.out
#SBATCH --error=run_10.err
#SBATCH --qos=long
conda activate qaravan
python rmps_purity.py 30 2 20000 open 10
