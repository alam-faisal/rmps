#!/bin/bash -l
#SBATCH --job-name=run_12
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=run_12.out
#SBATCH --error=run_12.err
#SBATCH --qos=long
conda activate qaravan
python rmps_purity.py 50 2 20000 open 12
