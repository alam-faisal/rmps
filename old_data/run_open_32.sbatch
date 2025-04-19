#!/bin/bash -l
#SBATCH --job-name=run_32
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=run_32.out
#SBATCH --error=run_32.err
#SBATCH --qos=long
conda activate qaravan
python rmps_purity.py 50 2 50000 open 32
