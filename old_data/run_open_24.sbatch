#!/bin/bash -l
#SBATCH --job-name=run_24
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=run_24.out
#SBATCH --error=run_24.err
#SBATCH --qos=long
conda activate qaravan
python rmps_purity.py 50 2 40000 open 24
