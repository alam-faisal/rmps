#!/bin/bash -l
#SBATCH --job-name=run_periodic_12
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=run_periodic_12.out
#SBATCH --error=run_periodic_12.err
#SBATCH --qos=long
conda activate qaravan
python rmps_purity.py 30 2 40000 periodic 12
