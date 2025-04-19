#!/bin/bash -l
#SBATCH --job-name=run_open_uniform_10
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=run_open_uniform_10.out
#SBATCH --error=run_open_uniform_10.err
#SBATCH --qos=long
conda activate qaravan
python rmps_purity.py 30 2 40000 open 10 uniform 1.0
