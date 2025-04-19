#!/bin/bash -l
#SBATCH --job-name=run_ti_linear
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=run_ti_linear.out
#SBATCH --error=run_ti_linear.err
#SBATCH --qos=long
conda activate qaravan
python rmps_purity.py 30 2 50000 ti linear
