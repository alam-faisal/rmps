#!/bin/bash -l
#SBATCH --job-name=run_ti_32
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=run_ti_32.out
#SBATCH --error=run_ti_32.err
#SBATCH --qos=long
conda activate qaravan
python rmps_purity.py 30 2 40000 ti 32
