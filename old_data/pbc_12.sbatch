#!/bin/bash -l
#SBATCH --job-name=pbc_12
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=pbc_12.out
#SBATCH --error=pbc_12.err
#SBATCH --qos=long
conda activate qaravan
python pbc_12.py
