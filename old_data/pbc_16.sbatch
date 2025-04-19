#!/bin/bash -l
#SBATCH --job-name=pbc_16
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=pbc_16.out
#SBATCH --error=pbc_16.err
#SBATCH --qos=long
conda activate qaravan
python pbc_16.py
