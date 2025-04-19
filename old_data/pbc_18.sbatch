#!/bin/bash -l
#SBATCH --job-name=pbc_18
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=pbc_18.out
#SBATCH --error=pbc_18.err
#SBATCH --qos=long
conda activate qaravan
python pbc_18.py
