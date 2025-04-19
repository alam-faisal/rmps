#!/bin/bash -l
#SBATCH --job-name=pbc_14
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=pbc_14.out
#SBATCH --error=pbc_14.err
#SBATCH --qos=long
conda activate qaravan
python pbc_14.py
