#!/bin/bash -l
#SBATCH --job-name=pbc_8
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=pbc_8.out
#SBATCH --error=pbc_8.err
#SBATCH --qos=long
conda activate qaravan
python pbc_8.py
