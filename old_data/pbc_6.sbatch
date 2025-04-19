#!/bin/bash -l
#SBATCH --job-name=pbc_6
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=pbc_6.out
#SBATCH --error=pbc_6.err
#SBATCH --qos=long
conda activate qaravan
python pbc_6.py
