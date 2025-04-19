#!/bin/bash -l
#SBATCH --job-name=pbc_20
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=pbc_20.out
#SBATCH --error=pbc_20.err
#SBATCH --qos=long
conda activate qaravan
python pbc_20.py
