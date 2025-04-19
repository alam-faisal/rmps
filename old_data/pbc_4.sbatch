#!/bin/bash -l
#SBATCH --job-name=pbc_4
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=pbc_4.out
#SBATCH --error=pbc_4.err
#SBATCH --qos=long
conda activate qaravan
python pbc_4.py
