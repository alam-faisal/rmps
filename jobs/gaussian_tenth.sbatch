#!/bin/bash -l
#SBATCH --job-name=tenth
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=tenth.out
#SBATCH --error=tenth.err
#SBATCH --qos=long
conda activate qaravan
python purity_jobs.py 'open even' 20 5 'gaussian' 0.1
