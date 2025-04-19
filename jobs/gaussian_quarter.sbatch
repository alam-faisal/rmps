#!/bin/bash -l
#SBATCH --job-name=quarter
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=quarter.out
#SBATCH --error=quarter.err
#SBATCH --qos=long
conda activate qaravan
python purity_jobs.py 'open even' 20 5 'gaussian' 0.25
