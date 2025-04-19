#!/bin/bash -l
#SBATCH --job-name=staggered
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2-0:00:00
#SBATCH --no-requeue
#SBATCH --output=staggered.out
#SBATCH --error=staggered.err
#SBATCH --qos=long
conda activate qaravan
python purity_jobs.py 'open staggered' 20 5