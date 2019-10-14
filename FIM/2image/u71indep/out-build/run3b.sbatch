#!/bin/bash
#SBATCH -J u71b3
#SBATCH -p dali
#SBATCH --account=pi-sepalmer
#SBATCH -n 1 # one core
#SBATCH -N 1 # on one node
#SBATCH --cpus-per-task=2
#SBATCH -t 2:00:00
#SBATCH --mem 15G
#SBATCH -o 3build_%A_%a.out
#SBATCH -e 3build_%A_%a.err
#SBATCH --exclusive

source /home/blyo/.bash_profile
source /home/blyo/.virtualenvs/py350/bin/activate

python -u ../code-build/3build${SLURM_ARRAY_TASK_ID}.py
