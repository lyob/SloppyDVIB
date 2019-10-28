#!/bin/bash
#SBATCH -J h71g
#SBATCH -p dali
#SBATCH --account=pi-sepalmer
#SBATCH -n 1 # one core
#SBATCH -N 1 # on one node
#SBATCH --cpus-per-task=12
#SBATCH -t 3:00:00
#SBATCH --mem 30G
#SBATCH -o gather_%A_%a.out
#SBATCH -e gather_%A_%a.err
#SBATCH --exclusive

source /home/blyo/.bash_profile
source /home/blyo/.virtualenvs/py350/bin/activate

mkdir /home/blyo/python_tests/additive/h71indep/data-scores/
python -u ../code-gather/gather_scores${SLURM_ARRAY_TASK_ID}.py
