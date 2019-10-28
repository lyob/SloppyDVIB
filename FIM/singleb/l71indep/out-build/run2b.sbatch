#!/bin/bash
#SBATCH -J l71b2
#SBATCH -p dali
#SBATCH --account=pi-sepalmer
#SBATCH -n 1 # one core
#SBATCH -N 1 # on one node
#SBATCH --cpus-per-task=6
#SBATCH -t 10:00:00
#SBATCH --mem 50G
#SBATCH -o 2build_%A_%a.out
#SBATCH -e 2build_%A_%a.err
#SBATCH --exclusive

source /home/blyo/.bash_profile
source /home/blyo/.virtualenvs/py350/bin/activate

mkdir /dali/sepalmer/blyo/additive/l71indep/
mkdir /dali/sepalmer/blyo/additive/l71indep/data-fim/
python -u ../code-build/2build${SLURM_ARRAY_TASK_ID}.py
