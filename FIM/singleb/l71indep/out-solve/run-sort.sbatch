#!/bin/bash
#SBATCH --job-name=h71sort
#SBATCH --output=sort.out
#SBATCH --error=sort.err
#SBATCH --time=3:00:00
#SBATCH --partition=dali
#SBATCH --account=pi-sepalmer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=55G
#SBATCH --export=ALL
#SBATCH --exclusive

source /home/blyo/.bash_profile
source /home/blyo/.virtualenvs/py350/bin/activate
python -u ../code-solve/sort.py
