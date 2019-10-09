#!/bin/bash
#SBATCH --job-name=h71plot10
#SBATCH --output=plot10.out
#SBATCH --error=plot10.err
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

mkdir /dali/sepalmer/blyo/h71indep/data-plots/
python -u ../code-plot/plotnorm10.py
