#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --time==00:01:00
#SBATCH --partition=sepalmer
#SBATCH --account=pi-sepalmer
#SBATCH --nodes=1
#SBATCH --ntasks=16

module load python/3.7.0

python FIM_data_collection.py